"""
Tests for LLM cost instrumentation and the per-document circuit breaker.

No network: the Groq client is replaced with a stub that returns canned usage.
"""
import pytest

from utils import llm_metrics
from utils.llm_metrics import (
    CostLimitExceeded,
    InstrumentedGroq,
    cost_budget,
    cost_of,
)


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeCompletion:
    def __init__(self, model, prompt_tokens, completion_tokens):
        self.model = model
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class FakeCompletions:
    def __init__(self, model="openai/gpt-oss-120b", prompt=1000, completion=1000):
        self.model, self.prompt, self.completion = model, prompt, completion
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeCompletion(
            kwargs.get("model", self.model), self.prompt, self.completion
        )


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


def make_client(agent="extractor", **kw):
    fake = FakeCompletions(**kw)
    return InstrumentedGroq(agent, client=FakeClient(fake)), fake


@pytest.fixture(autouse=True)
def _clean_metrics():
    llm_metrics.reset()
    yield
    llm_metrics.reset()


class TestCostCalculation:
    def test_uses_published_prices(self):
        # gpt-oss-120b: $0.15/1M input, $0.60/1M output.
        assert cost_of("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)
        assert cost_of("openai/gpt-oss-120b", 0, 1_000_000) == pytest.approx(0.60)

    def test_small_model_is_half_the_large_one(self):
        big = cost_of("openai/gpt-oss-120b", 1000, 1000)
        small = cost_of("openai/gpt-oss-20b", 1000, 1000)
        assert small == pytest.approx(big / 2)

    def test_unpriced_model_returns_none_not_zero(self):
        """An unknown model must never look free."""
        assert cost_of("llama-3.3-70b-versatile", 1000, 1000) is None
        assert cost_of("some-future-model", 1000, 1000) is None


class TestRecording:
    def test_tokens_come_from_the_response_usage(self):
        client, _ = make_client(prompt=123, completion=456)
        client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])

        call = llm_metrics.get_calls()[0]
        assert call.prompt_tokens == 123
        assert call.completion_tokens == 456
        assert call.total_tokens == 579
        assert call.cost_usd == pytest.approx(cost_of("openai/gpt-oss-120b", 123, 456))

    def test_agent_label_defaults_and_overrides(self):
        client, _ = make_client(agent="query")
        client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=[], _agent="query_sql"
        )

        assert [c.agent for c in llm_metrics.get_calls()] == ["query", "query_sql"]

    def test_agent_kwarg_is_not_forwarded_to_groq(self):
        """_agent is instrumentation-only and must not reach the API call."""
        client, fake = make_client()
        captured = {}
        fake.create = lambda **kw: (captured.update(kw), FakeCompletion(
            "openai/gpt-oss-120b", 1, 1))[1]

        client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=[], _agent="router_reasoning"
        )
        assert "_agent" not in captured

    def test_failed_call_is_recorded_then_reraised(self):
        client, fake = make_client()

        def explode(**kwargs):
            raise RuntimeError("groq down")

        fake.create = explode
        with pytest.raises(RuntimeError, match="groq down"):
            client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])

        call = llm_metrics.get_calls()[0]
        assert call.ok is False
        assert "groq down" in call.error

    def test_summarise_splits_by_agent(self):
        client, _ = make_client(agent="extractor", prompt=100, completion=100)
        client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        client.chat.completions.create(
            model="openai/gpt-oss-20b", messages=[], _agent="router_reasoning"
        )

        summary = llm_metrics.summarise()
        assert set(summary["per_agent"]) == {"extractor", "router_reasoning"}
        assert summary["total"]["calls"] == 2
        assert summary["total"]["prompt_tokens"] == 200


class TestCircuitBreaker:
    def test_calls_proceed_under_the_cap(self):
        client, fake = make_client(prompt=1000, completion=1000)
        with cost_budget(1.0):
            for _ in range(3):
                client.chat.completions.create(
                    model="openai/gpt-oss-120b", messages=[]
                )
        assert fake.calls == 3

    def test_breaker_trips_loudly_once_the_cap_is_passed(self):
        # One call at 1M/1M tokens costs $0.75, well over a $0.10 cap.
        client, fake = make_client(prompt=1_000_000, completion=1_000_000)
        with pytest.raises(CostLimitExceeded, match="Refusing to spend further"):
            with cost_budget(0.10, label="doc-42"):
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])

        # The breach is detected before the second call is made, so spend is
        # bounded to the one call that crossed the line.
        assert fake.calls == 1

    def test_error_names_the_label_and_the_numbers(self):
        client, _ = make_client(prompt=1_000_000, completion=1_000_000)
        with pytest.raises(CostLimitExceeded) as exc:
            with cost_budget(0.10, label="doc-42"):
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        message = str(exc.value)
        assert "doc-42" in message
        assert "0.10" in message

    def test_no_limit_disables_enforcement_but_still_records(self):
        client, fake = make_client(prompt=1_000_000, completion=1_000_000)
        with cost_budget(None):
            client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
            client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        assert fake.calls == 2
        assert llm_metrics.summarise()["total"]["calls"] == 2

    def test_budget_does_not_leak_outside_the_context(self):
        client, fake = make_client(prompt=1_000_000, completion=1_000_000)
        with pytest.raises(CostLimitExceeded):
            with cost_budget(0.10):
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
                client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])

        # Inside the budget only the first call was made; the second was blocked
        # before reaching the API. Outside the context there is no cap, so calls
        # proceed again.
        client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        assert fake.calls == 1 + 1

    def test_spend_is_readable_after_the_context_exits(self):
        client, _ = make_client(prompt=1000, completion=1000)
        with cost_budget(1.0) as budget:
            client.chat.completions.create(model="openai/gpt-oss-120b", messages=[])
        assert budget.spent_usd == pytest.approx(
            cost_of("openai/gpt-oss-120b", 1000, 1000)
        )

    def test_configured_cap_leaves_headroom_over_the_measured_baseline(self):
        """
        The cap is a runaway breaker, not a tuning knob: a normal document
        (~$0.0008, docs/COST.md) must sit far below it.
        """
        from config import Config

        assert Config.MAX_COST_PER_DOCUMENT_USD > 0.0008 * 10
