"""
LLM call instrumentation — token counts, latency and cost per Groq call.

Token counts come from the API response's `usage` field, never from counting
tokens locally. Cost is computed from Groq's published per-model pricing below.

This module records; it does not decide. Nothing here alters a prompt, a model
choice, or any agent's behaviour.

Usage:
    groq_client = instrumented_groq("extractor")
    groq_client.chat.completions.create(...)          # recorded as "extractor"
    groq_client.chat.completions.create(..., _agent="query_sql")   # override
"""
import threading
import time
from dataclasses import dataclass, field

from groq import Groq

from config import Config
from logger import logger

# ── Groq published pricing, USD per 1,000,000 tokens ──────────────────────────
# Source: https://console.groq.com/docs/models  (and /docs/models.md)
# Fetched: 2026-08-29. Cross-checked against
# https://console.groq.com/docs/model/openai/gpt-oss-120b
#
# Verbatim from the pricing table:
#   GPT OSS 120B         $0.15 input   $0.60 output
#   GPT OSS 20B          $0.075 input  $0.30 output
#   Safety GPT OSS 20B   $0.075 input  $0.30 output
#   Qwen/Qwen3.6-27B     $0.60 input   $3.00 output
#   Qwen/Qwen3.8-27B     $0.80 input   $4.00 output
#   Llama 3.1 8B         Contact Sales
#   Llama 3.3 70B        Contact Sales
#   Groq Compound        no price listed
#
# Models absent from this table have no published price. Their cost is recorded
# as None rather than estimated — an unpriced call must not silently become $0.
PRICING_PER_1M = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-safeguard-20b": {"input": 0.075, "output": 0.30},
    "qwen/qwen3.6-27b": {"input": 0.60, "output": 3.00},
    "qwen/qwen3.8-27b": {"input": 0.80, "output": 4.00},
}
PRICING_SOURCE = "https://console.groq.com/docs/models (fetched 2026-08-29)"


@dataclass
class LLMCall:
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    cost_usd: float | None
    ok: bool = True
    error: str | None = None

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


@dataclass
class _Recorder:
    calls: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, call):
        with self._lock:
            self.calls.append(call)

    def reset(self):
        with self._lock:
            self.calls = []

    def snapshot(self):
        with self._lock:
            return list(self.calls)


_recorder = _Recorder()


class CostLimitExceeded(RuntimeError):
    """Raised when a budgeted unit of work exceeds its cost cap."""


class _Budget(threading.local):
    """Per-thread cost budget. Thread-local so concurrent requests don't share."""

    limit_usd = None
    spent_usd = 0.0
    label = None


_budget = _Budget()


class cost_budget:
    """
    Context manager enforcing a per-unit-of-work cost cap.

        with cost_budget(Config.MAX_COST_PER_DOCUMENT_USD, label=doc_id):
            ...pipeline...

    The breaker fails loudly: once spend exceeds the cap, the next LLM call
    raises CostLimitExceeded rather than quietly continuing to spend. Cost is
    only known after a call returns, so the call that crosses the line is
    completed and recorded — the cap bounds the overshoot to one call, it does
    not prevent it.

    A limit of None disables enforcement while still recording spend.
    """

    def __init__(self, limit_usd, label=None):
        self.limit_usd = limit_usd
        self.label = label

    def __enter__(self):
        self._prev = (_budget.limit_usd, _budget.spent_usd, _budget.label)
        _budget.limit_usd = self.limit_usd
        _budget.spent_usd = 0.0
        _budget.label = self.label
        return self

    def __exit__(self, *exc):
        self.spent_usd = _budget.spent_usd
        _budget.limit_usd, _budget.spent_usd, _budget.label = self._prev
        return False

    @property
    def spent(self):
        return _budget.spent_usd


def _check_budget_before_call(agent):
    limit = _budget.limit_usd
    if limit is None:
        return
    if _budget.spent_usd >= limit:
        raise CostLimitExceeded(
            f"Cost cap exceeded before {agent} call: "
            f"${_budget.spent_usd:.6f} spent against a ${limit:.6f} limit"
            + (f" for {_budget.label}" if _budget.label else "")
            + ". Refusing to spend further."
        )


def _record_budget_spend(cost):
    if _budget.limit_usd is not None and cost:
        _budget.spent_usd += cost


def reset():
    """Clear recorded calls. Used between eval documents."""
    _recorder.reset()


def get_calls():
    return _recorder.snapshot()


def cost_of(model, prompt_tokens, completion_tokens):
    """
    USD for one call, or None when the model has no published price.
    Never estimates: an unknown model returns None, not zero.
    """
    price = PRICING_PER_1M.get(model)
    if price is None:
        return None
    return (prompt_tokens / 1_000_000) * price["input"] + (
        completion_tokens / 1_000_000
    ) * price["output"]


def summarise(calls=None):
    """Aggregate recorded calls, overall and per agent."""
    calls = get_calls() if calls is None else calls
    per_agent = {}
    for call in calls:
        bucket = per_agent.setdefault(
            call.agent,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cost_usd": 0.0, "latency_seconds": 0.0, "unpriced": 0, "failures": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += call.prompt_tokens
        bucket["completion_tokens"] += call.completion_tokens
        bucket["latency_seconds"] += call.latency_seconds
        if call.cost_usd is None:
            bucket["unpriced"] += 1
        else:
            bucket["cost_usd"] += call.cost_usd
        if not call.ok:
            bucket["failures"] += 1

    total = {
        "calls": sum(b["calls"] for b in per_agent.values()),
        "prompt_tokens": sum(b["prompt_tokens"] for b in per_agent.values()),
        "completion_tokens": sum(b["completion_tokens"] for b in per_agent.values()),
        "cost_usd": sum(b["cost_usd"] for b in per_agent.values()),
        "latency_seconds": sum(b["latency_seconds"] for b in per_agent.values()),
        "unpriced": sum(b["unpriced"] for b in per_agent.values()),
        "failures": sum(b["failures"] for b in per_agent.values()),
    }
    return {"per_agent": per_agent, "total": total}


class _InstrumentedCompletions:
    def __init__(self, inner, default_agent):
        self._inner = inner
        self._default_agent = default_agent

    def create(self, *args, _agent=None, **kwargs):
        agent = _agent or self._default_agent
        model = kwargs.get("model", "unknown")
        _check_budget_before_call(agent)
        started = time.perf_counter()
        try:
            completion = self._inner.create(*args, **kwargs)
        except Exception as exc:
            # A failed call still consumed time; record it so failures are visible
            # in the cost report rather than vanishing.
            _recorder.add(
                LLMCall(
                    agent=agent,
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_seconds=time.perf_counter() - started,
                    cost_usd=None,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

        latency = time.perf_counter() - started
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        actual_model = getattr(completion, "model", None) or model
        cost = cost_of(actual_model, prompt_tokens, completion_tokens)

        if cost is None:
            logger.warning(
                f"[Metrics] No published price for model {actual_model!r}; "
                f"cost recorded as unpriced. Source: {PRICING_SOURCE}"
            )

        _recorder.add(
            LLMCall(
                agent=agent,
                model=actual_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency,
                cost_usd=cost,
            )
        )
        _record_budget_spend(cost)
        logger.info(
            f"[Metrics] {agent}: {actual_model} "
            f"in={prompt_tokens} out={completion_tokens} "
            f"{latency:.2f}s "
            + (f"${cost:.6f}" if cost is not None else "unpriced")
            + (f" | budget ${_budget.spent_usd:.6f}/${_budget.limit_usd:.6f}"
               if _budget.limit_usd is not None else "")
        )
        return completion


class _InstrumentedChat:
    def __init__(self, inner, default_agent):
        self.completions = _InstrumentedCompletions(inner.completions, default_agent)


class InstrumentedGroq:
    """Thin proxy over a Groq client. Forwards everything, records create()."""

    def __init__(self, agent, client=None):
        self._client = client or Groq(api_key=Config.GROQ_API_KEY)
        self._agent = agent
        self.chat = _InstrumentedChat(self._client.chat, agent)

    def __getattr__(self, name):
        return getattr(self._client, name)


def instrumented_groq(agent):
    return InstrumentedGroq(agent)
