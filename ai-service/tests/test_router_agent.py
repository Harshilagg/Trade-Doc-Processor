"""
Router Agent tests — decision precedence only.

_apply_decision_rules is pure and LLM-free; the Groq call in route_decision
generates explanation text after the decision is already made. The reasoning
text is not under test. No test here makes a network call.
"""
import pytest

from services.router_agent import (
    DECISION_AMEND,
    DECISION_APPROVE,
    DECISION_REVIEW,
    _apply_decision_rules,
    route_decision,
)


def summary(mismatch=0, uncertain=0, match=0, total=8):
    return {
        "mismatch_count": mismatch,
        "uncertain_count": uncertain,
        "match_count": match,
        "total_fields": total,
    }


class TestDecisionPrecedence:
    def test_all_match_auto_approves(self):
        assert _apply_decision_rules(summary(match=8)) == (DECISION_APPROVE, 0.97)

    def test_mismatch_beats_uncertain(self):
        """Both present: amendment wins even though uncertain_count is higher."""
        decision, confidence = _apply_decision_rules(
            summary(mismatch=1, uncertain=3, match=4)
        )
        assert decision == DECISION_AMEND
        assert confidence == 0.90

    def test_uncertain_beats_all_match(self):
        decision, confidence = _apply_decision_rules(summary(uncertain=1, match=7))
        assert decision == DECISION_REVIEW
        assert confidence == 0.80

    def test_mismatch_wins_at_realistic_magnitudes(self):
        decision, confidence = _apply_decision_rules(
            summary(mismatch=2, uncertain=2, match=4)
        )
        assert decision == DECISION_AMEND
        assert confidence == 0.85

    def test_partial_match_count_does_not_auto_approve(self):
        """match_count < total with no mismatch/uncertain falls to the edge case."""
        assert _apply_decision_rules(summary(match=5)) == (DECISION_REVIEW, 0.50)

    def test_empty_results_go_to_review(self):
        # The auto-approve branch is guarded by `and total > 0`, so 0 == 0 here
        # must not be read as "everything matched".
        assert _apply_decision_rules(summary(total=0)) == (DECISION_REVIEW, 0.50)


class TestConfidenceFloors:
    """Both sides of each floor, so a live clamp is distinguishable from a
    formula that merely happens to equal the floor."""

    @pytest.mark.parametrize(
        "mismatch_count,expected",
        [(5, 0.70), (6, 0.70), (10, 0.70)],
    )
    def test_amendment_confidence_floor(self, mismatch_count, expected):
        # Formula 0.95 - n*0.05 reaches 0.70 at n=5 and would go below after.
        _, confidence = _apply_decision_rules(summary(mismatch=mismatch_count))
        assert confidence == expected

    def test_amendment_confidence_above_floor_still_varies(self):
        assert _apply_decision_rules(summary(mismatch=1))[1] == 0.90
        assert _apply_decision_rules(summary(mismatch=2))[1] == 0.85

    @pytest.mark.parametrize(
        "uncertain_count,expected",
        [(5, 0.60), (6, 0.60), (10, 0.60)],
    )
    def test_review_confidence_floor(self, uncertain_count, expected):
        # Formula 0.85 - n*0.05 reaches 0.60 at n=5 and would go below after.
        _, confidence = _apply_decision_rules(summary(uncertain=uncertain_count))
        assert confidence == expected


class TestRouteDecisionWrapperIsLlmIndependent:
    """The deterministic decision must survive total LLM failure."""

    def test_decision_unchanged_when_groq_call_raises(self, monkeypatch):
        import services.router_agent as router_agent

        def _explode(*args, **kwargs):
            raise RuntimeError("groq unavailable")

        monkeypatch.setattr(
            router_agent.groq_client.chat.completions, "create", _explode
        )

        result = route_decision(
            {
                "summary": summary(mismatch=1, uncertain=3, match=4),
                "field_results": {
                    "incoterms": {
                        "status": "mismatch",
                        "expected": "FOB | CIF",
                        "found": "EXW",
                        "confidence": 0.95,
                        "reason": "test",
                    }
                },
                "customer_name": "Nike",
            }
        )

        assert result["decision"] == DECISION_AMEND
        assert result["confidence"] == 0.90
        assert result["mismatch_count"] == 1
        # Fallback text is produced rather than left empty, but its wording is
        # not under test.
        assert result["reason"] != ""
