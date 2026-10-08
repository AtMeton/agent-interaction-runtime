"""Protected real-world consequence truth validation.

This module does not decide conversational intent, tone, depth, or state.  It
only checks whether a user-facing completion assertion is supported by the
Runtime's protected execution facts.  The assertion detector is deliberately
action-type agnostic: it recognizes completion/commitment grammar, while the
Runtime facts provide the protected real-world context.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence

from .models import ProtectedFacts


_COMPLETION_MARKER_RE = re.compile(
    r"(?:完成(?:了)?|成功(?:了)?|搞定(?:了)?|弄好(?:了)?|做好(?:了)?|做完(?:了)?|"
    r"处理好(?:了)?|办好(?:了)?|done|successfully|completed|finished|all\s+set)",
    flags=re.IGNORECASE,
)
_CHINESE_PREFIXED_COMPLETION_RE = re.compile(
    r"已(?:经)?(?P<predicate>[\u4e00-\u9fff]{1,48}(?:了)?)(?=$|[\s，。！？；：、,:!?;])"
)
_CHINESE_RESULTATIVE_RE = re.compile(
    r"[\u4e00-\u9fff]{1,16}(?:好|完|掉|出去|出来|下来)(?:了)?"
)
_ENGLISH_BOUND_COMPLETION_RE = re.compile(
    r"(?<![A-Za-z])(?:done|successfully|completed|finished|all\s+set)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_NON_CONSEQUENCE_PREDICATE_RE = re.compile(
    r"(?:明白|理解|懂(?:得)?|知道|知悉|知晓|了解|解释|说明|分析|说|讲|回答|"
    r"总结|澄清|记得|回忆|想清楚|弄清楚|想明白|看懂|听懂|读懂|研究|思考|"
    r"understand|know|explain|analy[sz]e|remember)",
    flags=re.IGNORECASE,
)
_MODAL_OR_HYPOTHETICAL_RE = re.compile(
    r"(?:如果|若|假如|要是|倘若|即使|哪怕|只要|除非|会|将|能|能够|可以|可能|"
    r"应该|希望|打算|准备|计划|愿意|would|will|can|could|might|should|plan\s+to)",
    flags=re.IGNORECASE,
)
_QUOTED_SPAN_RE = re.compile(
    r"“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|\"[^\"]*\"|'[^']*'"
)
_QUALIFIED_NEGATION_RE = re.compile(
    r"(?:没有|没|未|还没|尚未|未能|不能|无法|不支持|不可以|不确定|不能确认|无法确认|"
    r"不能说|无法说|不是|并非|"
    r"no\s+evidence|not\s+confirmed|cannot\s+confirm|can't\s+confirm|"
    r"unable\s+to\s+confirm|didn't|did\s+not|haven't|hasn't|not|never)",
    flags=re.IGNORECASE,
)
_INDUCEMENT_RE = re.compile(
    r"(?:告诉我|跟我说|直接说|直接告诉|就说|假装|装作|就当|当作).{0,24}"
    r"(?:已(?:经)?|完成|成功|搞定|弄好|做好|做完|处理好|办好|already|done|completed|finished)",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(
    r"(?:刚才|刚刚|之前|前面|那个|这件事|上一个).{0,24}"
    r"(?:已(?:经)?|完成|成功|好了|搞定|弄好|做完|好了吗|完成了吗|done|finished)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ConsequenceTruthResult:
    accepted: bool
    claim_detected: bool
    reasons: tuple[str, ...] = ()


class ConsequenceTruthGate:
    """Fail closed on unsupported protected completion claims."""

    @staticmethod
    def evaluate_outcome(
        *, outcome: str, protected_facts: ProtectedFacts
    ) -> ConsequenceTruthResult:
        """Validate a non-text outcome without turning it into execution truth.

        A Main Mind silence choice cannot acknowledge or conceal an external
        action request.  The Runtime may therefore reject that choice and use
        its existing truthful capability boundary, but it never infers a
        successful action from the absence of text.
        """

        if outcome != "silence":
            return ConsequenceTruthResult(accepted=True, claim_detected=False)
        if (
            protected_facts.side_effect_requested
            or protected_facts.side_effect_executed
            or protected_facts.execution_outcome in {
                "succeeded",
                "failed",
                "unknown",
                "contradictory",
            }
            or protected_facts.committed_durable_result
        ):
            return ConsequenceTruthResult(
                accepted=False,
                claim_detected=False,
                reasons=(
                    "consequence_truth_violation",
                    "silence_would_hide_consequence",
                ),
            )
        return ConsequenceTruthResult(accepted=True, claim_detected=False)

    @staticmethod
    def has_outcome_inducement(text: str) -> bool:
        return bool(_INDUCEMENT_RE.search(text))

    @staticmethod
    def has_outcome_follow_up(text: str) -> bool:
        return bool(_FOLLOW_UP_RE.search(text))

    @classmethod
    def _completion_claim(cls, text: str) -> bool:
        """Return whether text makes a committed external-completion claim.

        Chinese does not put whitespace between a subject and a completion
        marker.  The former boundary-based expression therefore missed
        ``我已经创建好...`` while still catching the same marker after a
        comma.  This parser intentionally stays small and deterministic: it
        recognizes resultative/perfective completion grammar, then rejects
        modality, quotation, negation and ordinary discourse/cognition
        predicates.  It does not enumerate external action types.
        """

        cleaned = unicodedata.normalize("NFKC", text).strip()
        if not cleaned or cleaned.endswith(("?", "？")):
            return False

        for sentence in re.split(r"[。！？!?；;\n]+", cleaned):
            sentence = sentence.strip()
            if not sentence or cls._is_hypothetical_sentence(sentence):
                continue
            quoted_spans = tuple(_QUOTED_SPAN_RE.finditer(sentence))
            candidates = [
                *_COMPLETION_MARKER_RE.finditer(sentence),
                *_CHINESE_PREFIXED_COMPLETION_RE.finditer(sentence),
                *_CHINESE_RESULTATIVE_RE.finditer(sentence),
                *_ENGLISH_BOUND_COMPLETION_RE.finditer(sentence),
            ]
            for match in candidates:
                if cls._inside_quote(match.start(), match.end(), quoted_spans):
                    continue
                if cls._candidate_is_qualified(sentence, match):
                    continue
                return True
        return False

    @staticmethod
    def _inside_quote(
        start: int, end: int, quoted_spans: Sequence[re.Match[str]]
    ) -> bool:
        return any(
            start >= span.start() and end <= span.end() for span in quoted_spans
        )

    @staticmethod
    def _is_hypothetical_sentence(sentence: str) -> bool:
        if not re.search(r"(?:如果|若|假如|要是|倘若|即使|哪怕|只要|除非)", sentence):
            return False
        # A conditional clause is not a report of a committed side effect,
        # even when the clause contains words such as “成功” or “完成”.
        return True

    @staticmethod
    def _candidate_is_qualified(sentence: str, match: re.Match[str]) -> bool:
        window_start = max(0, match.start() - 36)
        # Only the subject/modality/predicate side of the candidate is
        # relevant.  Looking past the candidate would qualify a valid claim
        # merely because a later independent clause says that something
        # “会……” happen.
        window_end = match.end()
        window = sentence[window_start:window_end]
        if _QUALIFIED_NEGATION_RE.search(window):
            return True
        if _MODAL_OR_HYPOTHETICAL_RE.search(window):
            return True

        # A completion marker can also describe an internal conversational
        # act.  Keep those ordinary expressions out of the protected
        # consequence grammar without introducing an action-word blacklist.
        if _NON_CONSEQUENCE_PREDICATE_RE.search(window):
            return True
        trailing_clause = re.split(r"[，。！？；：、,:!?;]", sentence[match.end() :], maxsplit=1)[0]
        if _NON_CONSEQUENCE_PREDICATE_RE.search(trailing_clause[:24]):
            return True
        return False

    @staticmethod
    def _evidence_supports_success(facts: ProtectedFacts) -> bool:
        return (
            facts.capability_available
            and facts.side_effect_executed
            and facts.execution_outcome == "succeeded"
            and facts.postcondition_validated
            and facts.committed_durable_result
            and facts.evidence_source == "runtime_executor"
        )

    @classmethod
    def evaluate(
        cls, *, text: str, protected_facts: ProtectedFacts
    ) -> ConsequenceTruthResult:
        claim_detected = cls._completion_claim(text)
        if not claim_detected:
            return ConsequenceTruthResult(accepted=True, claim_detected=False)

        protected_context = protected_facts.side_effect_requested or protected_facts.execution_outcome in {
            "not_executed",
            "failed",
            "unknown",
            "contradictory",
        }
        if not protected_context:
            return ConsequenceTruthResult(accepted=True, claim_detected=True)
        if cls._evidence_supports_success(protected_facts):
            return ConsequenceTruthResult(accepted=True, claim_detected=True)

        detail = (
            "evidence_contradicts_success_claim"
            if protected_facts.execution_outcome in {"failed", "contradictory"}
            else "success_claim_without_committed_evidence"
        )
        return ConsequenceTruthResult(
            accepted=False,
            claim_detected=True,
            reasons=("consequence_truth_violation", detail),
        )
