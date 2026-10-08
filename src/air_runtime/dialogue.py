'Deterministic local runtime dialogue analysis and minimal state transitions.'

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, Sequence

from .models import (
    ClarificationState,
    ConversationControlIntent,
    Commitment,
    DialogueStateDelta,
    DialogueState,
    GroundingFrame,
    OpenLoop,
    ReferenceBinding,
    RecentTurn,
    TopicFrame,
    TurnProposal,
    bounded_text,
    utc_now,
)


@dataclass(frozen=True)
class TurnHints:
    identity_request: bool = False
    capability_request: bool = False
    unknown_request: bool = False
    topic_switch: bool = False
    topic_resume: bool = False
    correction: bool = False
    acknowledgement: bool = False
    follow_up: bool = False
    ambiguous_reference: bool = False
    future_action_candidate: bool = False
    reference_label: str | None = None
    topic_candidate: str | None = None
    correction_topic: str | None = None
    concrete_answer: bool = False
    complex: bool = False
    casual_conversation: bool = False
    presence_check: bool = False
    pause_conversation: bool = False
    explicit_topic: bool = False
    same_turn_reference: bool = False
    reference_target: str | None = None
    comparison_options: tuple[str, ...] = ()
    preference_candidate: str | None = None
    preference_query: bool = False
    control_intent: str = ConversationControlIntent.NONE.value
    low_pressure_social: bool = False
    continue_current: bool = False
    no_shared_context: bool = False
    consequence_request_candidate: bool = False


_PUNCTUATION = " \t\r\n，。！？!?、:：;；,.'\"“”‘’()（）[]【】"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _strip_topic(value: str) -> str | None:
    value = value.strip(_PUNCTUATION)
    value = re.sub(r"^(聊聊|聊|说说|谈谈|谈|一下|一下子|的话|好吗|可以吗|呢|吧)\s*", "", value, flags=re.I)
    value = re.split(r"[，,；;。！？!?]", value, maxsplit=1)[0]
    value = re.sub(r"(?:吧|啊|呢|呀)$", "", value, flags=re.I)
    value = value.strip(_PUNCTUATION)
    if not value:
        return None
    return bounded_text(value, 120, "topic candidate")


def _normalized_phrase(text: str) -> str:
    return _compact(text).strip(_PUNCTUATION)


def _is_casual_conversation_signal(text: str) -> bool:
    """Recognise a small family of social conversation signals.

    This is deliberately structural rather than a list of exact sentences.
    It distinguishes a request to spend time talking from a request to name a
    topic, while still allowing explicit forms such as ``聊聊天气`` through
    the topic extractor below.
    """

    phrase = _normalized_phrase(text)
    patterns = (
        r"^(?:我们)?(?:来)?聊(?:聊|(?:会|一会)儿?(?:天)?|天)?(?:吧|啊|呢|呀)?$",
        r"^随便聊(?:聊)?(?:吧|啊|呢|呀)?$",
    )
    return any(re.fullmatch(pattern, phrase, flags=re.I) for pattern in patterns)


def _is_presence_check(text: str) -> bool:
    phrase = _normalized_phrase(text)
    return phrase in {"在吗", "在么", "在嘛", "在不在", "你在吗", "你在么", "你在嘛"}


def _is_pause_signal(text: str) -> bool:
    phrase = _normalized_phrase(text)
    return bool(
        re.fullmatch(
            r"^(?:算了|算了先(?:不|别)聊(?:这个|这件事|它)?|先(?:不|别)聊(?:这个|这件事|它)?|先不说(?:这个|这件事|它)?了?|暂时不聊(?:这个|这件事|它)?|别聊(?:这个|这件事|它)?)$",
            phrase,
            flags=re.I,
        )
    )


def _is_non_topic_correction(value: str) -> bool:
    phrase = _normalized_phrase(value)
    return phrase in {
        "这个意思",
        "那个意思",
        "这样",
        "那样",
        "刚才说反了",
        "刚刚说反了",
        "我刚才说反了",
        "我刚刚说反了",
        "我说反了",
        "说反了",
        "说错了",
        "我说错了",
    }


def _same_turn_antecedent(text: str) -> str | None:
    """Return a bounded antecedent when a generic reference has local context."""

    if _is_casual_conversation_signal(text) or _is_presence_check(text) or _is_pause_signal(text):
        return None
    match = re.search(
        r"(?:这个|那个)(?=(?:顺序|先后|安排|方案|做法|方向|选择|决定|话题|问题|怎么样|如何|呢|吗|吧|$))",
        text,
        flags=re.I,
    )
    if match is None:
        return None
    prefix = text[: match.start()]
    clauses = [item.strip(_PUNCTUATION) for item in re.split(r"[，,；;。！？!?]", prefix)]
    clauses = [item for item in clauses if item]
    while clauses and re.match(r"^(?:你觉得|你认为|你看|怎么看|你觉得我|你认为我)$", clauses[-1]):
        clauses.pop()
    if not clauses:
        return None
    specific_reference = bool(
        re.match(r"(?:顺序|先后|安排|方案|做法|方向|选择|决定|话题|问题|怎么样|如何|呢|吗|吧)", text[match.end() :], flags=re.I)
    )
    if not specific_reference and len(clauses) < 2:
        return None
    # Keep the two most local meaningful clauses so the binding is useful but
    # never turns the whole user message into durable semantic state.
    clauses = clauses[-2:]
    candidate = "；".join(clauses)
    if len(_normalized_candidate(candidate)) < 4:
        return None
    return bounded_text(candidate, 120, "same-turn reference target")


def _clean_option(value: str) -> str | None:
    value = value.strip(_PUNCTUATION)
    value = re.sub(
        r"^(?:(?:你觉得|你认为)(?:我)?|我)?\s*(?:(?:的)?下一步|接下来|短期内(?:其实)?|短期)?\s*"
        r"(?:是)?\s*(?:做|选|选择)?\s*",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"(?:好不好|好吧|好|呢|吗|嘛)$", "", value, flags=re.I).strip(_PUNCTUATION)
    if not value or value in {"一下", "这个", "那个"}:
        return None
    return bounded_text(value, 120, "comparison option")


def _comparison_options(text: str) -> tuple[str, ...]:
    match = re.search(r"还是", text)
    if match is None:
        return ()
    left_raw = re.split(r"[，,；;。！？!?]", text[: match.start()])[-1]
    right_raw = re.split(r"[，,；;。！？!?]", text[match.end() :])[0]
    left = _clean_option(left_raw)
    right = _clean_option(right_raw)
    if not left or not right or left == right:
        return ()
    return (left, right)


def _option_matching_text(text: str, options: Sequence[str]) -> str | None:
    normalized = _normalized_candidate(text)
    for option in options:
        if _normalized_candidate(option) in normalized:
            return option
    return None


def _preference_candidate(text: str, state: DialogueState, options: Sequence[str]) -> str | None:
    matched = _option_matching_text(text, options)
    if matched and any(cue in _compact(text) for cue in ("倾向", "偏向", "更想", "选择", "选", "想做")):
        return matched
    match = re.search(
        r"(?:更?倾向(?:于|选择)?|偏向(?:于)?|更想(?:做)?|想选|选择|选)\s*([^，。！？!?；;、,]+)",
        text,
        flags=re.I,
    )
    if match:
        candidate = _clean_option(match.group(1))
        if candidate:
            if options:
                return _option_matching_text(candidate, options)
            return candidate
    return None


def _other_option(state: DialogueState) -> str | None:
    if len(state.active_options) != 2 or state.active_preference not in state.active_options:
        return None
    return next(option for option in state.active_options if option != state.active_preference)


def _is_preference_query(text: str) -> bool:
    compact = _compact(text)
    return bool(
        re.search(r"(?:刚才|刚刚|之前)?(?:更?倾向|偏向|选的是|选择的是).*(?:哪个|哪一个|什么)", compact)
        or re.search(r"(?:刚才|刚刚|之前).{0,8}(?:更?倾向|偏向|选|选择).*(?:哪个|哪一个|什么)", compact)
        or re.search(r"(?:哪个|哪一个).*(?:倾向|偏向|选)", compact)
    )


def _topic_from_text(text: str, *, correction: bool = False) -> str | None:
    if _is_casual_conversation_signal(text) or _is_presence_check(text) or _is_pause_signal(text):
        return None
    patterns = (
        r"(?:换个话题|换一个话题)(?:聊聊|聊|说说)?\s*(.*)$",
        r"(?:我们\s*)?(?:来\s*)?(?:随便\s*)?(?:聊聊|聊一聊|聊(?:会儿?|一会儿?|一会)?|说说|谈谈|谈)\s*(?:关于\s*)?(.+)$",
        r"(?:关于)\s*(.+)$",
        r"(?:讨论|谈论)\s*(?:一下|关于)?\s*(.+)$",
        r"(?:let's|lets)\s+(?:talk|chat)\s+about\s+(.+)$",
        r"(?:change|switch)\s+(?:the\s+)?topic(?:\s+to)?\s+(.+)$",
    )
    if correction:
        patterns = (
            r"(?:其实是|我是说|改成|更正[:：]?)\s*(.+)$",
            r"(?:不对|不是)\s*[，,：:]?\s*(?:是|我说的是)?\s*(.+)$",
        ) + patterns
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = _strip_topic(match.group(1))
            if candidate and candidate not in {"一下", "一下吧"} and not _is_non_topic_correction(candidate):
                return candidate
    return None


def analyze_turn(text: str, state: DialogueState, recent_turns: Sequence[RecentTurn] = ()) -> TurnHints:
    """Extract only bounded, user-visible dialogue cues.

    This is intentionally a small heuristic layer.  It gives the Runtime a
    stable policy baseline while leaving a narrow provider boundary available
    for richer wording and future semantic implementations.
    """

    compact = _compact(text)
    casual_signal = _is_casual_conversation_signal(text)
    presence_check = _is_presence_check(text)
    pause_conversation = _is_pause_signal(text)
    has_shared_context = bool(
        state.current_topic
        or state.active_options
        or state.active_preference
        or state.reference_bindings
        or any(item.role == "user" and item.status != "failed" for item in recent_turns)
    )
    claims_prior_shared_context = any(
        cue in compact
        for cue in (
            "我们刚才讨论",
            "我们刚刚讨论",
            "我们之前讨论",
            "我们刚才聊",
            "我们之前聊",
            "刚才那个你还记得吗",
            "之前那个你还记得吗",
        )
    )
    no_shared_context = claims_prior_shared_context and not has_shared_context
    identity = any(
        cue in compact
        for cue in (
            "你是谁",
            "你是人类",
            "你是真人",
            "你是机器人",
            "你的身份",
            "whoareyou",
            "areyouhuman",
            "whatareyou",
        )
    )
    capability = any(
        cue in compact
        for cue in (
            "发消息",
            "发送消息",
            "帮我发",
            "发邮件",
            "发送邮件",
            "提醒我",
            "设置提醒",
            "删除文件",
            "修改文件",
            "调用工具",
            "执行操作",
            "sendamessage",
            "sendanemail",
            "remindme",
            "deleteafile",
            "execute",
            "callatool",
        )
    )
    consequence_request_candidate = capability or bool(
        re.search(
            r"(?:帮我|替我|帮忙|直接帮我|please|helpme|canyou|couldyou|dothis)",
            compact,
            flags=re.I,
        )
    )
    future_action = capability and any(
        cue in compact
        for cue in ("帮我发", "发消息", "发邮件", "提醒我", "设置提醒", "删除文件", "以后", "稍后", "明天", "send", "remind")
    )
    unknown = any(
        cue in compact
        for cue in (
            "当前进展",
            "最新状态",
            "生产状态",
            "现在什么情况",
            "你知道吗",
            "你记得吗",
            "whatsthelatest",
            "currentstatus",
            "doyouknow",
            "doyouremember",
        )
    )
    correction = any(
        cue in compact
        for cue in ("不对", "不是", "更正", "其实是", "我是说", "改成", "应该是", "说反了", "correction")
    )
    comparison_options = _comparison_options(text)
    topic = _topic_from_text(text)
    if no_shared_context:
        topic = None
        comparison_options = ()
    if comparison_options:
        topic = " / ".join(comparison_options)
    correction_topic = _topic_from_text(text, correction=True) if correction else None
    if correction_topic and _is_non_topic_correction(correction_topic):
        correction_topic = None
    same_turn_target = _same_turn_antecedent(text)
    switch_words = any(cue in compact for cue in ("换个话题", "换一个话题", "另外聊", "change the topic", "switch topic"))
    topic_switch = (
        bool(topic and (state.current_topic is None or topic != state.current_topic.topic))
        or switch_words
    ) and not no_shared_context
    resume_previous = any(
        cue in compact
        for cue in ("回到刚才", "回到原来", "回到之前", "上一个话题", "原来的话题", "resume", "go back")
    )
    continue_signal = any(
        cue in compact
        for cue in ("继续刚才", "继续之前", "接着刚才", "接着之前", "继续这个", "continue that")
    )
    resume = resume_previous or (continue_signal and state.current_topic is None and bool(state.topic_stack))
    continue_current = continue_signal and state.current_topic is not None
    label: str | None = None
    for candidate in ("刚才那个", "上一个", "原来的", "第二个", "第2个", "第一个", "这个", "那个", "it", "that"):
        if candidate in text.lower() or candidate in compact:
            label = candidate
            break
    ordinal = bool(label and re.search(r"(?:第\s*[12二一]\s*个|\b[12](?:st|nd)?\b)", text, flags=re.I))
    same_turn_reference = bool(label and same_turn_target)
    ambiguous_reference = bool(
        label
        and (
            ordinal
            or (
                state.clarification.status != "awaiting"
                and state.current_topic is None
                and label in {"这个", "那个", "it", "that"}
                and not pause_conversation
                and not correction
                and not resume
            )
        )
        and not same_turn_reference
        and not no_shared_context
    )
    acknowledgement = compact.strip("。.!！?？") in {
        "嗯",
        "嗯嗯",
        "好",
        "好的",
        "行",
        "收到",
        "继续",
        "ok",
        "okay",
        "yes",
        "好呀",
        "哈哈",
        "哈哈哈",
    }
    follow_up = any(
        cue in compact
        for cue in (
            "为什么",
            "然后呢",
            "接着呢",
            "继续说",
            "再说说",
            "你觉得呢",
            "你怎么看",
            "whatthen",
            "andthen",
            "why",
        )
    ) and not acknowledgement
    preference_query = _is_preference_query(text)
    preference_candidate = _preference_candidate(text, state, comparison_options or state.active_options)
    if preference_query:
        preference_candidate = None
    if correction and not correction_topic and not preference_candidate:
        preference_candidate = _other_option(state)
    concrete_answer = bool(
        state.clarification.status == "awaiting"
        and not topic_switch
        and not resume
        and not identity
        and not capability
        and not ambiguous_reference
        and not acknowledgement
        and len(compact) > 0
    )
    complex_turn = len(text) > 180 or any(
        cue in compact
        for cue in (
            "详细分析",
            "深入分析",
            "系统分析",
            "商业验证路径",
            "比较一下",
            "权衡",
            "取舍",
            "多步骤",
            "分步骤",
            "完整方案",
            "制定方案",
            "拆解一下",
            "规划一下",
            "为什么会",
            "prosandcons",
            "tradeoff",
            "compare",
            "analyze",
            "detailed",
        )
    )
    explicit_topic = (
        topic is not None
        and not casual_signal
        and not presence_check
        and not pause_conversation
        and not no_shared_context
    )
    low_pressure_social = any(
        cue in compact
        for cue in ("有点累", "有些累", "累了", "做项目做得有点久", "赶项目", "没怎么睡好", "睡不好", "唉")
    )
    if pause_conversation:
        control_intent = ConversationControlIntent.PAUSE_CURRENT_TOPIC.value
    elif correction:
        control_intent = ConversationControlIntent.CORRECT.value
    elif resume:
        control_intent = ConversationControlIntent.RESUME_TOPIC.value
    elif switch_words:
        control_intent = ConversationControlIntent.SWITCH_TOPIC.value
    elif acknowledgement:
        control_intent = ConversationControlIntent.ACKNOWLEDGE.value
    elif casual_signal or follow_up or continue_current or preference_query or preference_candidate or explicit_topic or same_turn_reference:
        control_intent = ConversationControlIntent.CONTINUE.value
    else:
        control_intent = ConversationControlIntent.NONE.value
    return TurnHints(
        identity_request=identity,
        capability_request=capability,
        unknown_request=unknown,
        topic_switch=topic_switch,
        topic_resume=resume,
        correction=correction,
        acknowledgement=acknowledgement,
        follow_up=follow_up,
        ambiguous_reference=ambiguous_reference,
        future_action_candidate=future_action,
        reference_label=label,
        topic_candidate=topic,
        correction_topic=correction_topic,
        concrete_answer=concrete_answer,
        complex=complex_turn,
        casual_conversation=(
            casual_signal
            or presence_check
            or pause_conversation
            or acknowledgement
            or follow_up
            or continue_current
            or (correction and not correction_topic)
            or low_pressure_social
        ),
        presence_check=presence_check,
        pause_conversation=pause_conversation,
        explicit_topic=explicit_topic,
        same_turn_reference=same_turn_reference,
        reference_target=same_turn_target,
        comparison_options=comparison_options,
        preference_candidate=preference_candidate,
        preference_query=preference_query,
        control_intent=control_intent,
        low_pressure_social=low_pressure_social,
        continue_current=continue_current,
        no_shared_context=no_shared_context,
        consequence_request_candidate=consequence_request_candidate,
    )


def build_grounding_frame(
    hints: TurnHints, state: DialogueState, recent_turns: Sequence[RecentTurn]
) -> GroundingFrame:
    verified: list[str] = []
    inferred: list[str] = []
    unknowns: list[str] = []
    if state.current_topic:
        verified.append(f"当前话题是：{state.current_topic.topic}")
    if state.active_goal:
        verified.append(f"当前目标是：{state.active_goal}")
    if hints.topic_resume and state.topic_stack:
        inferred.append("用户可能想恢复一个已有话题线程")
    if hints.unknown_request:
        unknowns.append("当前运行时没有该外部状态或事实的已验证来源")
    if hints.no_shared_context:
        unknowns.append("当前 actor/conversation 没有建立用户所指的共同对话历史")
    if hints.capability_request:
        unknowns.append('local runtime 没有外部工具或副作用执行能力')
    refs = tuple(binding.label for binding in state.reference_bindings if binding.status == "resolved")
    current = {
        "conversation_ref": state.conversation_ref,
        "topic": state.current_topic.topic if state.current_topic else "",
        "topic_stack_depth": len(state.topic_stack),
        "goal": state.active_goal or "",
        "active_option_count": len(state.active_options),
        "active_preference_known": bool(state.active_preference),
        "unresolved_reference_count": len(state.unresolved_references),
        "same_turn_reference": hints.same_turn_reference,
        "reference_target_present": bool(hints.reference_target),
        "preference_query": hints.preference_query,
        "control_intent": hints.control_intent,
        "continue_current": hints.continue_current,
        "no_shared_context": hints.no_shared_context,
        "open_loop_count": len([loop for loop in state.open_loops if loop.status == "open"]),
        "recent_turn_count": min(len(recent_turns), 12),
        "clarification": state.clarification.status,
    }
    return GroundingFrame(
        verified_facts=tuple(verified),
        inferred_facts=tuple(inferred),
        unknowns=tuple(unknowns),
        resolved_references=refs,
        current_state=current,
    )


def _replace_at(state: DialogueState, now: str, **changes: object) -> DialogueState:
    values = {
        "conversation_ref": state.conversation_ref,
        "current_topic": state.current_topic,
        "topic_stack": state.topic_stack,
        "active_goal": state.active_goal,
        "active_options": state.active_options,
        "active_preference": state.active_preference,
        "open_loops": state.open_loops,
        "commitments": state.commitments,
        "reference_bindings": state.reference_bindings,
        "unresolved_references": state.unresolved_references,
        "clarification": state.clarification,
        "revision": state.revision + 1,
        "updated_at_utc": now,
    }
    values.update(changes)
    return DialogueState(**values)


def _topic_frame(topic: str, now: str, goal: str | None = None) -> TopicFrame:
    return TopicFrame(topic=topic, goal=goal, entered_at_utc=now, last_active_at_utc=now)


def _close_reference_loops(loops: Iterable[OpenLoop]) -> tuple[OpenLoop, ...]:
    result: list[OpenLoop] = []
    for loop in loops:
        if loop.loop_type == "reference_clarification" and loop.status == "open":
            result.append(replace(loop, status="resolved"))
        else:
            result.append(loop)
    return tuple(result)


@dataclass(frozen=True)
class StateDeltaAdmission:
    """Result of deterministic admission for one proposed state delta."""

    delta: DialogueStateDelta
    accepted: bool
    reason: str


def _normalized_candidate(value: str) -> str:
    return re.sub(r"[\s\u3000，。！？!?、:：;；,.'\"“”‘’()（）\[\]【】]", "", value.lower())


def _candidate_is_grounded(text: str, candidate: str) -> bool:
    normalized_text = _normalized_candidate(text)
    normalized_candidate = _normalized_candidate(candidate)
    return bool(normalized_candidate) and normalized_candidate in normalized_text


def _candidate_is_lexical_fragment(candidate: str) -> bool:
    normalized = _normalized_candidate(candidate)
    if not normalized:
        return True
    if re.fullmatch(r"(?:会|一会|一会儿|会儿)天(?:吧|啊|呢|呀)?", normalized):
        return True
    if normalized in {"一下", "一下吧", "聊", "聊聊", "聊天", "这个", "那个", "它"}:
        return True
    return False


def _legacy_delta(proposal: TurnProposal | None, state: DialogueState) -> DialogueStateDelta | None:
    if proposal is None:
        return None
    if proposal.state_delta is not None:
        return proposal.state_delta
    if proposal.topic_candidate:
        operation = "push_topic" if state.current_topic is not None else "set_topic"
        return DialogueStateDelta(operation=operation, value=proposal.topic_candidate, reason="legacy_candidate")
    if proposal.goal_candidate:
        return DialogueStateDelta(operation="set_goal", value=proposal.goal_candidate, reason="legacy_candidate")
    if proposal.reference_candidate:
        return DialogueStateDelta(operation="bind_reference", value=proposal.reference_candidate, reason="legacy_candidate")
    if proposal.clarification_candidate:
        return DialogueStateDelta(
            operation="update_clarification", value=proposal.clarification_candidate, reason="legacy_candidate"
        )
    if proposal.open_loop_candidate:
        return DialogueStateDelta(operation="add_open_loop", value=proposal.open_loop_candidate, reason="legacy_candidate")
    if proposal.commitment_candidate:
        return DialogueStateDelta(
            operation="add_commitment", value=proposal.commitment_candidate, reason="legacy_candidate"
        )
    return None


def _reject_delta(reason: str) -> StateDeltaAdmission:
    return StateDeltaAdmission(DialogueStateDelta.no_change(reason=reason), False, reason)


def admit_state_delta(
    text: str,
    state: DialogueState,
    hints: TurnHints,
    proposal: TurnProposal | None,
) -> StateDeltaAdmission:
    """Admit only useful, grounded state changes before state construction.

    Deterministic control cues are preferred for topic/reference semantics.
    Provider candidates remain advisory and are ignored for ordinary social,
    acknowledgement, presence and follow-up turns unless a matching explicit
    user cue makes the candidate materially useful.
    """

    delta = _legacy_delta(proposal, state)

    # High-confidence user controls own the basic dialogue transitions.  This
    # keeps a minimal Direct proposal useful even when it omits state_delta.
    if hints.topic_resume and state.topic_stack:
        delta = DialogueStateDelta(operation="resume_topic", reason="user_resume")
    elif hints.correction and hints.preference_candidate:
        delta = DialogueStateDelta(
            operation="set_preference", value=hints.preference_candidate, reason="correction_preference"
        )
    elif hints.correction and hints.correction_topic:
        delta = DialogueStateDelta(
            operation="set_topic", value=hints.correction_topic, reason="correction_supersedes"
        )
    elif hints.preference_candidate:
        delta = DialogueStateDelta(
            operation="set_preference", value=hints.preference_candidate, reason="explicit_preference"
        )
    elif hints.comparison_options and hints.topic_candidate:
        operation = "push_topic" if state.current_topic and state.current_topic.topic != hints.topic_candidate else "set_topic"
        delta = DialogueStateDelta(operation=operation, value=hints.topic_candidate, reason="comparison_topic")
    elif hints.explicit_topic and hints.topic_candidate:
        operation = "push_topic" if state.current_topic and state.current_topic.topic != hints.topic_candidate else "set_topic"
        delta = DialogueStateDelta(operation=operation, value=hints.topic_candidate, reason="explicit_topic")
    elif hints.ambiguous_reference and hints.reference_label:
        delta = DialogueStateDelta(
            operation="update_clarification", value=hints.reference_label, reason="ambiguous_reference"
        )
    elif state.clarification.status == "awaiting" and hints.concrete_answer:
        delta = DialogueStateDelta(
            operation="update_clarification", value="resolve", reason="clarification_answer"
        )
    elif (
        hints.reference_label
        and (state.current_topic or hints.same_turn_reference)
        and not hints.ambiguous_reference
        and not hints.pause_conversation
        and not (hints.correction and not hints.correction_topic)
    ):
        delta = DialogueStateDelta(
            operation="bind_reference",
            value=hints.reference_label,
            target=(state.current_topic.topic if state.current_topic else hints.reference_target),
            reason="resolved_reference",
        )

    if delta is None or delta.operation == "no_change":
        return StateDeltaAdmission(DialogueStateDelta.no_change(reason="no_conversational_mutation"), True, "no_change")

    operation = delta.operation
    value = delta.value
    if operation in {"set_topic", "push_topic"}:
        if not value or _candidate_is_lexical_fragment(value):
            return _reject_delta("topic_candidate_is_lexical_fragment")
        grounded = _candidate_is_grounded(text, value) if text else delta.reason in {
            "explicit_topic",
            "correction_supersedes",
            "comparison_topic",
        }
        if not grounded and delta.reason == "comparison_topic" and hints.comparison_options:
            grounded = all(_candidate_is_grounded(text, option) for option in hints.comparison_options)
        if not grounded:
            return _reject_delta("topic_candidate_not_grounded")
        if hints.casual_conversation or hints.presence_check or hints.pause_conversation:
            return _reject_delta("casual_turn_cannot_mutate_topic")
        if not hints.explicit_topic and not (hints.correction and hints.correction_topic) and delta.reason != "comparison_topic":
            return _reject_delta("topic_change_requires_explicit_user_cue")
        if state.current_topic and state.current_topic.topic == value:
            return StateDeltaAdmission(DialogueStateDelta.no_change(reason="topic_already_current"), True, "topic_already_current")
        return StateDeltaAdmission(delta, True, delta.reason or "topic_admitted")

    if operation == "resume_topic":
        if not hints.topic_resume or not state.topic_stack:
            return _reject_delta("resume_without_user_resume_cue")
        return StateDeltaAdmission(delta, True, delta.reason or "resume_admitted")

    if operation == "set_goal":
        grounded = _candidate_is_grounded(text, value) if text else False
        if not value or not grounded:
            return _reject_delta("goal_candidate_not_grounded")
        if not any(cue in _compact(text) for cue in ("目标", "想要", "希望", "打算", "决定", "要做", "goal")):
            return _reject_delta("goal_requires_explicit_user_intent")
        return StateDeltaAdmission(delta, True, delta.reason or "goal_admitted")

    if operation == "clear_goal":
        if not any(cue in _compact(text) for cue in ("清除目标", "取消目标", "不用这个目标", "目标取消", "clear goal")):
            return _reject_delta("goal_clear_requires_explicit_user_intent")
        return StateDeltaAdmission(delta, True, delta.reason or "goal_clear_admitted")

    if operation in {"bind_reference", "replace_reference"}:
        if not value or not (state.current_topic or hints.same_turn_reference):
            return _reject_delta("reference_without_grounded_antecedent")
        if not hints.reference_label:
            return _reject_delta("reference_requires_anaphoric_user_cue")
        if hints.reference_label and value != hints.reference_label:
            return _reject_delta("reference_label_not_grounded")
        grounded = _candidate_is_grounded(text, value) if text else delta.reason == "resolved_reference"
        if not grounded:
            return _reject_delta("reference_candidate_not_grounded")
        expected_targets = {
            item
            for item in (
                hints.reference_target,
                state.current_topic.topic if state.current_topic else None,
            )
            if item
        }
        if delta.target and delta.target not in expected_targets:
            return _reject_delta("reference_target_not_grounded")
        target = delta.target or hints.reference_target or (state.current_topic.topic if state.current_topic else None)
        if not target:
            return _reject_delta("reference_target_missing")
        return StateDeltaAdmission(
            replace(delta, target=target),
            True,
            delta.reason or "reference_admitted",
        )

    if operation == "set_preference":
        options = hints.comparison_options or state.active_options
        if not value or not options or value not in options:
            return _reject_delta("preference_candidate_not_in_option_set")
        if not hints.preference_candidate or hints.preference_candidate != value:
            return _reject_delta("preference_requires_explicit_user_cue")
        return StateDeltaAdmission(delta, True, delta.reason or "preference_admitted")

    if operation == "update_clarification":
        if hints.ambiguous_reference and value:
            return StateDeltaAdmission(delta, True, delta.reason or "clarification_admitted")
        if value == "resolve" and state.clarification.status == "awaiting" and hints.concrete_answer:
            return StateDeltaAdmission(delta, True, delta.reason or "clarification_resolution_admitted")
        return _reject_delta("clarification_not_grounded")

    if operation == "clear_clarification":
        allowed = state.clarification.status == "awaiting" and (
            hints.topic_switch or hints.topic_resume or hints.correction
        )
        return StateDeltaAdmission(delta, allowed, "clarification_clear" if allowed else "clarification_clear_rejected")

    if operation in {"add_open_loop", "update_open_loop", "remove_open_loop"}:
        if operation == "add_open_loop" and len(state.open_loops) >= 8:
            return _reject_delta("open_loop_bound_reached")
        if operation == "add_open_loop" and (
            not value
            or not (_candidate_is_grounded(text, value) if text else False)
            or not any(cue in _compact(text) for cue in ("以后", "稍后", "下次", "记得", "open loop"))
        ):
            return _reject_delta("open_loop_requires_grounded_future_cue")
        if operation != "add_open_loop" and not delta.target:
            return _reject_delta("open_loop_target_required")
        if operation != "add_open_loop" and not any(
            loop.loop_id == delta.target or loop.loop_type == delta.target for loop in state.open_loops
        ):
            return _reject_delta("open_loop_target_not_found")
        return StateDeltaAdmission(delta, True, delta.reason or "open_loop_admitted")

    if operation in {"add_commitment", "update_commitment", "resolve_commitment"}:
        if operation == "add_commitment" and len(state.commitments) >= 8:
            return _reject_delta("commitment_bound_reached")
        if operation == "add_commitment" and (
            not value
            or not (_candidate_is_grounded(text, value) if text else False)
            or not any(cue in _compact(text) for cue in ("我会", "我将", "答应", "下次", "commit"))
        ):
            return _reject_delta("commitment_requires_explicit_future_cue")
        if operation != "add_commitment" and not delta.target:
            return _reject_delta("commitment_target_required")
        if operation != "add_commitment" and not any(
            commitment.commitment_id == delta.target for commitment in state.commitments
        ):
            return _reject_delta("commitment_target_not_found")
        return StateDeltaAdmission(delta, True, delta.reason or "commitment_admitted")

    return _reject_delta("unsupported_state_delta")


def admit_dialogue_state_delta(
    text: str,
    state: DialogueState,
    hints: TurnHints,
    proposal: TurnProposal | None = None,
) -> StateDeltaAdmission:
    """Named façade for callers that want the semantic admission boundary."""

    return admit_state_delta(text, state, hints, proposal)


def _take_most_salient_topic(stack: list[TopicFrame]) -> TopicFrame | None:
    """Remove the most recently active bounded thread from a topic stack."""

    if not stack:
        return None
    index, topic = max(
        enumerate(stack),
        key=lambda item: (item[1].last_active_at_utc, item[0]),
    )
    stack.pop(index)
    return topic


def apply_dialogue_update(
    state: DialogueState,
    hints: TurnHints,
    proposal: TurnProposal | None,
    *,
    actor_ref: str,
    now: str | None = None,
    admitted_delta: DialogueStateDelta | None = None,
) -> DialogueState:
    """Apply one already-admitted delta and always advance the turn revision."""

    now = now or utc_now()
    admission = (
        StateDeltaAdmission(admitted_delta, True, "pre_admitted")
        if admitted_delta is not None
        else admit_state_delta("", state, hints, proposal)
    )
    delta = admission.delta
    current = state.current_topic
    stack = list(state.topic_stack)
    loops = list(state.open_loops)
    bindings = list(state.reference_bindings)
    unresolved = list(state.unresolved_references)
    clarification = state.clarification
    active_goal = state.active_goal
    active_options = list(state.active_options)
    active_preference = state.active_preference

    if hints.comparison_options:
        active_options = list(hints.comparison_options[:4])
        if active_preference not in active_options:
            active_preference = None

    if delta.operation == "resume_topic" and stack:
        resumed = _take_most_salient_topic(stack)
        assert resumed is not None
        current = replace(resumed, last_active_at_utc=now)
    elif delta.operation == "set_topic" and delta.value:
        # A correction supersedes the old topic rather than creating a second
        # unresolved branch in the topic stack.
        current = _topic_frame(delta.value, now, active_goal)
    elif delta.operation == "push_topic" and delta.value:
        if current is not None and current.topic != delta.value:
            stack.append(replace(current, last_active_at_utc=now))
        current = _topic_frame(delta.value, now, active_goal)
    elif current is not None and any(
        (
            hints.continue_current,
            hints.preference_query,
            hints.preference_candidate is not None,
            hints.correction,
        )
    ):
        current = replace(current, last_active_at_utc=now)

    if delta.operation == "update_clarification" and delta.value and delta.value != "resolve":
        label = delta.value
        bindings = [binding for binding in bindings if binding.reference_key != label]
        bindings.append(
            ReferenceBinding(
                reference_key=label,
                label=label,
                target_ref=None,
                confidence=0.0,
                status="ambiguous",
                created_at_utc=now,
                owner_actor_ref=actor_ref,
                conversation_ref=state.conversation_ref,
            )
        )
        if label not in unresolved:
            unresolved.append(label)
        clarification = ClarificationState(
            status="awaiting",
            question=f"你说的“{label}”具体指哪一项？",
            subject=label,
            attempts=0,
            created_at_utc=now,
        )
        loops = [
            loop
            for loop in loops
            if not (loop.loop_type == "reference_clarification" and loop.status == "open")
        ]
        loops.append(
            OpenLoop(
                loop_id=f"loop-ref-{state.revision + 1}",
                owner_actor_ref=actor_ref,
                conversation_ref=state.conversation_ref,
                loop_type="reference_clarification",
                status="open",
                summary=f"澄清引用：{label}",
                resume_condition="用户说明该引用具体指向",
                created_at_utc=now,
            )
        )
    elif delta.operation == "update_clarification" and delta.value == "resolve" and clarification.status == "awaiting":
        subject = clarification.subject
        if subject:
            bindings = [binding for binding in bindings if binding.reference_key != subject]
            bindings.append(
                ReferenceBinding(
                    reference_key=subject,
                    label=subject,
                    target_ref=bounded_text("当前回答对象", 120, "reference target"),
                    confidence=0.75,
                    status="resolved",
                    created_at_utc=now,
                    owner_actor_ref=actor_ref,
                    conversation_ref=state.conversation_ref,
                )
            )
            unresolved = [item for item in unresolved if item != subject]
        loops = list(_close_reference_loops(loops))
        clarification = ClarificationState.none()
    elif delta.operation in {"bind_reference", "replace_reference"} and delta.value:
        label = delta.value
        bindings = [binding for binding in bindings if binding.reference_key != label]
        target_ref = delta.target or (current.topic if current is not None else None)
        if current is None and target_ref:
            # A same-turn reference can establish the actual subject without
            # promoting the control phrase ("这个") to a topic.
            current = _topic_frame(target_ref, now, active_goal)
        bindings.append(
            ReferenceBinding(
                reference_key=label,
                label=label,
                target_ref=target_ref,
                confidence=0.9,
                status="resolved",
                created_at_utc=now,
                owner_actor_ref=actor_ref,
                conversation_ref=state.conversation_ref,
            )
        )
    elif delta.operation == "clear_clarification":
        clarification = ClarificationState.none()

    if delta.operation == "set_goal" and delta.value:
        active_goal = delta.value
        if current is not None:
            current = replace(current, goal=active_goal, last_active_at_utc=now)
    elif delta.operation == "clear_goal":
        active_goal = None
        if current is not None:
            current = replace(current, goal=None, last_active_at_utc=now)

    if delta.operation == "set_preference" and delta.value:
        active_preference = delta.value

    if delta.operation == "add_open_loop" and delta.value:
        loops.append(
            OpenLoop(
                loop_id=f"loop-dialogue-{state.revision + 1}",
                owner_actor_ref=actor_ref,
                conversation_ref=state.conversation_ref,
                loop_type=delta.target or "dialogue",
                status="open",
                summary=delta.value,
                resume_condition="用户继续说明或明确结束",
                created_at_utc=now,
            )
        )
    elif delta.operation in {"update_open_loop", "remove_open_loop"} and delta.target:
        updated_loops: list[OpenLoop] = []
        for loop in loops:
            matches = loop.loop_id == delta.target or loop.loop_type == delta.target
            if not matches:
                updated_loops.append(loop)
            elif delta.operation == "update_open_loop" and delta.value:
                updated_loops.append(replace(loop, summary=delta.value))
            else:
                updated_loops.append(replace(loop, status="resolved"))
        loops = updated_loops

    if delta.operation == "add_commitment" and delta.value:
        commitments = list(state.commitments)
        commitments.append(
            Commitment(
                commitment_id=f"commitment-dialogue-{state.revision + 1}",
                owner_actor_ref=actor_ref,
                conversation_ref=state.conversation_ref,
                text=delta.value,
                status="active",
                created_at_utc=now,
            )
        )
    elif delta.operation in {"update_commitment", "resolve_commitment"} and delta.target:
        commitments = list(state.commitments)
        for index, commitment in enumerate(commitments):
            if commitment.commitment_id == delta.target:
                if delta.operation == "resolve_commitment":
                    commitments[index] = replace(commitment, status="resolved", resolved_at_utc=now)
                elif delta.value:
                    commitments[index] = replace(commitment, text=delta.value)
    else:
        commitments = list(state.commitments)

    # Switching topics suspends a clarification in a non-modal open loop; it
    # does not erase the user's unresolved reference.
    if delta.operation in {"set_topic", "push_topic", "resume_topic"} and clarification.status == "awaiting":
        clarification = ClarificationState.none()

    stack = stack[-5:]
    loops = loops[-8:]
    bindings = bindings[-12:]
    unresolved = unresolved[-12:]
    return _replace_at(
        state,
        now,
        current_topic=current,
        topic_stack=tuple(stack),
        active_goal=active_goal,
        active_options=tuple(active_options),
        active_preference=active_preference,
        open_loops=tuple(loops),
        commitments=tuple(commitments),
        reference_bindings=tuple(bindings),
        unresolved_references=tuple(unresolved),
        clarification=clarification,
    )
