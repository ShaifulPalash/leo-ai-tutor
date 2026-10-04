"""Leo AI Tutor - Streamlit UI with a live Agent Activity panel.

Run:  streamlit run app.py
"""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

# set_page_config must be the FIRST Streamlit call.
st.set_page_config(page_title="Leo - Multi-Agent AI Tutor", page_icon="🦁", layout="wide")

from leo.cli import offline_components  # noqa: E402
from leo.config import get_settings  # noqa: E402
from leo.crew import ICONS, LeoSession, TurnResult  # noqa: E402
from leo.logging_setup import setup_logging  # noqa: E402

setup_logging()
settings = get_settings()

# Display info for the status board (kept here so the UI starts without loading CrewAI).
AGENTS = [
    ("coordinator", "🧭", "Coordinator"),
    ("explainer", "📖", "Explainer"),
    ("quiz_master", "📝", "Quiz Master"),
    ("evaluator", "✅", "Evaluator"),
]
BADGES = {
    "idle": "⚪ idle",
    "working": "🟡 working…",
    "done": "🟢 done",
    "cached": "🔵 cached",
    "failed": "🔴 failed",
    "stopped": "🟠 stopped",
}


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def init_state() -> None:
    """Create the persistent variables on the first run only."""
    for key, default in {
        "session": None,
        "messages": [],
        "flash": [],
        "offline": False,
        "quiz_ref": None,
        "quiz_serial": 0,
    }.items():
        st.session_state.setdefault(key, default)


def start_session(name: str, offline: bool) -> None:
    """Create a fresh LeoSession (offline mode uses the scripted runner)."""
    if offline:
        memory, runner = offline_components()
        session = LeoSession(name, memory=memory, runner=runner)
    else:
        session = LeoSession(name)
    st.session_state.session = session
    st.session_state.offline = offline
    st.session_state.messages = [{"role": "assistant", "content": session.greeting()}]
    st.session_state.flash = []
    st.session_state.quiz_ref = None


# ----------------------------------------------------------------------
# Agent Activity panel (re-drawn live while agents work)
# ----------------------------------------------------------------------
def agent_states(events: list, final: bool) -> dict[str, tuple[str, str, str]]:
    """Latest (status, detail, model) per agent, derived from the event list."""
    states = {key: ("idle", "", "") for key, _, _ in AGENTS}
    for event in events:
        if event.agent in states and event.status in ("working", "done", "cached", "failed"):
            states[event.agent] = (event.status, event.detail, event.model)
    if final:  # an agent still 'working' after the turn ended did not finish
        states = {
            k: (("stopped", d, m) if s == "working" else (s, d, m))
            for k, (s, d, m) in states.items()
        }
    return states


def render_activity(slot, session: LeoSession, final: bool = True) -> None:
    """Draw the whole panel into `slot`. Calling it again replaces the old drawing."""
    with slot.container():
        st.markdown("### 🤖 Agent Activity")
        states = agent_states(session.events, final)
        for key, emoji, name in AGENTS:
            status, detail, model = states[key]
            st.markdown(f"{emoji} **{name}** — {BADGES[status]}")
            if detail:
                text = detail.replace("$", "\\$")
                st.caption(text + (f" · `{model}`" if model else ""))

        st.markdown("**Timeline** (newest first)")
        recent = session.events[-10:][::-1]
        if not recent:
            st.caption("Nothing yet. Ask Leo to teach you something!")
        for event in recent:
            if event.status == "handoff":
                st.markdown(f"`{event.time}` 🔀 **{event.agent} → {event.to_agent}**")
            else:
                text = event.detail.replace("$", "\\$")
                st.markdown(
                    f"`{event.time}` {ICONS.get(event.agent, '•')} "
                    f"**{event.agent}** · {event.status} · {text}"
                )

        handoffs = [e for e in session.events if e.status == "handoff"][-4:]
        if handoffs:
            st.markdown("**Latest handoffs** (what was passed on)")
            for event in reversed(handoffs):
                with st.expander(f"{event.agent} → {event.to_agent} · {event.time}"):
                    st.json(event.payload)

        stats = session.stats
        st.markdown("**Usage this session**")
        st.markdown(
            "| LLM calls | Tokens | LLM time | Cache hits |\n|---|---|---|---|\n"
            f"| {stats['llm_calls']} | {stats['tokens']:,} | {stats['latency_s']:.1f}s "
            f"| {stats['cache_hits']} |"
        )
        if stats["fallbacks"] or stats["safe_defaults"]:
            st.caption(
                f"⚠️ fallback model used {stats['fallbacks']}×, "
                f"safe default used {stats['safe_defaults']}×"
            )


def run_turn(
    slot, session: LeoSession, action: Callable[[], TurnResult], user_text: str = ""
) -> None:
    """Run one session action with the live panel attached, then refresh the page."""
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
    session.on_event = lambda event: render_activity(slot, session, final=False)
    try:
        with st.spinner(
            "Leo's team is working… (the first request can take ~40s " "while CrewAI loads)"
        ):
            result = action()
    finally:
        session.on_event = None  # never keep a callback to a stale placeholder

    if result.reply:
        st.session_state.messages.append({"role": "assistant", "content": result.reply})
    for note in result.notes:
        st.session_state.flash.append(("info", note))
    if result.reteaching:
        st.session_state.flash.append(
            (
                "warning",
                f"Score below {round(session.settings.score_threshold * 100)}%: "
                f"Leo is re-teaching your weak concepts (round {result.round}/"
                f"{session.settings.max_relearn_rounds}), then a mini quiz.",
            )
        )
    st.rerun()  # redraw the page with the new state


# ----------------------------------------------------------------------
# Left-column pieces
# ----------------------------------------------------------------------
def show_flash() -> None:
    """Show one-time messages (notes, warnings), then clear them."""
    for kind, text in st.session_state.flash:
        {"info": st.info, "warning": st.warning, "error": st.error, "success": st.success}.get(
            kind, st.info
        )(text)
    st.session_state.flash = []


def show_lesson(session: LeoSession) -> None:
    lesson = session.explanation
    if lesson is None:
        return
    title = f"📖 {lesson.topic or session.topic} · {session.level} · {session.style} style"

    def body() -> None:
        st.write(lesson.explanation)
        st.markdown("**Key points**")
        for point in lesson.key_points:
            st.markdown(f"- {point}")
        if lesson.example:
            st.info(f"💡 {lesson.example}")

    if session.stage == "lesson":
        with st.container(border=True):
            st.markdown(f"#### {title}")
            body()
    else:  # during a quiz, keep the lesson tucked away
        with st.expander(title):
            body()


def show_report(session: LeoSession) -> None:
    report = session.last_report
    if report is None:
        return
    grade = report.grade
    pct = round(grade.score * 100)
    with st.container(border=True):
        st.markdown(
            f"#### 📊 Latest result: **{pct}%** ({grade.correct_count}/{grade.total})"
            f" — {'✅ passed' if report.passed else '🔁 needs another look'}"
        )
        st.progress(grade.score)
        st.write(report.feedback.overall)
        for tip in report.feedback.tips:
            st.markdown(f"- **{tip.concept}:** {tip.tip}")
        if grade.strong_concepts:
            st.caption("Strong: " + ", ".join(grade.strong_concepts))


def show_quiz(slot, session: LeoSession) -> None:
    quiz = session.quiz
    if quiz is None:
        return
    if st.session_state.quiz_ref is not quiz:  # a new quiz -> fresh widget keys
        st.session_state.quiz_ref = quiz  # (holding the reference keeps `is` safe)
        st.session_state.quiz_serial += 1
    serial = st.session_state.quiz_serial

    with st.form(f"quiz_form_{serial}"):
        st.markdown(
            f"#### 📝 Quiz: {quiz.topic or session.topic} " f"({len(quiz.questions)} questions)"
        )
        choices: dict[int, Optional[int]] = {}
        for q in quiz.questions:
            choices[q.id] = st.radio(
                f"**{q.id}. {q.question}**",
                options=[0, 1, 2, 3],
                index=None,
                format_func=lambda i, q=q: f"{'ABCD'[i]}) {q.options[i]}",
                key=f"quiz{serial}_q{q.id}",
            )
        submitted = st.form_submit_button("✅ Submit answers", type="primary")

    if submitted:
        answers = {qid: choice for qid, choice in choices.items() if choice is not None}
        if not answers:
            st.warning(
                "Pick at least one answer first. You can also ask Leo for "
                "something else in the chat box below."
            )
        else:
            run_turn(slot, session, lambda: session.submit_answers(answers))


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
init_state()
if st.session_state.session is None:
    start_session("Student", offline=False)  # auto-start; no API call is made here
session: LeoSession = st.session_state.session

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🦁 Leo controls")
    st.text_input("Your name", value="Student", max_chars=40, key="name_input")
    st.toggle(
        "Offline demo mode (no API calls)",
        value=st.session_state.offline,
        key="offline_toggle",
        help="Scripted answers about Fractions. Rehearse the UI without using "
        "any free-tier quota.",
    )
    if st.button("▶️ Start / reset session", type="primary"):
        start_session(
            st.session_state.name_input.strip() or "Student", st.session_state.offline_toggle
        )
        st.rerun()

    st.divider()
    st.markdown("**🧠 Memory**")
    st.caption(session.profile_summary())
    if st.button("🗑️ Forget me"):
        session.memory.forget_student(session.student)
        st.session_state.flash.append(("success", "Your stored memory was deleted."))
        st.rerun()

    with st.expander("⚙️ Settings (read-only)"):
        st.json(settings.safe_summary())  # never contains API keys
    if session.trace.path.exists():
        st.download_button(
            "⬇️ Download run trace (JSON)",
            data=session.trace.path.read_text(encoding="utf-8"),
            file_name=session.trace.path.name,
            mime="application/json",
        )
    if settings.debug_mode:
        st.warning("DEBUG_MODE is on: prompts and raw outputs are saved in the trace.")

# ---- Header ----
mode_label = "🧪 Offline demo mode" if st.session_state.offline else "🌐 Live mode (free-tier LLMs)"
st.markdown(
    "<div style='padding:16px 22px;border-radius:14px;"
    "background:linear-gradient(90deg,#6C5CE7,#00B894);color:white;'>"
    "<div style='font-size:2rem;font-weight:700'>🦁 Leo</div>"
    "<div>Multi-agent AI tutor · 🧭 Coordinator → 📖 Explainer → 📝 Quiz Master → "
    f"✅ Evaluator &nbsp;|&nbsp; {mode_label}</div></div>",
    unsafe_allow_html=True,
)
st.write("")

if not st.session_state.offline and not (
    settings.model_chain("fast") and settings.model_chain("smart")
):
    st.error(
        "No usable API keys found. Add GROQ_API_KEY / GEMINI_API_KEY to `.env`, "
        "or switch on **Offline demo mode** in the sidebar and press Start."
    )
    st.stop()

# ---- Toolbar: human-in-the-loop quick actions (work at any stage) ----
has_topic = bool(session.topic)
toolbar = st.columns(4)
quick_request: Optional[str] = None
if toolbar[0].button("🪄 Explain simpler", disabled=not has_topic):
    quick_request = "explain simpler"
if toolbar[1].button("⏭️ Skip to quiz", disabled=not has_topic):
    quick_request = "skip to the quiz"
if toolbar[2].button("🔥 Harder quiz", disabled=not has_topic):
    quick_request = "give me a harder quiz"
if toolbar[3].button("🔄 New topic"):
    quick_request = "change topic"

# ---- Two-column body ----
left, right = st.columns([3, 2])
activity_slot = right.empty()
render_activity(activity_slot, session, final=True)  # static draw before any action

if quick_request:
    run_turn(
        activity_slot,
        session,
        lambda: session.handle_message(quick_request),
        user_text=quick_request,
    )

with left:
    show_flash()
    for message in st.session_state.messages[-6:]:
        avatar = "🦁" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    if not session.topic:
        st.markdown("**Try a topic:**")
        for sample in ("Fractions", "Photosynthesis", "Python loops"):
            if st.button(f"📚 {sample}", key=f"sample_{sample}"):
                run_turn(
                    activity_slot,
                    session,
                    lambda s=sample: session.handle_message(f"teach me {s}"),
                    user_text=f"teach me {sample}",
                )

    show_report(session)
    show_lesson(session)

    if session.stage == "lesson":
        if st.button("📝 Start quiz", type="primary", key="start_quiz"):
            run_turn(activity_slot, session, lambda: session.start_quiz())
    elif session.stage == "quiz":
        show_quiz(activity_slot, session)
    elif session.stage == "done":
        st.success("Round complete! Type a new topic below to keep learning.")

if settings.debug_mode:
    with st.expander("🐞 Debug: last 20 events"):
        st.json([e.as_dict() for e in session.events[-20:]])
st.caption(
    "Free-tier LLMs have per-minute limits. If Leo says the team is busy, "
    "wait a minute and try again. Your progress is saved."
)

# ---- Chat input: type a topic OR intervene mid-run ("explain simpler", "skip to quiz") ----
user_input = st.chat_input("Type a topic, or tell Leo what you need…")
if user_input:
    run_turn(
        activity_slot, session, lambda: session.handle_message(user_input), user_text=user_input
    )
