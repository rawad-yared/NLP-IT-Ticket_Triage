from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

try:
    from app.triage_engine import TicketTriageRuntime
except ModuleNotFoundError:
    from triage_engine import TicketTriageRuntime


BASE_DIR = Path(__file__).resolve().parents[1]

SAMPLE_TICKETS = {
    "VPN Login Failure": (
        "Hello IT team, after the password reset this morning I cannot connect to VPN. "
        "Cisco AnyConnect says authentication failed and times out after 30 seconds. "
        "This blocks access to internal dashboards."
    ),
    "Billing Dispute": (
        "We were billed for 180 seats while our active users are 130. "
        "Please review the invoice and clarify pro-rated charges for this month."
    ),
    "Service Outage": (
        "Our customer portal has been unreachable for 45 minutes with repeated 503 errors. "
        "Orders are failing and support calls are increasing rapidly. Need immediate escalation."
    ),
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
          --bg: #f4efe6;
          --card: #fffdf8;
          --ink: #132029;
          --muted: #4f5f66;
          --accent: #0f766e;
          --accent-2: #c2410c;
          --ring: #d6ccc2;
          --shadow: 0 16px 40px rgba(19, 32, 41, 0.08);
        }

        .stApp {
          background:
            radial-gradient(circle at 8% 12%, rgba(15,118,110,0.15), transparent 30%),
            radial-gradient(circle at 92% 18%, rgba(194,65,12,0.16), transparent 28%),
            linear-gradient(180deg, #f7f1e7 0%, #f4efe6 55%, #efe8dd 100%);
          color: var(--ink);
          font-family: "Manrope", "Segoe UI", sans-serif;
        }

        .hero {
          background: linear-gradient(120deg, rgba(15,118,110,.92), rgba(11,94,88,.92));
          border: 1px solid rgba(255,255,255,.20);
          border-radius: 18px;
          padding: 1.3rem 1.4rem;
          box-shadow: var(--shadow);
          margin-bottom: 1rem;
          color: #f6fffd;
        }

        .hero h1 {
          font-family: "Fraunces", Georgia, serif;
          margin: 0 0 .35rem 0;
          font-size: 1.95rem;
          letter-spacing: 0.2px;
        }

        .hero p {
          margin: 0;
          font-size: .98rem;
          opacity: .95;
        }

        .soft-card {
          background: var(--card);
          border: 1px solid var(--ring);
          border-radius: 16px;
          padding: .95rem 1rem;
          box-shadow: var(--shadow);
        }

        .metric-card {
          background: var(--card);
          border: 1px solid var(--ring);
          border-radius: 14px;
          padding: .85rem 1rem;
          height: 100%;
          box-shadow: var(--shadow);
        }

        .metric-label {
          color: var(--muted);
          font-size: .78rem;
          text-transform: uppercase;
          letter-spacing: .07em;
          margin-bottom: .15rem;
        }

        .metric-value {
          font-size: 1.2rem;
          font-weight: 700;
          color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_runtime() -> TicketTriageRuntime:
    return TicketTriageRuntime(base_dir=BASE_DIR)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_user_guide() -> None:
    """Render the built-in user guide tab."""
    st.header("User Guide")

    st.subheader("Overview")
    st.markdown(
        """
This app automatically triages IT support tickets using NLP models.
Paste a ticket description and the system returns:

| Output | Description |
|--------|-------------|
| **Department** | Predicted routing department (e.g., *Network Operations*, *Billing and Payments*) |
| **Urgency** | Priority level (*low*, *medium*, *high*, *critical*) |
| **Tags** | Key topic phrases extracted from the ticket |
| **Summary** | Optional short abstractive summary (powered by T5-small) |
| **JSON** | Full structured output for downstream integration |
"""
    )

    st.subheader("Step-by-Step Usage")
    st.markdown(
        """
1. **Enter a ticket** -- Type or paste a ticket description into the text area,
   or click one of the sample ticket buttons to load a pre-filled example.
2. **Adjust settings** (optional) -- Use the sidebar to set a Ticket ID,
   change the number of tags, or enable/disable the summary.
3. **Click "Triage Ticket"** -- The models will run inference and display results.
4. **Review results** -- Department, urgency, and tags appear as metric cards.
   The JSON panel shows the full structured output you can copy.
"""
    )

    st.subheader("Interpreting Results")
    st.markdown(
        """
- **Confidence scores** -- Each prediction (department, urgency) includes a
  confidence percentage. Higher values indicate the model is more certain.
  Scores below ~70% may warrant manual review.
- **Department labels** -- Correspond to the routing categories the model was
  trained on (e.g., *Network Operations*, *Software Support*,
  *Billing and Payments*, *Hardware Support*).
- **Urgency levels** -- Ordered from *low* to *critical*. Use these to
  prioritize response queues.
- **Tags** -- Extracted keyphrases that highlight the core topics. Useful for
  search, filtering, and at-a-glance understanding.
- **Summary** -- A concise restatement of the ticket. Enable it in the sidebar
  under "Generate summary". The first run may take a moment while the T5-small
  model downloads.
"""
    )

    st.subheader("Tips for Best Results")
    st.markdown(
        """
- **Be specific** -- Include concrete details like error messages, affected
  systems, and the number of impacted users.
- **Avoid very short input** -- Single-word or very vague descriptions produce
  lower-confidence predictions.
- **Use plain language** -- The models were trained on real IT tickets written
  in natural language, so conversational tone works well.
- **Enable summary for long tickets** -- The summarizer condenses verbose
  descriptions into a one-sentence overview.
"""
    )


def _render_triage_tab() -> None:
    """Render the main triage functionality tab."""
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Ticket Input")
    sample_cols = st.columns(3)
    for (label, text), col in zip(SAMPLE_TICKETS.items(), sample_cols):
        if col.button(label, use_container_width=True):
            st.session_state["ticket_text"] = text

    ticket_text = st.text_area(
        "Paste ticket description",
        key="ticket_text",
        height=210,
        placeholder=(
            "Example: Users in finance cannot access VPN after password reset. "
            "Connection times out and blocks payroll operations."
        ),
    )
    run_btn = st.button("Triage Ticket", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        st.info("Enter ticket text and click `Triage Ticket`.")
        return

    # Read sidebar values from session state / sidebar widgets
    ticket_id = st.session_state.get("_sidebar_ticket_id", "demo_ticket_001")
    top_k_tags = st.session_state.get("_sidebar_top_k_tags", 5)
    include_summary = st.session_state.get("_sidebar_include_summary", False)

    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(
            "Could not initialize models. Make sure local checkpoints exist in "
            "`models/<task>_model/best`, or that the Hugging Face Hub model IDs "
            "are configured (see README for deployment instructions)."
        )
        st.exception(exc)
        return

    with st.spinner("Running triage..."):
        try:
            output = runtime.triage_ticket(
                ticket_text=ticket_text,
                ticket_id=ticket_id,
                top_k_tags=top_k_tags,
                include_summary=include_summary,
            )
        except Exception as exc:
            st.error("Triage failed for this input.")
            st.exception(exc)
            return

    dept = output["department"]
    urg = output["urgency"]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Department", dept["label"])
        st.progress(float(dept["confidence"]))
        st.caption(f"Confidence: {dept['confidence']:.2%}")
    with c2:
        metric_card("Urgency", urg["label"])
        st.progress(float(urg["confidence"]))
        st.caption(f"Confidence: {urg['confidence']:.2%}")
    with c3:
        metric_card("Tags", str(len(output["tags"])))
        st.markdown(" ".join([f"`{tag}`" for tag in output["tags"]]) or "_No tags_")

    out_l, out_r = st.columns([1.05, 1.2])
    with out_l:
        st.subheader("Summary")
        if output["summary"]:
            st.write(output["summary"])
        else:
            st.caption("Summary disabled in sidebar settings.")
    with out_r:
        st.subheader("JSON Output")
        st.code(json.dumps(output, indent=2), language="json")


def app() -> None:
    st.set_page_config(
        page_title="IT Ticket Triage Assistant",
        page_icon="🎫",
        layout="wide",
    )
    inject_styles()

    st.markdown(
        """
        <section class="hero">
          <h1>IT Ticket Triage Assistant</h1>
          <p>Submit one ticket and receive department routing, urgency, tags, and optional summary in JSON.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Inference Settings")
        ticket_id = st.text_input("Ticket ID", value="demo_ticket_001", key="_sidebar_ticket_id")
        top_k_tags = st.slider("Number of tags", min_value=3, max_value=8, value=5, key="_sidebar_top_k_tags")
        include_summary = st.toggle("Generate summary", value=False, key="_sidebar_include_summary")
        st.caption("Summary uses `t5-small`; first run downloads weights if not cached.")
        st.markdown("---")
        st.caption("Model path:")
        st.code(str(BASE_DIR / "models"), language="text")

        st.markdown("---")
        with st.expander("Quick Reference"):
            st.markdown(
                """
**Departments:** Network Operations, Software Support,
Billing and Payments, Hardware Support, and others
from the training set.

**Urgency levels:** low, medium, high, critical

**Tags:** Auto-extracted keyphrases (3-8 configurable).

**Summary:** Toggle on in sidebar. Uses T5-small
abstractive summarization.

**Tip:** Click the *User Guide* tab for full documentation.
"""
            )

    # ── Main area: tabs ──────────────────────────────────────────────
    triage_tab, guide_tab = st.tabs(["Triage", "User Guide"])

    with triage_tab:
        _render_triage_tab()

    with guide_tab:
        _render_user_guide()


if __name__ == "__main__":
    app()
