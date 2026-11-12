"""Basic Streamlit interface. Run: python -m streamlit run app.py"""
import streamlit as st
from src.agent import CHECKS, update_evidence
from src.report import make_report
from src.service import assess_offer

st.set_page_config(page_title="InternShield", page_icon="🛡️")
st.title("InternShield")
st.write("An explainable internship and job-offer risk assessment.")
st.caption("Educational prototype • English text • No live verification • No offer storage")
st.info("Use fictional examples. Do not paste identity documents, passwords, OTPs or bank details.")

choices = {"Unknown": None, "Yes": True, "No": False}
with st.form("offer_form"):
    text = st.text_area("Offer text", height=160)
    company = st.text_input("Company name (context only; not automatically verified)")
    email = st.text_input("Recruiter email (optional)")
    fee = st.selectbox("Any upfront fee requested?", list(choices))
    interview = st.selectbox("Did an interview occur?", list(choices))
    submitted = st.form_submit_button("Analyse offer")

if submitted:
    # Clear old output first: an invalid new submission must not retain old results.
    st.session_state.pop("offer", None)
    st.session_state.pop("assessment", None)
    if len(text.strip()) < 30:
        st.error("Enter at least 30 characters of meaningful English offer text.")
    elif email and (email.count("@") != 1 or not all(email.split("@"))):
        st.error("Enter a valid recruiter email or leave it blank.")
    else:
        offer = {"offer_text": text, "company_name": company, "recruiter_email": email}
        evidence = {}
        if choices[fee] is not None:
            evidence["asks_for_fee"] = choices[fee]
        if choices[interview] is not None:
            evidence["interview_occurred"] = choices[interview]
        try:
            assessment = assess_offer(offer, evidence)
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            st.session_state.update(offer=offer, evidence=evidence, history=[], assessment=assessment)

if "assessment" in st.session_state:
    result = st.session_state.assessment
    risk = result["risk"]
    st.subheader(f"{risk['risk_level'].title()} risk · {risk['risk_score']}/100")
    st.write(risk["explanation"])
    a, b, c = st.columns(3)
    a.metric("Model fraud-class score", f"{result['model']['fraud_score']:.1%}")
    b.metric("Rule severity", f"{result['rules']['rule_score']}/100")
    c.metric("Known evidence", f"{result['evidence_completeness']:.0%}")
    st.caption("Model class names describe resemblance to dataset labels; they do not verify the offer.")
    st.subheader("Warning explanations")
    if result["rules"]["flags"]:
        for flag in result["rules"]["flags"]:
            st.write(f"{flag['rule_id']} · {flag['severity'].upper()} · {flag['reason']}")
    else:
        st.write("No configured warning rule fired. This does not establish safety.")
    st.subheader("Next manual verification")
    agent = result["agent"]
    if agent["check_id"] is None:
        st.write(agent["reason"])
    else:
        check_id = agent["check_id"]
        st.write(agent["reason"])
        st.caption(f"UCS plan: {' → '.join(agent['plan'])} · total effort {agent['plan_cost']} units. "
                   "Costs are illustrative effort units, not time or information gain.")
        with st.form(f"check_{check_id}_{len(st.session_state.history)}"):
            answer = st.selectbox(CHECKS[check_id]["question"], ["Not checked yet", "Yes", "No", "Unable to determine"])
            recorded = st.form_submit_button("Record manual check")
        if recorded:
            if answer == "Not checked yet":
                st.warning("Perform the check, or choose Unable to determine.")
            else:
                outcome = {"Yes": True, "No": False, "Unable to determine": None}[answer]
                st.session_state.evidence = update_evidence(st.session_state.evidence, check_id, outcome)
                st.session_state.history.append({"check_id": check_id, "outcome": outcome, "cost": agent["cost"], "source": "user-reported manual check"})
                st.session_state.assessment = assess_offer(st.session_state.offer, st.session_state.evidence)
                st.rerun()
    st.subheader("Suggested next steps")
    st.write("Verify the offer through independently found company contact details. "
             "Do not pay an upfront fee or share passwords/OTPs. Treat unresolved evidence cautiously.")
    with st.expander("Inspect the complete reasoning trace"):
        st.json(result)
        st.json(st.session_state.history)
    st.download_button("Download assessment (.txt)", make_report(result, st.session_state.history),
                       file_name="internshield_assessment.txt", mime="text/plain")
