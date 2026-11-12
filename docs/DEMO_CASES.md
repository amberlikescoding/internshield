# Three fictional demonstrations

These are hand-written behaviour examples, not independent benchmark data. Model scores can differ between scenarios and should not be memorised as facts about real offers.

## A — upfront fee and secret request

Offer: "Congratulations! You have been selected for a remote data entry internship. No interview is needed. Pay a registration fee before joining and share your OTP to activate the employee account. Start immediately."

Company: Demo Hiring Services. Email: recruiting.demo@gmail.com. Fee: Yes. Interview: No.

Expected: R01, R02, R03 and R06 fire; rule severity caps at 100; High final risk; agent stops due to fee warning. No live checks claimed.

## B — ordinary offer, still needs independent evidence

Offer: "We invite you to a twelve week software engineering internship. You will work with a mentor on Python testing and internal tools. Following your technical interview, our team will discuss the role, stipend, working hours and written employment terms. No payment is required from applicants."

Company: Example Engineering. Email: careers@example.com. Fee: No. Interview: Yes.

Expected: configured rules initially do not fire. Agent plans company check (2) then recruiter confirmation (2), total 4. Record Yes to both only as a fictional demonstration; inspect completeness rising to 100%. A low result never proves safety.

## C — unresolved evidence

Use B's text, Fee: Unknown, Interview: Unknown. Record Unable to determine for the fee check. The check is marked attempted but completeness remains 0%. The agent moves on without repeatedly asking the same question. Record further answers and inspect replanning and the final stop condition. Unknowns must remain null in the trace.

Before the viva: one person operates the UI, a second explains the model/rules, and a third explains the agent/fuzzy trace. Swap roles so all three can explain the full project.
