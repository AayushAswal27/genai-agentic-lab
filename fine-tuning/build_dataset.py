import json
import random

random.seed(42)

positive_texts = [
    ("The new onboarding flow shipped last Tuesday and signups are already up 30%. "
     "The team pulled it off ahead of schedule and the design reviews were smooth.",
     "Onboarding flow shipped early and signups rose 30%.",
     ["onboarding", "growth", "product"],
     ["Monitor signup numbers over the next month", "Share design review notes with the team"]),
    ("Really happy with how the Q3 offsite went. Everyone was engaged, the workshops "
     "landed well, and we left with a clear roadmap for next quarter.",
     "Q3 offsite was engaging and produced a clear roadmap.",
     ["offsite", "planning", "team"],
     ["Circulate the finalized roadmap", "Book the Q4 offsite venue"]),
    ("Customer support resolved the billing outage in under two hours and followed up "
     "with every affected account. Feedback has been overwhelmingly positive.",
     "Billing outage resolved quickly with positive customer feedback.",
     ["support", "billing", "reliability"],
     ["Write a postmortem for the outage", "Thank the support team publicly"]),
    ("The mentorship program hit its first cohort milestone. Mentees reported real "
     "progress and several mentors want to sign up again next round.",
     "Mentorship program's first cohort was a success.",
     ["mentorship", "learning", "community"],
     ["Open signups for the next cohort", "Collect testimonials from mentees"]),
    ("Our open-source library crossed 5k stars this week. Contributions are coming in "
     "steadily and the docs overhaul made a visible difference.",
     "Open-source library passed 5k stars with steady contributions.",
     ["open-source", "community", "documentation"],
     ["Highlight top contributors in the changelog", "Plan the next docs sprint"]),
]

negative_texts = [
    ("The deployment failed twice overnight and we still don't know the root cause. "
     "On-call was paged three times and the dashboard is showing elevated error rates.",
     "Repeated overnight deployment failures with unknown root cause.",
     ["deployment", "incident", "reliability"],
     ["Investigate the root cause", "Review on-call escalation load"]),
    ("Sales missed target for the second straight month. The pipeline looks thin and "
     "a couple of key deals slipped to next quarter.",
     "Sales missed target again with a thin pipeline.",
     ["sales", "revenue", "pipeline"],
     ["Audit the current pipeline", "Follow up on slipped deals"]),
    ("The migration corrupted a chunk of user records and the rollback took longer than "
     "expected. Several customers noticed missing data.",
     "Migration corrupted records and rollback was slow.",
     ["migration", "data", "incident"],
     ["Restore affected user records", "Add validation before the next migration"]),
    ("Morale on the platform team is low after the last reorg. People feel unclear on "
     "ownership and a few strong engineers have started interviewing elsewhere.",
     "Low morale on the platform team after the reorg.",
     ["morale", "reorg", "retention"],
     ["Clarify team ownership", "Schedule 1:1s with at-risk engineers"]),
    ("The vendor raised prices 40% with two weeks notice and their support has gotten "
     "noticeably worse. We're locked in until the contract renews.",
     "Vendor hiked prices sharply while support declined.",
     ["vendor", "cost", "contract"],
     ["Evaluate alternative vendors", "Flag the renewal date to finance"]),
]

neutral_texts = [
    ("The weekly metrics report is attached. Traffic held roughly flat, latency is "
     "within normal range, and no incidents were logged this week.",
     "Weekly metrics were stable with no incidents.",
     ["metrics", "monitoring"],
     ["File the report in the shared drive"]),
    ("We're switching the standup from 10am to 9:30am starting Monday to accommodate "
     "the new timezone spread on the team.",
     "Standup time moves to 9:30am on Monday.",
     ["scheduling", "team"],
     ["Update the calendar invite", "Notify the wider team"]),
    ("The API now supports pagination on the search endpoint. Existing clients are "
     "unaffected; the new parameters are optional.",
     "Search endpoint added optional pagination.",
     ["api", "search"],
     ["Update the API docs", "Add pagination examples"]),
    ("Inventory counts for the warehouse are complete. Numbers match the system of "
     "record with a small variance on two SKUs under review.",
     "Warehouse inventory counted with minor variance.",
     ["inventory", "operations"],
     ["Review the two flagged SKUs"]),
    ("The design system got a minor version bump. Spacing tokens were renamed; a "
     "codemod is available for teams that need to migrate.",
     "Design system minor bump renamed spacing tokens.",
     ["design-system", "tooling"],
     ["Run the codemod where needed", "Skim the migration notes"]),
]

instruction_templates = [
    "Read the following text and return the structured summary as JSON.\n\n{text}",
    "Extract the key details from this into the JSON schema.\n\n{text}",
    "Analyze the text below and give me the structured output.\n\n{text}",
    "Turn this into the standard JSON format.\n\n{text}",
    "Here's a note. Summarize it into the schema.\n\n{text}",
    "Process this text and respond with the JSON.\n\n{text}",
]

def make_examples():
    rows = []
    buckets = [
        ("positive", positive_texts),
        ("negative", negative_texts),
        ("neutral", neutral_texts),
    ]
    for sentiment, texts in buckets:
        for text, summary, topics, actions in texts:
            for tmpl in random.sample(instruction_templates, 4):
                instruction = tmpl.format(text=text)
                response = json.dumps({
                    "summary": summary,
                    "sentiment": sentiment,
                    "topics": topics,
                    "action_items": actions,
                }, ensure_ascii=False)
                rows.append({"instruction": instruction, "response": response})
    random.shuffle(rows)
    return rows

rows = make_examples()
with open("data/schema_sft.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(rows)} examples")
