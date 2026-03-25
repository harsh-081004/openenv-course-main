"""Task and email definitions for the Email Triage environment."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EmailItem:
    email_id: str
    sender: str
    subject: str
    body: str
    urgency: str
    department: str
    sender_tier: str
    business_impact: int
    minutes_to_breach: int
    reference_summary: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    difficulty: str
    objective: str
    description: str
    required_fields: List[str]


EMAILS: List[EmailItem] = [
    EmailItem(
        email_id="email-001",
        sender="maya.chen@acme.com",
        subject="Payment failed twice for invoice 4921",
        body=(
            "Hi team, our card was charged but the invoice still shows unpaid. "
            "We attempted payment twice today and both failed with error code 54. "
            "Please help resolve this so our account stays in good standing."
        ),
        urgency="urgent",
        department="billing",
        sender_tier="standard",
        business_impact=78,
        minutes_to_breach=25,
        reference_summary="Investigate payment failures for invoice 4921 and confirm account status.",
    ),
    EmailItem(
        email_id="email-002",
        sender="leo@northwind.io",
        subject="API returning 500 on /v2/orders",
        body=(
            "Hello, our integration started failing this morning with 500 errors on /v2/orders. "
            "We did not change our client. Please advise on any outages or fixes."
        ),
        urgency="urgent",
        department="technical",
        sender_tier="enterprise",
        business_impact=92,
        minutes_to_breach=12,
        reference_summary="Report ongoing 500 errors on /v2/orders and request outage guidance.",
    ),
    EmailItem(
        email_id="email-003",
        sender="hr@contoso.com",
        subject="Request for updated W-9",
        body=(
            "We need your updated W-9 for our vendor records. "
            "Please send the form by the end of next week."
        ),
        urgency="normal",
        department="hr",
        sender_tier="standard",
        business_impact=40,
        minutes_to_breach=2880,
        reference_summary="Request updated W-9 form for vendor records by next week.",
    ),
    EmailItem(
        email_id="email-004",
        sender="priya@lumen.ai",
        subject="Interested in enterprise pricing",
        body=(
            "Hi, we are evaluating enterprise plans and need pricing for 500 seats. "
            "Can you share a quote and any onboarding timelines?"
        ),
        urgency="normal",
        department="sales",
        sender_tier="strategic",
        business_impact=86,
        minutes_to_breach=180,
        reference_summary="Ask for enterprise pricing quote for 500 seats and onboarding timeline.",
    ),
    EmailItem(
        email_id="email-005",
        sender="support@bluebird.dev",
        subject="Can you enable dark mode?",
        body=(
            "Hello, is dark mode available? If not, do you have an ETA? "
            "Our team uses the app late at night."
        ),
        urgency="low",
        department="general",
        sender_tier="standard",
        business_impact=25,
        minutes_to_breach=10080,
        reference_summary="Ask about dark mode availability and ETA for the product.",
    ),
    EmailItem(
        email_id="email-006",
        sender="legal@futura.io",
        subject="SSN exposure in onboarding form",
        body=(
            "We discovered that a new hire uploaded a form containing SSNs. "
            "Please advise on secure handling and confirm compliance steps."
        ),
        urgency="urgent",
        department="hr",
        sender_tier="enterprise",
        business_impact=95,
        minutes_to_breach=10,
        reference_summary="Report SSN exposure and request compliance handling guidance.",
    ),
    EmailItem(
        email_id="email-007",
        sender="ops@asterix-logistics.com",
        subject="Daily shipment label generation blocked",
        body=(
            "Our warehouse cannot print shipping labels since 6 AM and orders are piling up. "
            "Please restore the label API before the noon dispatch cutoff."
        ),
        urgency="urgent",
        department="technical",
        sender_tier="enterprise",
        business_impact=90,
        minutes_to_breach=35,
        reference_summary="Report blocked label API and request urgent fix before dispatch cutoff.",
    ),
    EmailItem(
        email_id="email-008",
        sender="finance@orionfoods.com",
        subject="Quarter-end credit memo mismatch",
        body=(
            "The credit memo totals in your portal differ from our ledger by $42,000. "
            "We need this corrected before close of business for quarter reporting."
        ),
        urgency="urgent",
        department="billing",
        sender_tier="strategic",
        business_impact=88,
        minutes_to_breach=55,
        reference_summary="Flag quarter-end credit memo mismatch and request urgent billing correction.",
    ),
    EmailItem(
        email_id="email-009",
        sender="it-security@novalabs.io",
        subject="Potential PCI data retention issue",
        body=(
            "Our audit found card data fragments in exported logs. "
            "Please confirm immediate containment and required compliance actions."
        ),
        urgency="urgent",
        department="hr",
        sender_tier="enterprise",
        business_impact=97,
        minutes_to_breach=20,
        reference_summary="Report PCI data retention concern and request immediate compliance containment.",
    ),
]

TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="task-urgency",
        name="Urgency Classification",
        difficulty="easy",
        objective="Classify the email urgency as urgent, normal, or low.",
        description="Pick the correct urgency label based on impact and time sensitivity.",
        required_fields=["urgency"],
    ),
    TaskSpec(
        task_id="task-routing",
        name="Department Routing",
        difficulty="medium",
        objective="Route the email to the correct department.",
        description=(
            "Choose the department that should handle the request: billing, "
            "technical, sales, hr, or general."
        ),
        required_fields=["department"],
    ),
    TaskSpec(
        task_id="task-full-triage",
        name="Full Triage",
        difficulty="hard",
        objective="Classify urgency, route department, and draft a summary response.",
        description=(
            "Provide urgency, department, and a one-sentence response summary "
            "that captures the user request."
        ),
        required_fields=["urgency", "department", "summary"],
    ),
]
