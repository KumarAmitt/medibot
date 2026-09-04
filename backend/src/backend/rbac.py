from __future__ import annotations

ROLES = ("doctor", "nurse", "billing_executive", "technician", "admin")

COLLECTION_ACCESS: dict[str, list[str]] = {
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

ROLE_COLLECTIONS: dict[str, list[str]] = {
    role: [collection for collection, roles in COLLECTION_ACCESS.items() if role in roles]
    for role in ROLES
}

SQL_RAG_ROLES = frozenset({"billing_executive", "admin"})

ROLE_LABELS = {
    "doctor": "Doctor",
    "nurse": "Nurse",
    "billing_executive": "Billing Executive",
    "technician": "Technician",
    "admin": "Administrator",
}

COLLECTION_LABELS = {
    "general": "General",
    "clinical": "Clinical",
    "nursing": "Nursing",
    "billing": "Billing & Insurance",
    "equipment": "Equipment",
}

# Keyword hints used only to write a clear refusal — retrieval itself is always
# filtered by access_roles in Qdrant, so this never grants extra documents.
COLLECTION_INTENT: dict[str, tuple[str, ...]] = {
    "billing": (
        "billing code",
        "billing codes",
        "insurance billing",
        "claim submission",
        "cashless",
        "pre-auth",
        "preauthorisation",
        "pre-authorisation",
        "icd-10",
        "icd 10",
        "package rate",
        "tpa",
        "reimbursement claim",
        "exclusion code",
    ),
    "clinical": (
        "treatment protocol",
        "drug formulary",
        "diagnostic guideline",
        "dosage",
        "clinical protocol",
        "standard drug",
    ),
    "nursing": (
        "nursing procedure",
        "icu nursing",
        "infection control",
        "iv cannula",
        "central line",
        "patient care guideline",
    ),
    "equipment": (
        "equipment manual",
        "calibration",
        "maintenance schedule",
        "equipment operation",
        "fault code",
        "ventilator calibration",
    ),
}


def collections_for_role(role: str) -> list[str]:
    return ROLE_COLLECTIONS.get(role, ["general"])


def access_roles_for_collection(collection: str) -> list[str]:
    return COLLECTION_ACCESS.get(collection, ["admin"])


def infer_restricted_collections(question: str, role: str) -> list[str]:
    allowed = set(collections_for_role(role))
    text = question.lower()
    blocked: list[str] = []
    for collection, keywords in COLLECTION_INTENT.items():
        if collection in allowed:
            continue
        if any(keyword in text for keyword in keywords):
            blocked.append(collection)
    return blocked


def rbac_refusal_message(role: str, blocked: list[str] | None = None) -> str:
    allowed = collections_for_role(role)
    labels = [COLLECTION_LABELS[c] for c in allowed]
    if len(labels) == 1:
        allowed_label = labels[0]
    else:
        allowed_label = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    role_label = ROLE_LABELS.get(role, role)
    if blocked:
        denied = ", ".join(COLLECTION_LABELS[c] for c in blocked)
        return (
            f"As a {role_label}, you don't have access to {denied} documents. "
            f"I can only answer questions from the {allowed_label} collections."
        )
    return (
        f"As a {role_label}, you don't have access to that information. "
        f"I can only answer questions from the {allowed_label} collections."
    )


def sql_refusal_message(role: str) -> str:
    role_label = ROLE_LABELS.get(role, role)
    return (
        f"As a {role_label}, you don't have access to operational analytics. "
        "SQL RAG over claims and maintenance tickets is limited to billing executives and administrators."
    )
