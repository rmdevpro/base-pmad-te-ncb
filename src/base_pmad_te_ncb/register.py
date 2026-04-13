"""
TE — Package registration entry point.

Called by the bootstrap kernel's package_registry when this package
is loaded via convention-based import.

Returns a TERegistration dict with the Imperator flow builder and
identity/purpose declarations.
"""


def register() -> dict:
    """Register the TE's cognitive StateGraphs.

    Returns a dict with:
    - identity: What the Imperator is
    - purpose: What the Imperator is for
    - imperator_builder: callable that builds the compiled Imperator StateGraph
    """
    from base_pmad_te_ncb.imperator_flow import build_imperator_flow

    return {
        "identity": "Imperator",
        "purpose": "pMAD management and conversational interface",
        "imperator_builder": build_imperator_flow,
        "tools_required": [],
        "flows": {},
    }
