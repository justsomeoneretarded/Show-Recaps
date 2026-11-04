"""Evidence-backed television recaps, with separate season and series packages."""
from .models import RecapSpec, recap_seasons, validate_script_scope

__all__ = ["RecapSpec", "recap_seasons", "validate_script_scope"]
