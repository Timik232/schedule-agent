from __future__ import annotations

import pytest

from src.utils.validators import ValidationError, ensure_whitelisted_identifier


def test_ensure_whitelisted_identifier_normalizes_input() -> None:
    value = ensure_whitelisted_identifier(" ИКМО-05-21 ", ["икмо-05-21", "пи-01"])
    assert value == "икмо-05-21"


def test_ensure_whitelisted_identifier_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        ensure_whitelisted_identifier("ikmo-99", ["икмо-05-21"])
