from dataclasses import dataclass, field
from typing import Any, Optional

from .optiondescription import OptionDescription, SETTING_DESCRIPTION_KEY


@dataclass(frozen=True)
class TextOption(OptionDescription):
    """A free-text setting, rendered as a single-line edit box.

    Added for the LLM commander, which needs a provider base URL and a model
    identifier -- both are arbitrary strings that cannot be offered as a fixed
    choice list because the user may point the application at a local Ollama
    server or at a model that did not exist when this build was made.

    Secrets must **not** use this option type: ``Settings`` is pickled into the
    campaign save and dumped to JSON by the settings window, so anything stored
    here is readable by anyone the save is shared with. See
    :mod:`game.ai_commander.secretstore` for the key handling.
    """

    #: Shown greyed out when the value is empty.
    placeholder: Optional[str] = None
    #: Values longer than this are rejected by the UI.
    max_length: int = 512


def text_option(
    text: str,
    page: str,
    section: str,
    default: str,
    placeholder: Optional[str] = None,
    max_length: int = 512,
    detail: Optional[str] = None,
    tooltip: Optional[str] = None,
    causes_expensive_game_update: bool = False,
    **kwargs: Any,
) -> str:
    return field(
        metadata={
            SETTING_DESCRIPTION_KEY: TextOption(
                page,
                section,
                text,
                detail,
                tooltip,
                causes_expensive_game_update,
                placeholder=placeholder,
                max_length=max_length,
            )
        },
        default=default,
        **kwargs,
    )
