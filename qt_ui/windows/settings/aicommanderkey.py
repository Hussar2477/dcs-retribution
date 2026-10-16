"""API key entry for the LLM commander.

The key is deliberately *not* a :class:`~game.settings.Settings` field. Settings
are pickled into every campaign save and written out as JSON by the settings
window's export, so a key stored there would end up in files players routinely
share when reporting bugs. It lives in a separate, permission-restricted file in
the Retribution user data directory instead, managed by
:class:`~game.ai_commander.secretstore.SecretStore`.

The widget never shows the key back to the user: once saved it reports only a
mask, and the entry field is cleared.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from game.ai_commander.secretstore import SecretStore


class AiCommanderKeyBox(QGroupBox):
    """Save/clear controls for the provider API key."""

    def __init__(self, store: SecretStore | None = None) -> None:
        super().__init__("Provider API Key")
        self.store = store if store is not None else SecretStore()

        layout = QVBoxLayout()
        self.setLayout(layout)

        explanation = QLabel(
            "<strong>OpenRouter API key</strong><br />"
            "Stored outside the campaign save, in a file only your user account "
            "can read. It is never written to the save, the settings export or "
            "the AI decision log. Leave empty when using a local provider such "
            "as Ollama."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.entry = QLineEdit()
        self.entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.entry.setPlaceholderText("sk-or-v1-...")
        self.entry.setClearButtonEnabled(True)
        self.entry.returnPressed.connect(self.save_key)
        layout.addWidget(self.entry)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save key")
        self.save_button.clicked.connect(self.save_key)
        buttons.addWidget(self.save_button)

        self.clear_button = QPushButton("Remove stored key")
        self.clear_button.clicked.connect(self.clear_key)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        self.status.setText(self.store.describe())
        self.clear_button.setEnabled(self.store.load_stored() is not None)

    def save_key(self) -> None:
        secret = self.entry.text().strip()
        if not secret:
            self.status.setText("Nothing entered, so the stored key is unchanged.")
            return
        if self.store.save(secret):
            # Clear immediately: the plaintext must not linger in a widget that
            # can be revealed by toggling echo mode from a debugger or a plugin.
            self.entry.clear()
            self.refresh()
        else:
            self.status.setText(
                "Could not write the key file. Check that the Retribution user "
                "data directory is writable."
            )

    def clear_key(self) -> None:
        if self.store.clear():
            self.entry.clear()
            self.refresh()
        else:
            self.status.setText("Could not remove the stored key file.")
