"""The brief must show a few recent turns so the model can stop repeating a
losing approach.

A live campaign showed the model running the same strategy and priorities turn
after turn while bleeding forces, never reacting. Folding a compact history of
its own recent decisions into the brief gives it the signal to adapt. These
tests pin the rendering down and confirm it leaks no BLUE information (the
history is entirely RED's own past decisions).
"""

from __future__ import annotations

from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import IntelProjector, PriorTurnSummary
from tests.ai_commander import fakes


def _recent() -> tuple[PriorTurnSummary, ...]:
    return (
        PriorTurnSummary(
            turn=14,
            strategy="offensive",
            reserve_policy="aggressive",
            target_set_order=("enemy_aircraft", "enemy_battle_positions"),
            rejected_element_count=9,
        ),
        PriorTurnSummary(
            turn=13,
            strategy="offensive",
            reserve_policy="aggressive",
            target_set_order=("enemy_aircraft", "enemy_battle_positions"),
            rejected_element_count=7,
        ),
    )


class TestRecentTurnsRendering:
    def test_block_renders_when_several_turns_are_known(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project(
            recent_decisions=_recent()
        )
        rendered = brief.render_compact()
        assert "[RECENT TURNS]" in rendered
        assert "turn=14 strategy=offensive" in rendered
        assert "turn=13 strategy=offensive" in rendered
        # The reminder that repeating a losing approach is the problem.
        assert "change" in rendered.lower()

    def test_block_absent_with_only_one_recent_turn(self) -> None:
        # A single prior turn is already covered by [LAST TURN]; no duplication.
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project(
            recent_decisions=_recent()[:1]
        )
        assert "[RECENT TURNS]" not in brief.render_compact()

    def test_block_absent_when_no_history(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert "[RECENT TURNS]" not in brief.render_compact()

    def test_recent_turns_leak_no_blue_information(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project(
            recent_decisions=_recent()
        )
        blob = fakes.serialise_everything(brief.to_dict(), brief.render_compact())
        assert fakes.blue_leaks_in(blob) == []
