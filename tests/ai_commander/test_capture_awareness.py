"""Front-line and base-capture awareness in the brief the RED model reads.

The commander could always advance the front and take bases through aggressive
postures, but the brief used to *say the opposite* ("you cannot ... move the
front line, capture bases"), and never explained what each posture did or how a
capture is actually achieved. These tests pin down the corrected wording, the
posture legend, the capture chain, and the per-front ``capture=`` status --
which must expose only RED-observable facts (a blocking-position count and a
capturable flag), never anything BLUE-private.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from game.ai_commander.enums import FrontPosture, IntelPolicy
from game.ai_commander.intel import IntelProjector
from game.ai_commander.postures import capture_status_for
from game.theater import FrontLine
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


def _rendered(**kwargs: Any) -> str:
    _, game = fakes.synthetic_game(**kwargs)
    return IntelProjector(game, IntelPolicy.REALISTIC).project().render_compact()


class TestConstraintWording:
    def test_the_misleading_capture_prohibition_is_gone(self) -> None:
        text = _rendered()
        # The old text falsely told the model it could not do these at all.
        assert "capture bases" not in text
        assert "move the front line, capture bases" not in text

    def test_it_explains_capture_is_indirect_via_aggressive_postures(self) -> None:
        text = _rendered()
        assert "no direct 'capture' order" in text
        assert "breakthrough" in text


class TestPostureLegend:
    def test_every_posture_is_explained(self) -> None:
        text = _rendered()
        assert "[POSTURE LEGEND]" in text
        for posture in FrontPosture:
            assert f"{posture.value}:" in text

    def test_breakthrough_is_described_as_advance_and_capture(self) -> None:
        text = _rendered()
        assert "advance AND capture" in text


class TestCaptureChain:
    def test_the_capture_chain_names_the_target_set_and_posture(self) -> None:
        text = _rendered()
        assert "[CAPTURE CHAIN]" in text
        assert "enemy_battle_positions" in text
        assert "breakthrough" in text


class TestPerFrontCaptureStatus:
    def test_a_capturable_front_reports_it(self) -> None:
        # The synthetic campaign has a 2.13:1 force ratio and no blocking
        # positions, so a breakthrough could take the base now.
        text = _rendered()
        assert "capture=available" in text

    def test_a_thin_force_reports_needing_an_advantage(self) -> None:
        # Too few deployable units for the breakthrough precondition, but still
        # no blocking positions.
        text = _rendered(red_deployable=1)
        assert "capture=needs force advantage" in text

    def test_a_blocked_front_reports_the_blocking_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import game.commander.battlepositions as bp

        class _Blocked:
            blocking_capture = [object(), object(), object()]

        monkeypatch.setattr(
            bp.BattlePositions,
            "for_control_point",
            classmethod(lambda cls, cp: _Blocked()),
        )
        text = _rendered()
        assert "capture=blocked (3 enemy battle position(s)" in text
        assert "enemy_battle_positions" in text


class TestCaptureStatusLeaksNothing:
    def test_capture_status_contains_no_blue_private_values(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        for front in brief.fronts:
            blob = fakes.serialise_everything(front.capture_status)
            assert fakes.blue_leaks_in(blob) == []

    def test_capture_status_for_returns_a_plain_string(self) -> None:
        campaign, game = fakes.synthetic_game()
        status = capture_status_for(cast(FrontLine, campaign.front), fakes.Player.RED)
        assert isinstance(status, str)
        assert status.startswith("capture=")
