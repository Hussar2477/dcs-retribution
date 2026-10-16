"""Cumulative campaign memory for the RED commander.

Beyond the last-mission after-action and the most recent observed types, RED is
given a memory aggregated across EVERY recorded turn of the campaign: the union
of the distinct enemy TYPES it has seen and the threat causes that keep recurring.
It is built solely from RED's own stored decision-log records -- the same
information RED already had -- so it grants no new access to hidden BLUE state.
These tests pin the aggregation, the widened recent-turns window, the rendering
and the fairness boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from game.ai_commander.audit import AiDecisionRecord, AuditLog, RECENT_TURNS_WINDOW
from game.ai_commander.enums import IntelPolicy
from game.ai_commander.intel import (
    CampaignMemory,
    IntelProjector,
    ObservedEnemyForces,
    build_campaign_memory,
)
from tests.ai_commander import fakes


def _brief_dict(
    *,
    aircraft: tuple[str, ...] = (),
    air_defence: tuple[str, ...] = (),
    ground: tuple[str, ...] = (),
    naval: tuple[str, ...] = (),
    seen: tuple[str, ...] = (),
    air_causes: Optional[Mapping[str, int]] = None,
    ground_causes: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """A stored ``intel_brief`` payload of the shape RedCommanderBrief writes."""

    return {
        "observed_enemy_forces": {
            "aircraft_types": list(aircraft),
            "air_defence_types": list(air_defence),
            "ground_types": list(ground),
            "naval_types": list(naval),
        },
        "after_action": {
            "enemy_aircraft_types_seen": list(seen),
            "red_aircraft_lost_by_cause": dict(air_causes or {}),
            "red_ground_units_lost_by_cause": dict(ground_causes or {}),
        },
    }


def _record(turn: int, **kwargs: Any) -> dict[str, Any]:
    return {"turn_id": turn, "intel_brief": _brief_dict(**kwargs)}


class TestBuildCampaignMemory:
    def test_unions_distinct_types_across_turns(self) -> None:
        records = [
            _record(1, air_defence=("ZONE-SAM-A",), naval=("SHIP-X",)),
            _record(2, air_defence=("ZONE-SAM-B",), naval=("SHIP-X",)),
        ]
        memory = build_campaign_memory(records)
        assert memory.turns_recorded == 2
        # Union across turns, de-duplicated (SHIP-X seen on both turns once).
        assert memory.observed.air_defence_types == ("ZONE-SAM-A", "ZONE-SAM-B")
        assert memory.observed.naval_types == ("SHIP-X",)

    def test_aircraft_merge_from_both_channels_and_dedupe(self) -> None:
        # The same real aircraft seen once as a raw id (observed) and once as a
        # display name (after-action) is a single distinct type.
        records = [
            _record(1, aircraft=("FA-18C_hornet",), seen=("F/A-18C Hornet (Lot 20)",))
        ]
        memory = build_campaign_memory(records)
        assert memory.observed.aircraft_types == ("F/A-18C Hornet",)

    def test_recurring_threat_flagged_with_red_turn_count(self) -> None:
        records = [
            _record(1, air_causes={"enemy_aircraft": 2}),
            _record(2, air_causes={"enemy_aircraft": 1, "ground_sam": 1}),
            _record(3, air_causes={"enemy_aircraft": 3}),
        ]
        memory = build_campaign_memory(records)
        # enemy_aircraft recurred on 3 turns; ground_sam on only 1, so is omitted.
        assert memory.recurring_threats == (
            "aircraft losses to enemy aircraft (3 turns)",
        )

    def test_single_turn_cause_is_not_recurring(self) -> None:
        records = [_record(1, air_causes={"ground_sam": 5})]
        assert build_campaign_memory(records).recurring_threats == ()

    def test_unknown_cause_is_never_flagged(self) -> None:
        records = [
            _record(1, air_causes={"unknown": 1}),
            _record(2, air_causes={"unknown": 1}),
        ]
        assert build_campaign_memory(records).recurring_threats == ()

    def test_air_and_ground_causes_sorted_by_recurrence(self) -> None:
        records = [
            _record(
                1, air_causes={"enemy_aircraft": 1}, ground_causes={"ground_fire": 1}
            ),
            _record(
                2, air_causes={"enemy_aircraft": 1}, ground_causes={"ground_fire": 1}
            ),
            _record(3, ground_causes={"ground_fire": 1}),
        ]
        memory = build_campaign_memory(records)
        # ground_fire recurred on 3 turns, enemy_aircraft on 2 -> ground first.
        assert memory.recurring_threats == (
            "ground losses to ground fire (3 turns)",
            "aircraft losses to enemy aircraft (2 turns)",
        )

    def test_count_is_red_turns_not_enemy_numbers(self) -> None:
        # A single turn with a huge loss count is still only one turn: recurrence
        # is measured in RED's own turns, never in enemy strength.
        records = [_record(1, air_causes={"enemy_aircraft": 99})]
        assert build_campaign_memory(records).recurring_threats == ()

    def test_empty_and_malformed_records_are_safe(self) -> None:
        assert build_campaign_memory([]).is_empty
        assert build_campaign_memory(
            [{}, {"turn_id": 1}, "junk", {"intel_brief": "not-a-mapping"}]
        ).is_empty


class TestCampaignMemoryModel:
    def test_default_is_empty(self) -> None:
        assert CampaignMemory().is_empty

    def test_empty_when_no_types_or_threats_even_with_turns(self) -> None:
        assert CampaignMemory(turns_recorded=3).is_empty

    def test_to_dict_is_types_only(self) -> None:
        memory = CampaignMemory(
            turns_recorded=2,
            observed=ObservedEnemyForces(aircraft_types=("F-15C",)),
            recurring_threats=("aircraft losses to enemy aircraft (2 turns)",),
        )
        data = memory.to_dict()
        assert data["turns_recorded"] == 2
        assert data["observed"]["aircraft_types"] == ["F-15C"]
        assert data["recurring_threats"] == [
            "aircraft losses to enemy aircraft (2 turns)"
        ]


class TestAuditLogCampaignMemory:
    def test_aggregates_written_records(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path)
        for turn, brief in (
            (
                1,
                _brief_dict(
                    air_defence=("ZONE-SAM-A",), air_causes={"enemy_aircraft": 1}
                ),
            ),
            (
                2,
                _brief_dict(
                    air_defence=("ZONE-SAM-B",), air_causes={"enemy_aircraft": 1}
                ),
            ),
        ):
            log.write(
                AiDecisionRecord(
                    campaign_id_hash="abc123", turn_id=turn, intel_brief=brief
                )
            )
        memory = log.campaign_memory("abc123")
        assert memory.turns_recorded == 2
        assert memory.observed.air_defence_types == ("ZONE-SAM-A", "ZONE-SAM-B")
        assert memory.recurring_threats == (
            "aircraft losses to enemy aircraft (2 turns)",
        )

    def test_before_turn_excludes_current_and_later(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path)
        for turn, tag in ((1, "ZONE-SAM-A"), (2, "ZONE-SAM-B"), (3, "ZONE-SAM-C")):
            log.write(
                AiDecisionRecord(
                    campaign_id_hash="abc123",
                    turn_id=turn,
                    intel_brief=_brief_dict(air_defence=(tag,)),
                )
            )
        memory = log.campaign_memory("abc123", before_turn=3)
        assert memory.turns_recorded == 2
        assert memory.observed.air_defence_types == ("ZONE-SAM-A", "ZONE-SAM-B")

    def test_missing_campaign_is_empty(self, tmp_path: Path) -> None:
        assert AuditLog(tmp_path).campaign_memory("nope").is_empty


class TestRecentTurnsWindow:
    def test_window_is_six(self) -> None:
        assert RECENT_TURNS_WINDOW == 6

    def test_recent_summaries_surfaces_up_to_six_turns(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path)
        for turn in range(1, 9):  # eight decided turns
            log.write(AiDecisionRecord(campaign_id_hash="abc123", turn_id=turn))
        recent = log.recent_summaries("abc123", before_turn=9)
        assert len(recent) == RECENT_TURNS_WINDOW
        assert [s.turn for s in recent] == [8, 7, 6, 5, 4, 3]


class TestCampaignMemoryInTheBrief:
    def test_absent_when_empty(self) -> None:
        _, game = fakes.synthetic_game()
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project()
        assert brief.campaign_memory.is_empty
        assert "[CAMPAIGN MEMORY]" not in brief.render_compact()

    def test_renders_types_and_recurring_threats(self) -> None:
        _, game = fakes.synthetic_game()
        memory = CampaignMemory(
            turns_recorded=4,
            observed=ObservedEnemyForces(
                aircraft_types=("F-15C", "F-16C"), naval_types=("Ticonderoga",)
            ),
            recurring_threats=("aircraft losses to enemy aircraft (3 turns)",),
        )
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project(
            campaign_memory=memory
        )
        rendered = brief.render_compact()
        assert "[CAMPAIGN MEMORY]" in rendered
        assert "aircraft seen: F-15C, F-16C" in rendered
        assert "naval seen: Ticonderoga" in rendered
        assert "recurring threats:" in rendered
        assert "- aircraft losses to enemy aircraft (3 turns)" in rendered
        # The header states plainly it is cumulative and types-only.
        assert "cumulative" in rendered.lower()
        assert "not a count" in rendered.lower()

    def test_leaks_no_blue_information(self) -> None:
        _, game = fakes.synthetic_game()
        memory = CampaignMemory(
            turns_recorded=3,
            observed=ObservedEnemyForces(
                aircraft_types=("F-15C",), ground_types=("M-1 Abrams",)
            ),
            recurring_threats=("aircraft losses to enemy aircraft (2 turns)",),
        )
        brief = IntelProjector(game, IntelPolicy.REALISTIC).project(
            campaign_memory=memory
        )
        blob = fakes.serialise_everything(brief.to_dict(), brief.render_compact())
        assert fakes.blue_leaks_in(blob) == []
