"""The RED capability index must be accurate and RED-only.

The capability index is the list of aircraft, ground units and ships RED can
actually field or buy this turn. Two things have to hold at once:

* **Accuracy.** Every entry has to match the campaign's own unit data --
  otherwise the model is steered towards orders the legality checker will just
  reject, wasting the turn. The spot-checks here read the same synthetic
  faction the builder read and assert the rendered numbers agree.
* **Anti-cheat.** The index is compiled from RED's coalition only. It must never
  carry a BLUE-private value, because the whole point of ACTIVE mode is that the
  opponent plans from the same information a human RED player would have. The
  sentinel scan is the same blunt instrument used in :mod:`test_intel_leak`.

The synthetic faction (see :mod:`tests.ai_commander.fakes`) fields two RED
squadrons (a fielded RED-JET multirole and a fielded RED-BOMBER) plus an
*unfielded* RED-INTERCEPTOR the faction owns on paper but has no airframes of,
and four ground unit types of which three are purchasable and one (a SAM) is
air-defence-only and must be excluded from the buy list.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from game.ai_commander.capabilities import (
    CAPABILITY_CACHE,
    CAPABILITY_SCHEMA_VERSION,
    CapabilityIndex,
    CapabilityIndexBuilder,
    capability_index_for,
    render_capability_index,
)
from game.ato.flighttype import FlightType
from tests.ai_commander import fakes


@pytest.fixture(autouse=True)
def _objective_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes.patch_objective_finder(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_capability_cache() -> Iterator[None]:
    """Synthetic factions reuse ids; a stale cache entry would poison a test."""

    CAPABILITY_CACHE.clear()
    yield
    CAPABILITY_CACHE.clear()


def _index() -> tuple[fakes.SyntheticCampaign, CapabilityIndex]:
    campaign, _ = fakes.synthetic_game()
    return campaign, capability_index_for(campaign.red)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestIndexIsAccurate:
    def test_schema_version_is_stamped(self) -> None:
        _, index = _index()
        assert index.schema_version == CAPABILITY_SCHEMA_VERSION

    def test_it_lists_exactly_the_factions_aircraft(self) -> None:
        _, index = _index()
        assert index.aircraft_ids == ("RED-JET", "RED-BOMBER", "RED-INTERCEPTOR")

    def test_fielded_flag_tracks_owned_airframes(self) -> None:
        _, index = _index()
        jet = index.aircraft_for("RED-JET")
        bomber = index.aircraft_for("RED-BOMBER")
        interceptor = index.aircraft_for("RED-INTERCEPTOR")
        assert jet is not None and jet.is_fielded
        assert bomber is not None and bomber.is_fielded
        # The faction operates the interceptor but owns no airframes of it, so
        # it is advertised as a type yet flagged unfielded.
        assert interceptor is not None and not interceptor.is_fielded
        assert interceptor.squadron_count == 0

    def test_squadron_count_and_owned_airframes_match_the_faction(self) -> None:
        _, index = _index()
        jet = index.aircraft_for("RED-JET")
        assert jet is not None
        assert jet.squadron_count == 1
        assert jet.owned_airframes == 8

    def test_prices_match_the_underlying_unit_data(self) -> None:
        _, index = _index()
        jet = index.aircraft_for("RED-JET")
        bomber = index.aircraft_for("RED-BOMBER")
        assert jet is not None and jet.price == 22
        assert bomber is not None and bomber.price == 34

    def test_roles_are_reported_per_airframe(self) -> None:
        _, index = _index()
        jet = index.aircraft_for("RED-JET")
        assert jet is not None
        assert set(jet.role_names()) == {"BARCAP", "TARCAP", "CAS", "SEAD", "Escort"}

    def test_max_flight_size_is_carried(self) -> None:
        _, index = _index()
        jet = index.aircraft_for("RED-JET")
        bomber = index.aircraft_for("RED-BOMBER")
        assert jet is not None and jet.max_flight_size == 4
        assert bomber is not None and bomber.max_flight_size == 2

    def test_aircraft_capable_of_a_task_are_resolved(self) -> None:
        _, index = _index()
        sead = {a.unit_id for a in index.aircraft_capable_of(FlightType.SEAD)}
        strike = {a.unit_id for a in index.aircraft_capable_of(FlightType.STRIKE)}
        assert "RED-JET" in sead
        assert "RED-BOMBER" in strike
        # The jet has no strike role, so it is not offered for strike.
        assert "RED-JET" not in strike


class TestGroundUnitsAndPurchasability:
    def test_it_lists_the_factions_ground_units(self) -> None:
        _, index = _index()
        assert set(index.ground_unit_ids) == {
            "RED-ARTY",
            "RED-TRUCK",
            "RED-TANK",
            "RED-SAM",
        }

    def test_only_procurable_ground_units_are_purchasable(self) -> None:
        _, index = _index()
        # The SAM is an air-defence type the faction fields but cannot buy from
        # a front-line base, so it must not appear in the buy list.
        assert set(index.purchasable_ground_unit_ids) == {
            "RED-ARTY",
            "RED-TRUCK",
            "RED-TANK",
        }
        assert "RED-SAM" not in index.purchasable_ground_unit_ids

    def test_ground_unit_prices_match(self) -> None:
        _, index = _index()
        tank = index.ground_unit_for("RED-TANK")
        arty = index.ground_unit_for("RED-ARTY")
        truck = index.ground_unit_for("RED-TRUCK")
        assert tank is not None and tank.price == 12
        assert arty is not None and arty.price == 18
        assert truck is not None and truck.price == 4


class TestNoShips:
    def test_a_land_faction_advertises_no_ships(self) -> None:
        _, index = _index()
        assert index.ship_ids == ()


# ---------------------------------------------------------------------------
# Anti-cheat: RED-only, no BLUE leaks
# ---------------------------------------------------------------------------


class TestIndexIsRedOnly:
    def _blob(self, index: CapabilityIndex) -> str:
        return fakes.serialise_everything(
            index.to_dict(),
            index.render_compact(),
            index.render_compact((FlightType.SEAD, FlightType.STRIKE)),
        )

    def test_no_blue_sentinel_appears_in_the_index(self) -> None:
        _, index = _index()
        assert fakes.blue_leaks_in(self._blob(index)) == []

    def test_the_faction_name_is_red(self) -> None:
        _, index = _index()
        # A land faction's own name is public; the guard is only that it is RED's.
        assert "BLUE" not in index.faction_name.upper()

    def test_render_names_the_real_red_aircraft(self) -> None:
        """Guards against a vacuous index that leaks nothing by saying nothing."""

        _, index = _index()
        rendered = index.render_compact()
        assert "RED-JET" in rendered
        assert "RED-BOMBER" in rendered

    def test_the_builder_never_reads_the_opponent(self) -> None:
        """The builder is constructed from RED's coalition and only that."""

        campaign, _ = fakes.synthetic_game()
        builder = CapabilityIndexBuilder(campaign.red)
        index = builder.build()
        assert fakes.blue_leaks_in(fakes.serialise_everything(index.to_dict())) == []


class TestContentHashAndCaching:
    def test_content_hash_is_stable_across_rebuilds(self) -> None:
        campaign_a, _ = fakes.synthetic_game()
        first = capability_index_for(campaign_a.red).content_hash()
        CAPABILITY_CACHE.clear()
        campaign_b, _ = fakes.synthetic_game()
        second = capability_index_for(campaign_b.red).content_hash()
        assert first == second

    def test_content_hash_is_a_short_hex_digest(self) -> None:
        _, index = _index()
        digest = index.content_hash()
        assert len(digest) == 16
        int(digest, 16)  # raises if it is not hex

    def test_render_capability_index_helper_matches_the_index(self) -> None:
        campaign, _ = fakes.synthetic_game()
        rendered = render_capability_index(campaign.red)
        assert "RED-JET" in rendered
