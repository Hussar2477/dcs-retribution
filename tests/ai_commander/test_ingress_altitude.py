"""The ingress altitude band's effect on the flight plan.

Item 1 gives the RED AI commander a single altitude lever: an optional per-flight
ingress band ("low"/"medium"/"high"). :meth:`WaypointBuilder.get_altitude` turns
that band into a real bias -- "low" hugs the floor of the doctrine envelope,
"high" the ceiling -- while never escaping the doctrine clamp and never touching
helicopters, which keep their fixed combat AGL.

These tests exercise the production ``get_altitude`` directly against a light
stub, so they pin the behaviour without standing up a whole coalition.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, cast

from game.ato.flightplans.waypointbuilder import WaypointBuilder
from game.utils import Distance, feet

_MIN = feet(15_000)
_MAX = feet(35_000)


def _altitude(*, is_helo: bool, band: Optional[str]) -> Distance:
    """Call the real ``get_altitude`` with a minimal stubbed ``self``."""

    flight = SimpleNamespace(
        is_helo=is_helo,
        plane_altitude_offset=0,
        ingress_band=band,
    )
    doctrine = SimpleNamespace(min_combat_altitude=_MIN, max_combat_altitude=_MAX)
    settings = SimpleNamespace(heli_combat_alt_agl=100)
    stub = SimpleNamespace(flight=flight, doctrine=doctrine, settings=settings)
    return WaypointBuilder.get_altitude(cast(WaypointBuilder, stub), feet(25_000))


class TestIngressBandBiasesAltitude:
    def test_low_band_flies_at_the_doctrine_floor(self) -> None:
        assert _altitude(is_helo=False, band="low") == _MIN

    def test_high_band_flies_at_the_doctrine_ceiling(self) -> None:
        assert _altitude(is_helo=False, band="high") == _MAX

    def test_the_band_orders_low_below_medium_below_high(self) -> None:
        low = _altitude(is_helo=False, band="low")
        medium = _altitude(is_helo=False, band="medium")
        high = _altitude(is_helo=False, band="high")
        assert low <= medium <= high

    def test_no_band_matches_the_medium_default(self) -> None:
        assert _altitude(is_helo=False, band=None) == _altitude(
            is_helo=False, band="medium"
        )

    def test_the_bias_never_escapes_the_doctrine_clamp(self) -> None:
        for band in ("low", "medium", "high", None):
            altitude = _altitude(is_helo=False, band=band)
            assert _MIN <= altitude <= _MAX

    def test_a_helicopter_ignores_the_band(self) -> None:
        # Helos always fly their fixed combat AGL, whatever the commander asked.
        expected = feet(100)
        for band in ("low", "medium", "high", None):
            assert _altitude(is_helo=True, band=band) == expected
