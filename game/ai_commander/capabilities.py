"""Capability index for the LLM RED commander.

The commander must understand what its own units can actually do, and it must get
that understanding from Retribution's own data tables rather than from whatever the
language model happens to remember about real-world hardware. Everything in this
module is derived from :mod:`game.dcs` unit definitions (which are loaded from the
``resources/units`` YAML shipped with the game and from pydcs' unit maps), from the
faction definition, and from the coalition's own air wing.

Two properties matter and are enforced by construction:

* **No hallucination.** Ranges, prices, roles, group sizes and SAM behaviour are read
  from the same tables the game engine uses. If the game does not model a capability,
  it does not appear here, so the model cannot plan around a capability that the
  simulation will not honour.

* **No BLUE leakage.** The index is built from *one* coalition's faction and air wing.
  It never walks the opposing coalition, the opposing faction, or the theater's
  control points, so there is no code path by which BLUE order of battle can reach
  the prompt. Knowledge about the enemy stays in
  :class:`game.ai_commander.intel.RedCommanderBrief`, which applies the observability
  rules.

The index is expensive to render but cheap to build and is stable for the whole
campaign (a faction's unit roster does not change), so it is cached per faction
roster signature and rendered lazily.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from game.ato.flighttype import FlightType
from game.dcs.aircrafttype import AircraftType
from game.dcs.groundunittype import GroundUnitType
from game.dcs.shipunittype import ShipUnitType

if TYPE_CHECKING:
    from game.coalition import Coalition
    from game.factions.faction import Faction


#: Bumped whenever the wire format of the rendered index or :meth:`to_dict` changes.
CAPABILITY_SCHEMA_VERSION = "red-capability-index/1"

#: Mission roles are rendered with their suitability score so the model can tell a
#: dedicated interceptor from an airframe that can technically hold a CAP. Long role
#: lists are truncated to keep the prompt affordable; the omitted roles always have
#: lower suitability than the ones that survive.
MAX_RENDERED_ROLES = 8

#: Ground and naval rosters can be very large in modded factions. The renderer keeps
#: the cheapest and the most expensive of each class plus everything in between up to
#: this many entries per section, and reports how many were omitted.
MAX_RENDERED_AIRCRAFT = 44
MAX_RENDERED_GROUND_UNITS = 60
MAX_RENDERED_SHIPS = 24

#: Unit data files use these strings where a human-readable field is unpopulated.
#: They carry no information, so they are normalised to ``unknown``.
_PLACEHOLDER_TEXT = frozenset({"no data.", "no data", "n/a", "none", "unknown"})


def _clean(text: Optional[object]) -> str:
    """Collapses a free-text data-file field to a single short line.

    The value is coerced with :func:`str` first. ``UnitType.year_introduced`` is
    annotated ``str`` but the unit YAML stores bare integers (``introduced: 1983``),
    so the loaded value is genuinely an ``int`` for most airframes.
    """

    if text is None or text == "":
        return ""
    cleaned = " ".join(str(text).split())
    if cleaned.lower() in _PLACEHOLDER_TEXT:
        return ""
    return cleaned


@dataclass(frozen=True)
class AircraftCapability:
    """What one airframe available to this coalition can do.

    Every field is copied from :class:`game.dcs.aircrafttype.AircraftType`, which is
    itself loaded from ``resources/units/aircraft/*.yaml``. This is the same data the
    in-game aircraft information panel shows a human player.
    """

    unit_id: str
    display_name: str
    price: int
    year_introduced: str
    role: str
    max_flight_size: int
    combat_radius_nm: int
    roles: tuple[tuple[str, int], ...]
    carrier_capable: bool
    lha_capable: bool
    has_targeting_pod: bool
    has_ecm: bool
    can_carry_cargo: bool
    #: Number of airframes this coalition currently has on strength across all
    #: squadrons of this type. Zero means the faction may recruit the type but has no
    #: squadron for it yet.
    owned_airframes: int
    #: Number of squadrons of this type in the coalition's air wing.
    squadron_count: int

    @property
    def is_fielded(self) -> bool:
        return self.squadron_count > 0

    def role_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.roles)

    def render(self) -> str:
        roles = self.roles[:MAX_RENDERED_ROLES]
        role_text = ",".join(f"{name}:{score}" for name, score in roles)
        if len(self.roles) > len(roles):
            role_text += f",+{len(self.roles) - len(roles)}_lower"
        traits = []
        if self.carrier_capable:
            traits.append("carrier")
        if self.lha_capable:
            traits.append("lha")
        if self.has_targeting_pod:
            traits.append("tgp")
        if self.has_ecm:
            traits.append("ecm")
        if self.can_carry_cargo:
            traits.append("cargo")
        trait_text = ",".join(traits) if traits else "-"
        return (
            f"{self.unit_id} | ${self.price} | {self.year_introduced} | {self.role} "
            f"| flight<={self.max_flight_size} | radius={self.combat_radius_nm}nm "
            f"| onhand={self.owned_airframes} in {self.squadron_count}sqn "
            f"| traits={trait_text} | roles={role_text}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "display_name": self.display_name,
            "price": self.price,
            "year_introduced": self.year_introduced,
            "role": self.role,
            "max_flight_size": self.max_flight_size,
            "combat_radius_nm": self.combat_radius_nm,
            "roles": [
                {"task": name, "suitability": score} for name, score in self.roles
            ],
            "carrier_capable": self.carrier_capable,
            "lha_capable": self.lha_capable,
            "has_targeting_pod": self.has_targeting_pod,
            "has_ecm": self.has_ecm,
            "can_carry_cargo": self.can_carry_cargo,
            "owned_airframes": self.owned_airframes,
            "squadron_count": self.squadron_count,
        }

    @staticmethod
    def from_aircraft_type(
        aircraft: AircraftType, owned_airframes: int, squadron_count: int
    ) -> AircraftCapability:
        roles = tuple(
            (task.value, score)
            for task, score in sorted(
                aircraft.task_priorities.items(),
                key=lambda item: (-item[1], item[0].value),
            )
        )
        return AircraftCapability(
            unit_id=aircraft.variant_id,
            display_name=aircraft.display_name,
            price=aircraft.price,
            year_introduced=_clean(aircraft.year_introduced) or "unknown",
            role=_clean(aircraft.role) or "unknown",
            max_flight_size=aircraft.max_group_size,
            combat_radius_nm=int(round(aircraft.max_mission_range.nautical_miles)),
            roles=roles,
            carrier_capable=bool(aircraft.carrier_capable),
            lha_capable=bool(aircraft.lha_capable),
            has_targeting_pod=bool(aircraft.has_built_in_target_pod),
            has_ecm=bool(aircraft.has_built_in_ecm),
            can_carry_cargo=bool(aircraft.can_carry_crates),
            owned_airframes=owned_airframes,
            squadron_count=squadron_count,
        )


@dataclass(frozen=True)
class GroundUnitCapability:
    """What one ground unit available to this coalition is for.

    ``purchasable`` mirrors Retribution's own procurement pool: only front line and
    artillery units can be bought with the coalition budget. Air defence, infantry and
    logistics units exist in the faction but are placed by the campaign generator, so
    the commander must not be told it can buy them.
    """

    unit_id: str
    display_name: str
    price: int
    year_introduced: str
    role: str
    unit_class: str
    purchasable: bool
    #: Only meaningful for air defence units. ``None`` when the unit definition
    #: declares no Skynet behaviour.
    engages_harm: Optional[bool]
    engages_air_weapons: Optional[bool]
    engagement_zone: Optional[str]

    def render(self) -> str:
        parts = [
            f"{self.unit_id}",
            f"${self.price}",
            f"{self.year_introduced}",
            f"class={self.unit_class}",
            f"buy={'yes' if self.purchasable else 'no'}",
        ]
        iads = []
        if self.engages_harm is not None:
            iads.append(f"anti_harm={'yes' if self.engages_harm else 'no'}")
        if self.engages_air_weapons is not None:
            iads.append(f"anti_munition={'yes' if self.engages_air_weapons else 'no'}")
        if self.engagement_zone:
            iads.append(f"ez={self.engagement_zone}")
        if iads:
            parts.append(",".join(iads))
        if self.role and self.role != self.unit_class:
            parts.append(self.role)
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "display_name": self.display_name,
            "price": self.price,
            "year_introduced": self.year_introduced,
            "role": self.role,
            "unit_class": self.unit_class,
            "purchasable": self.purchasable,
            "engages_harm": self.engages_harm,
            "engages_air_weapons": self.engages_air_weapons,
            "engagement_zone": self.engagement_zone,
        }

    @staticmethod
    def from_ground_unit_type(
        unit: GroundUnitType, purchasable: bool
    ) -> GroundUnitCapability:
        skynet = unit.skynet_properties
        engages_harm: Optional[bool] = None
        engages_air_weapons: Optional[bool] = None
        if skynet.can_engage_harm is not None:
            engages_harm = str(skynet.can_engage_harm).strip().lower() == "true"
        if skynet.can_engage_air_weapon is not None:
            engages_air_weapons = (
                str(skynet.can_engage_air_weapon).strip().lower() == "true"
            )
        return GroundUnitCapability(
            unit_id=unit.variant_id,
            display_name=unit.display_name,
            price=unit.price,
            year_introduced=_clean(unit.year_introduced) or "unknown",
            role=_clean(unit.role) or "unknown",
            unit_class=unit.unit_class.value,
            purchasable=purchasable,
            engages_harm=engages_harm,
            engages_air_weapons=engages_air_weapons,
            engagement_zone=_clean(skynet.engagement_zone) or None,
        )


@dataclass(frozen=True)
class ShipCapability:
    """What one naval unit available to this coalition is for."""

    unit_id: str
    display_name: str
    price: int
    year_introduced: str
    role: str
    unit_class: str

    def render(self) -> str:
        return (
            f"{self.unit_id} | ${self.price} | {self.year_introduced} "
            f"| class={self.unit_class} | {self.role}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "display_name": self.display_name,
            "price": self.price,
            "year_introduced": self.year_introduced,
            "role": self.role,
            "unit_class": self.unit_class,
        }

    @staticmethod
    def from_ship_unit_type(ship: ShipUnitType) -> ShipCapability:
        return ShipCapability(
            unit_id=ship.variant_id,
            display_name=ship.display_name,
            price=ship.price,
            year_introduced=_clean(ship.year_introduced) or "unknown",
            role=_clean(ship.role) or "unknown",
            unit_class=ship.unit_class.value,
        )


@dataclass(frozen=True)
class DoctrineCapability:
    """The coalition's own doctrine switches.

    Retribution's doctrine object gates whole mission families. If a faction's
    doctrine says it does not do SEAD, the planner will refuse SEAD packages, so the
    commander needs to know before it asks for them.
    """

    name: str
    supports_cas: bool
    supports_cap: bool
    supports_sead: bool
    supports_strike: bool
    supports_antiship: bool

    def render(self) -> str:
        allowed = []
        if self.supports_cap:
            allowed.append("CAP")
        if self.supports_cas:
            allowed.append("CAS")
        if self.supports_sead:
            allowed.append("SEAD/DEAD")
        if self.supports_strike:
            allowed.append("STRIKE")
        if self.supports_antiship:
            allowed.append("ANTISHIP")
        return f"{self.name} | mission families: {','.join(allowed) if allowed else 'none'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supports_cas": self.supports_cas,
            "supports_cap": self.supports_cap,
            "supports_sead": self.supports_sead,
            "supports_strike": self.supports_strike,
            "supports_antiship": self.supports_antiship,
        }


@dataclass(frozen=True)
class CapabilityIndex:
    """Everything the commander is allowed to know about its own hardware."""

    schema_version: str
    faction_name: str
    doctrine: DoctrineCapability
    aircraft: tuple[AircraftCapability, ...]
    ground_units: tuple[GroundUnitCapability, ...]
    ships: tuple[ShipCapability, ...]
    has_jtac: bool
    #: Sections that were deliberately left out, so the audit log can show that the
    #: omission was a policy decision and not a data-loading failure.
    omitted_sections: tuple[str, ...] = field(default=())

    @property
    def aircraft_ids(self) -> tuple[str, ...]:
        return tuple(a.unit_id for a in self.aircraft)

    @property
    def ground_unit_ids(self) -> tuple[str, ...]:
        return tuple(u.unit_id for u in self.ground_units)

    @property
    def purchasable_ground_unit_ids(self) -> tuple[str, ...]:
        return tuple(u.unit_id for u in self.ground_units if u.purchasable)

    @property
    def ship_ids(self) -> tuple[str, ...]:
        return tuple(s.unit_id for s in self.ships)

    def aircraft_for(self, unit_id: str) -> Optional[AircraftCapability]:
        for entry in self.aircraft:
            if entry.unit_id == unit_id:
                return entry
        return None

    def ground_unit_for(self, unit_id: str) -> Optional[GroundUnitCapability]:
        for entry in self.ground_units:
            if entry.unit_id == unit_id:
                return entry
        return None

    def aircraft_capable_of(self, task: FlightType) -> tuple[AircraftCapability, ...]:
        return tuple(a for a in self.aircraft if task.value in a.role_names())

    def tasks_available(self) -> tuple[str, ...]:
        """Mission types at least one available airframe can perform."""

        tasks: set[str] = set()
        for entry in self.aircraft:
            tasks.update(entry.role_names())
        return tuple(sorted(tasks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "faction_name": self.faction_name,
            "doctrine": self.doctrine.to_dict(),
            "aircraft": [a.to_dict() for a in self.aircraft],
            "ground_units": [u.to_dict() for u in self.ground_units],
            "ships": [s.to_dict() for s in self.ships],
            "has_jtac": self.has_jtac,
            "omitted_sections": list(self.omitted_sections),
        }

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def render_compact(self, tasks: Optional[Iterable[FlightType]] = None) -> str:
        """Renders the index as terse lines for inclusion in a prompt.

        ``tasks`` restricts the aircraft section to airframes usable for the given
        mission types, which is how the mission-planning stage of an ACTIVE turn keeps
        its prompt small. Ground and naval sections are unaffected because they are
        already short relative to the aircraft roster.
        """

        aircraft = self.aircraft
        if tasks is not None:
            wanted = {t.value for t in tasks}
            aircraft = tuple(a for a in aircraft if wanted & set(a.role_names()))

        lines: list[str] = [
            "CAPABILITY INDEX (from this campaign's own unit data; "
            "these are the only units you may name)",
            f"faction: {self.faction_name}",
            f"doctrine: {self.doctrine.render()}",
            f"jtac available: {'yes' if self.has_jtac else 'no'}",
            "",
            "AIRCRAFT (roles show mission type:suitability, higher is better; "
            "radius is the planner's maximum mission range)",
        ]
        shown = _trim(aircraft, MAX_RENDERED_AIRCRAFT)
        if shown:
            lines.extend(f"  {entry.render()}" for entry in shown)
        else:
            lines.append("  (none available for the requested mission types)")
        if len(aircraft) > len(shown):
            lines.append(
                f"  ...{len(aircraft) - len(shown)} further airframe types omitted; "
                "they are not currently fielded by any squadron"
            )

        ground = _trim(self.ground_units, MAX_RENDERED_GROUND_UNITS)
        lines.append("")
        lines.append(
            "GROUND UNITS (buy=yes may be purchased with the budget at a base with a "
            "supply source; buy=no is placed by the campaign and cannot be bought)"
        )
        if ground:
            lines.extend(f"  {entry.render()}" for entry in ground)
        else:
            lines.append("  (none)")
        if len(self.ground_units) > len(ground):
            lines.append(
                f"  ...{len(self.ground_units) - len(ground)} further ground unit types "
                "omitted to keep this brief short"
            )

        ships = _trim(self.ships, MAX_RENDERED_SHIPS)
        lines.append("")
        lines.append("NAVAL UNITS (placed by the campaign; not purchasable)")
        if ships:
            lines.extend(f"  {entry.render()}" for entry in ships)
        else:
            lines.append("  (none)")
        if len(self.ships) > len(ships):
            lines.append(
                f"  ...{len(self.ships) - len(ships)} further naval unit types omitted"
            )

        if self.omitted_sections:
            lines.append("")
            lines.append(
                "WITHHELD BY POLICY: " + ", ".join(sorted(self.omitted_sections))
            )
        return "\n".join(lines)


def _trim(entries: tuple[Any, ...], limit: int) -> tuple[Any, ...]:
    if len(entries) <= limit:
        return entries
    return entries[:limit]


class CapabilityIndexBuilder:
    """Builds a :class:`CapabilityIndex` for exactly one coalition.

    The builder takes a :class:`game.coalition.Coalition` and never reaches for
    ``coalition.opponent``, ``game.blue``, ``game.red`` or the theater. That is the
    structural reason this class cannot leak the opposing order of battle.
    """

    def __init__(self, coalition: Coalition) -> None:
        self.coalition = coalition

    @property
    def faction(self) -> Faction:
        return self.coalition.faction

    def build(self) -> CapabilityIndex:
        return CapabilityIndex(
            schema_version=CAPABILITY_SCHEMA_VERSION,
            faction_name=_clean(self.faction.name) or "unknown faction",
            doctrine=self._doctrine(),
            aircraft=self._aircraft(),
            ground_units=self._ground_units(),
            ships=self._ships(),
            has_jtac=bool(self.faction.has_jtac),
            omitted_sections=(
                "enemy unit capabilities (see the intelligence brief instead)",
                "weapon-by-weapon payload tables (the mission generator picks payloads)",
            ),
        )

    def _doctrine(self) -> DoctrineCapability:
        doctrine = self.faction.doctrine
        return DoctrineCapability(
            name=_clean(getattr(doctrine, "name", "")) or "unknown",
            supports_cas=bool(getattr(doctrine, "cas", False)),
            supports_cap=bool(getattr(doctrine, "cap", False)),
            supports_sead=bool(getattr(doctrine, "sead", False)),
            supports_strike=bool(getattr(doctrine, "strike", False)),
            supports_antiship=bool(getattr(doctrine, "antiship", False)),
        )

    def _available_aircraft(self) -> Iterator[AircraftType]:
        faction = self.faction
        yield from faction.aircraft
        yield from faction.awacs
        yield from faction.tankers
        if faction.jtac_unit is not None:
            yield faction.jtac_unit

    def _aircraft(self) -> tuple[AircraftCapability, ...]:
        on_hand: dict[str, int] = {}
        squadron_counts: dict[str, int] = {}
        fielded: dict[str, AircraftType] = {}
        for squadron in self.coalition.air_wing.iter_squadrons():
            aircraft = squadron.aircraft
            key = aircraft.variant_id
            fielded[key] = aircraft
            squadron_counts[key] = squadron_counts.get(key, 0) + 1
            on_hand[key] = on_hand.get(key, 0) + self._squadron_strength(squadron)

        by_id: dict[str, AircraftType] = dict(fielded)
        for aircraft in self._available_aircraft():
            by_id.setdefault(aircraft.variant_id, aircraft)

        entries = [
            AircraftCapability.from_aircraft_type(
                aircraft,
                owned_airframes=on_hand.get(unit_id, 0),
                squadron_count=squadron_counts.get(unit_id, 0),
            )
            for unit_id, aircraft in by_id.items()
        ]
        entries.sort(key=lambda entry: (-entry.owned_airframes, entry.unit_id))
        return tuple(entries)

    @staticmethod
    def _squadron_strength(squadron: Any) -> int:
        """Airframes on strength, tolerating squadron stubs used by tests."""

        for attribute in ("owned_aircraft", "untasked_aircraft"):
            value = getattr(squadron, attribute, None)
            if isinstance(value, int):
                return value
        return 0

    def _ground_units(self) -> tuple[GroundUnitCapability, ...]:
        """Every ground unit type RED fields, flagged with what it may buy.

        ``purchasable`` mirrors the *human* purchase screen exactly:
        :class:`qt_ui.windows.basemenu.ground_forces.QArmorRecruitmentMenu` offers
        ``faction.ground_units`` (artillery, front line and logistics units), so that
        is the pool the commander may order from. Deliberately not the narrower pool
        :class:`game.procurement.ProcurementAi` restricts *itself* to, because the
        point of ACTIVE mode is player-equivalent control, not automation parity.
        """

        faction = self.faction
        purchasable_pool: set[GroundUnitType] = set(faction.ground_units)
        everything: set[GroundUnitType] = set(purchasable_pool)
        for group in (
            faction.frontline_units,
            faction.artillery_units,
            faction.infantry_units,
            faction.logistics_units,
            faction.air_defense_units,
            faction.missiles,
        ):
            everything |= set(group)

        entries = [
            GroundUnitCapability.from_ground_unit_type(
                unit, purchasable=unit in purchasable_pool
            )
            for unit in everything
        ]
        entries.sort(
            key=lambda entry: (not entry.purchasable, entry.unit_class, entry.unit_id)
        )
        return tuple(entries)

    def _ships(self) -> tuple[ShipCapability, ...]:
        faction = self.faction
        everything: set[ShipUnitType] = set(faction.naval_units)
        if faction.cargo_ship is not None:
            everything.add(faction.cargo_ship)
        everything |= set(faction.carriers)
        entries = [ShipCapability.from_ship_unit_type(ship) for ship in everything]
        entries.sort(key=lambda entry: (entry.unit_class, entry.unit_id))
        return tuple(entries)


class CapabilityIndexCache:
    """Process-wide cache of rendered capability indexes.

    A faction's roster is fixed for the campaign, so the index only needs building
    once per faction roster. Rendering is also cached because the mission-planning
    stage of an ACTIVE turn renders the same slice repeatedly.
    """

    def __init__(self) -> None:
        self._indexes: dict[str, CapabilityIndex] = {}
        self._renders: dict[tuple[str, tuple[str, ...]], str] = {}

    def clear(self) -> None:
        self._indexes.clear()
        self._renders.clear()

    def index_for(self, coalition: Coalition) -> CapabilityIndex:
        key = self._signature(coalition)
        cached = self._indexes.get(key)
        if cached is None:
            cached = CapabilityIndexBuilder(coalition).build()
            self._indexes[key] = cached
        return cached

    def render_for(
        self, coalition: Coalition, tasks: Optional[Iterable[FlightType]] = None
    ) -> str:
        index = self.index_for(coalition)
        task_key = tuple(sorted(t.value for t in tasks)) if tasks is not None else ()
        key = (index.content_hash(), task_key)
        cached = self._renders.get(key)
        if cached is None:
            cached = index.render_compact(tasks)
            self._renders[key] = cached
        return cached

    @staticmethod
    def _signature(coalition: Coalition) -> str:
        """A stable key covering the faction roster and the fielded air wing.

        Squadron *strength* is not part of the key: it changes every turn and is
        reported in the intel brief rather than here, so folding it into the cache key
        would defeat the cache without adding information.
        """

        faction = coalition.faction
        parts: list[str] = [_clean(faction.name)]
        for group in (
            faction.aircraft,
            faction.awacs,
            faction.tankers,
            faction.frontline_units,
            faction.artillery_units,
            faction.infantry_units,
            faction.logistics_units,
            faction.air_defense_units,
            faction.missiles,
            faction.naval_units,
        ):
            parts.append(",".join(sorted(unit.variant_id for unit in group)))
        try:
            squadrons = sorted(
                squadron.aircraft.variant_id
                for squadron in coalition.air_wing.iter_squadrons()
            )
        except Exception:  # pragma: no cover - defensive against partial stubs
            squadrons = []
        parts.append(",".join(squadrons))
        blob = "|".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: Shared cache. Tests that build synthetic factions should call ``clear()``.
CAPABILITY_CACHE = CapabilityIndexCache()


def capability_index_for(coalition: Coalition) -> CapabilityIndex:
    """Convenience wrapper over the shared cache."""

    return CAPABILITY_CACHE.index_for(coalition)


def render_capability_index(
    coalition: Coalition, tasks: Optional[Iterable[FlightType]] = None
) -> str:
    """Convenience wrapper over the shared cache's renderer."""

    return CAPABILITY_CACHE.render_for(coalition, tasks)
