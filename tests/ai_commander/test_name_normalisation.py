"""Normalisation and de-duplication of enemy unit and weapon type names.

The intel channels collect the *same* real type under several spellings at once:
a raw DCS type id (``FA-18C_hornet``), a human display name from another channel
(``F/A-18C Hornet (Lot 20)``), or a weapon carried as its full internal path
(``weapons.shells.M256_120_HE``). :mod:`game.ai_commander.naming` collapses those
to a single clean name so the memory the commander reads is tidy. These tests pin
that behaviour, including the guarantee that a name is never dropped -- an
unresolved id merely degrades to a tidied form of itself.
"""

from __future__ import annotations

from game.ai_commander.naming import (
    dedupe_type_names,
    dedupe_weapon_names,
    normalise_type_name,
    normalise_weapon_name,
)


class TestTypeNameNormalisation:
    def test_raw_id_and_display_name_collapse_to_one(self) -> None:
        # A raw DCS id and its human display name are the same real aircraft and
        # must not appear twice; the cleaner display name wins.
        result = dedupe_type_names(["FA-18C_hornet", "F/A-18C Hornet (Lot 20)"])
        assert result == ("F/A-18C Hornet",)

    def test_order_does_not_matter(self) -> None:
        forward = dedupe_type_names(["FA-18C_hornet", "F/A-18C Hornet (Lot 20)"])
        reverse = dedupe_type_names(["F/A-18C Hornet (Lot 20)", "FA-18C_hornet"])
        assert forward == reverse == ("F/A-18C Hornet",)

    def test_ship_id_resolves_to_a_readable_display_name(self) -> None:
        # A terse raw ship id resolves through pydcs to a proper name.
        (name,) = dedupe_type_names(["PERRY"])
        assert "Perry" in name
        assert name != "PERRY"

    def test_unresolved_id_is_tidied_never_dropped(self) -> None:
        # An id with no pydcs entry keeps its identity, merely tidied.
        assert normalise_type_name("SOME_MADE_UP_TYPE") == "SOME MADE UP TYPE"
        assert dedupe_type_names(["SOME_MADE_UP_TYPE"]) == ("SOME MADE UP TYPE",)

    def test_blank_names_are_dropped(self) -> None:
        assert dedupe_type_names(["", "   ", "F-15C"]) == ("F-15C",)

    def test_list_is_capped(self) -> None:
        assert len(dedupe_type_names([f"TYPE-{i:03d}" for i in range(50)])) == 24


class TestWeaponNameNormalisation:
    def test_internal_weapon_path_is_stripped_and_tidied(self) -> None:
        assert normalise_weapon_name("weapons.shells.M256_120_HE") == "M256 120 HE"

    def test_already_clean_weapon_name_is_kept(self) -> None:
        assert normalise_weapon_name("AIM-120C") == "AIM-120C"

    def test_dedupe_weapon_names_normalises_and_sorts(self) -> None:
        result = dedupe_weapon_names(
            ["weapons.shells.M256_120_HE", "AIM-120C", "weapons.bombs.GBU_12"]
        )
        assert result == ("AIM-120C", "GBU 12", "M256 120 HE")

    def test_weapon_path_and_display_name_collapse(self) -> None:
        # The same weapon carried once as an internal path and once as a clean
        # display name is a single distinct weapon.
        result = dedupe_weapon_names(["weapons.missiles.AIM_120C", "AIM 120C"])
        assert result == ("AIM 120C",)

    def test_weapon_path_in_type_channel_is_still_tidied(self) -> None:
        # A weapon path that strays into the type channel is routed to the weapon
        # normaliser rather than dropped.
        assert normalise_type_name("weapons.shells.M256_120_HE") == "M256 120 HE"
