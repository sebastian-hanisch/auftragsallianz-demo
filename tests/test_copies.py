"""Die wörtliche Kopie des Zufallsgenerators (SplitMix64) aus ab-test-demo: gleiche Datei bis auf die Kopfzeile, und der Strom ist der dokumentierte."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"aal_rng.py": ("ab-test-demo", "abt_rng.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0 liefert die bekannten ersten Werte der Konvention (Vigna); der Block-Generator liefert dieselben Zahlen."""
    from aal_rng import SplitMix64, uniform_block
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
    r2 = SplitMix64(7)
    assert uniform_block(7, 4).tolist() == [r2.uniform() for _ in range(4)]


def test_no_other_random_source_is_used_for_findings():
    """Kein `random`, kein `numpy.random` im Kern (nur im Zufalls-Seed-Button der Oberfläche, der keinen Befund erzeugt)."""
    for name in ("aal_scenario.py", "aal_algorithm.py", "aal_game.py", "aal_pool.py", "aal_evaluation.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "import random" not in text and "numpy.random" not in text and "np.random" not in text, name
