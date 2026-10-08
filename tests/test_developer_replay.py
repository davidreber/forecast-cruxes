"""Developer replay: recorded matrices through the CPU arithmetic, exactly.

This is an internal test of the port, not verification of the paper. The
fixture also holds the recorded baselines; their replay lives with the
baseline procedures in the verification repository. It takes
three recorded score caches of the paper's March Madness headline pool and the
numbers gathered from them at the time, and replays the whole CPU side with the
one thing the package now does differently put back: the recorded runs seeded
every permutation null with the constant 42, so this test passes a seed
function that returns 42. Everything else is the package's own default.

Equality is exact. Both sides run the same arithmetic on the same inputs, so a
tolerance would only hide a real change. What "verified" means for the paper is
the tolerance based reviewer simulation in the separate verification
repository, which rescores from the data artifact on a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cruxes import ScoreMatrices, set_inclusion_curve

FIXTURE = Path(__file__).parent / "fixtures" / "recorded_replay_objects59.json"


def constant_seed(question_id, contrasts_per_side):
    """What the recorded runs did: one seed for every null."""
    return 42


@pytest.fixture(scope="module")
def recorded():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def questions(recorded):
    return {
        question: ScoreMatrices.from_mapping(data)
        for question, data in recorded["questions"].items()
    }


def test_the_fixture_covers_a_zero_overlap_question(recorded):
    shared = [
        q["set_inclusion"]["25"]["shared_frac"] for q in recorded["questions"].values()
    ]
    assert 0.0 in shared and max(shared) >= 0.4


@pytest.mark.parametrize("question", ["akron-vs-texas-tech", "arizona-vs-purdue", "cal-baptist-vs-kansas"])
def test_set_inclusion_replays_exactly(recorded, questions, question):
    curve = set_inclusion_curve(
        questions[question],
        recorded["max_contrasts_per_side"],
        question,
        permutation_seed=constant_seed,
    )
    expected = recorded["questions"][question]["set_inclusion"]
    assert sorted(curve) == sorted(int(k) for k in expected)
    for count, record in curve.items():
        assert json.loads(json.dumps(record)) == expected[str(count)], count


def test_the_default_seeding_is_not_the_recorded_one(recorded, questions):
    """The package's own default draws different nulls; that is the point of it."""
    question = "akron-vs-texas-tech"
    curve = set_inclusion_curve(questions[question], 25, question)
    expected = recorded["questions"][question]["set_inclusion"]["25"]
    assert curve[25]["shared_frac"] == expected["shared_frac"]
    assert curve[25]["permutation"] != expected["permutation"]
