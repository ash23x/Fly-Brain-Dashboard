import math

import numpy as np

from compass_sim import CompassSim
from mushroom_sim import MushroomSim


def test_compass_loads_and_steps(compass_dir):
    sim = CompassSim(compass_dir, seed=0)
    assert sim.n == 18 * 4
    assert len(sim.epg_idx) == 36 and len(sim.pen_left) == 9 and len(sim.pen_right) == 9
    sim.cue(90.0, strength=2.0, steps=60)
    for _ in range(300):
        out = sim.step()
    assert 0.0 <= out["heading"] < 360.0
    assert 0.0 <= out["strength"] <= 1.0
    assert all(isinstance(i, int) for i in out["spikes"])
    assert not np.isnan(sim.v).any()


def test_compass_turn_is_clamped_and_ramped(compass_dir):
    sim = CompassSim(compass_dir, seed=0)
    sim.set_turn(5.0)
    assert sim.turn_cmd == 1.0
    sim.step()
    assert 0.0 < sim.turn < 1.0          # ramps toward the command, never jumps
    sim.reset()
    assert sim.turn == 0.0 and sim.turn_cmd == 0.0


def test_compass_reignites_after_reset(compass_dir):
    sim = CompassSim(compass_dir, seed=0, reignite_after=5)
    sim.reset()
    for _ in range(200):
        sim.step()
    assert sim.reignitions >= 1


def test_mushroom_odour_code_and_learning(mushroom_dir):
    sim = MushroomSim(mushroom_dir, seed=0)
    assert set(sim.odours) == {"A", "B", "C"}
    assert sim.n_plastic > 0 and math.isclose(sim.factor.max(), 1.0)
    sim.set_odour("A")
    for _ in range(200):
        out = sim.step()
    assert out["odour"] == "A"
    assert out["n_depressed"] == 0           # no dopamine, no learning
    sim.set_reward(1.0)
    for _ in range(200):
        out = sim.step()
    assert out["n_depressed"] > 0            # odour + reward depresses KC->MBON synapses
    assert sim.factor.min() >= sim.f_min
    trace = sim.mbon_trace()
    assert len(trace) == 6 and max(trace) > 0
    sim.forget()
    assert math.isclose(sim.factor.min(), 1.0)
    assert sim.mbon_trace() == [0.0] * 6


def test_mushroom_bad_odour_rejected(mushroom_dir):
    sim = MushroomSim(mushroom_dir, seed=0)
    try:
        sim.set_odour("Z")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown odour should raise")
