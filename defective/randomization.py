"""Randomisation that is present but describes an impossible world.

Separated from the env cfg on purpose: the defect here is not that
randomisation is missing — the whole-repo rule would catch that — it is that a
range someone typed includes a value no physical quantity takes.

The simulator will not refuse it. It integrates a negative mass and produces
confident nonsense, which is the same failure mode the asset gate's inertia
checks exist for, one layer up.
"""

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

import mdp


@configclass
class BrokenRandomisationCfg:
    # DEFECT: a sign flip. `(-0.5, 2.0)` was probably meant as an *additive*
    # range around the nominal mass, but this call multiplies.
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"mass_range": (-0.5, 2.0), "operation": "scale"},
    )

    # DEFECT: friction cannot be negative under any surface model.
    material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"static_friction_range": (-0.2, 1.4)},
    )
