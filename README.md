# reference-task

A minimal but realistic Isaac Lab task, serving two purposes at once.

**The shape a SimLiverse training repo takes.** Copy `clean/` and change the
task. It shows the division the platform assumes: task code, scene, sensors,
rewards and randomisation live in *your* repo; the robot and scene USD live in
the asset registry and are named by `simliverse.yaml`.

**The fixture the repo gate is tested against.** `simliverse-core` vendors this
tree at `api/tests/fixtures/reference_task/` and runs the rule pack over both
variants. The rules parse with `ast`, so the files here are real Python rather
than stubs — a fixture that does not parse proves nothing about a parser.

> The two copies are kept identical by hand today. If they drift, the one in
> `simliverse-core` is what the tests actually run, and this one is the
> documentation that quietly stopped being true. Worth automating before that
> happens rather than after.

## What lives where

| Piece | Here | Asset registry |
|---|---|---|
| Task code, rewards, terminations | ✅ | |
| Scene config, sensor declarations | ✅ | |
| Domain randomisation ranges | ✅ | |
| Algorithm hyperparameters | ✅ | |
| Robot USD / URDF | | ✅ |
| Scene / terrain USD | | ✅ |
| Props | | ✅ |

Code and assets have different lifecycles, which is why the split is where it
is. A URDF is reviewed by whoever owns the hardware, gated for inertia and
joint limits, and versioned by its bytes. Keeping it in git would make a run's
robot "whatever was on that branch" — not something anyone reproduces a year
later.

## The two trees

`clean/` is what a healthy task looks like. **The repo gate finds nothing in
it**, and a test asserts exactly that. This matters more than the defect
detection: a gate that fires on correct code gets switched off, and then it
catches nothing at all.

`defective/` carries one instance of each defect the static rules can see. None
is a syntax error and none would fail to import — that is the point. Every one
produces a run that starts, trains, reports a number, and is quietly worthless.

| Defect | What it costs you |
|---|---|
| Reward term with `weight=0.0` | Reported as part of the reward, contributes nothing to it |
| Reward weights spanning >3 orders of magnitude | The small term cannot move the gradient against the large one |
| `mass_range=(-0.5, 2.0)` with `operation="scale"` | Negative mass. The simulator integrates it rather than refusing |
| Sensor `prim_path` not scene-rooted | Attaches to nothing; empty frames for the whole run, no error |
| No terminations | Episodes never end, reward plateaus, nothing in the logs says why |
| No `EventCfg` | No domain randomisation — the policy fits one exact set of sim parameters |

Note that `clean/` *also* has a negative `mass_range`, `(-2.0, 2.0)` — with
`operation="add"`, where it means ±2 kg around nominal and is correct. The gate
reads the operation before objecting. An earlier version did not, and fired on
the healthy tree.

## What the gate does not check

Whether the reward is *good* — whether it produces the behaviour you wanted.
That is research, and no static rule can see it. The gate reports that a term is
numerically invisible; it never claims a reward is wrong.

Half the interesting checks need the *resolved* config object, because
inheritance and `__post_init__` mean the source does not contain the final
values. Those belong to a smoke stage that instantiates the env in the
container — cheap next to a training run, but a job rather than a parse.
