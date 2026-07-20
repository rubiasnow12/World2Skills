---
name: unprotected-left-turn
description: Use when the ego must turn left across oncoming traffic at an unsignalized intersection, yielding to cross and oncoming vehicles.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Unprotected Left Turn

## When to use
The ego approaches an unsignalized intersection intending to turn left, and
must cross the path of oncoming and cross traffic that has right of way.

## Procedure
1. Stop or slow at the intersection entry and observe oncoming traffic.
2. Estimate the time gap to the nearest oncoming vehicle.
3. If the gap is below the yield threshold, keep waiting (decelerate/hold).
4. When the gap is sufficient, accelerate through the conflict zone.
5. Once past the conflict zone, resume normal speed on the target lane.

## Reasoning cues
- A larger relative speed of oncoming traffic requires a larger accepted gap.
- Never commit on a marginal gap; waiting one more cycle is cheap, a collision
  is not.

## Failure modes & recovery
- Committing on an insufficient gap risks a side collision - abort by holding
  if the ego has not yet entered the conflict zone.
- Over-cautious freezing blocks the intersection - if no vehicle is within the
  yield gap, proceed rather than waiting indefinitely.
