---
name: lane-change-overtake
description: Use when a slower vehicle ahead blocks progress on a multi-lane highway and the ego should change lanes to overtake when an adjacent lane gap is safe.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Lane Change Overtake

## When to use
The ego is on a multi-lane highway behind a slower lead vehicle, an adjacent
lane is available, and overtaking would restore desired speed.

## Procedure
1. Detect that the lead vehicle is slower than the ego's target speed.
2. Check the adjacent lane for a safe gap (front and rear).
3. If a gap exists, change lanes toward it; otherwise keep following.
4. After the lane change, accelerate to pass the overtaken vehicle.
5. Resume cruising once clear.

## Reasoning cues
- Prefer the left lane for overtaking; a rear vehicle closing fast shrinks the
  usable gap.
- Abort the lane change if the target gap closes before commitment.

## Failure modes & recovery
- Changing into an occupied gap causes a side collision - re-check the rear gap
  immediately before committing.
- Oscillating between lanes wastes time - only change when the speed gain is
  worth it.
