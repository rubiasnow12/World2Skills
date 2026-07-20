---
name: facilitate-highway-merge
description: Use when the ego is on the main highway approaching an on-ramp and should keep speed while making room for a vehicle merging in from the access ramp.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Facilitate Highway Merge

## When to use
The ego drives on the main highway near an on-ramp where another vehicle is
merging in. The ego should maintain high speed and avoid collision while
creating room for the merging vehicle.

## Procedure
1. Detect a vehicle on the access ramp intending to merge.
2. Estimate whether the ego and the merging vehicle will conflict.
3. If a conflict is likely, create room: decelerate slightly or change to the
   left lane if it is clear.
4. If no conflict, maintain speed.
5. Resume normal cruising once the merging vehicle has settled in.

## Reasoning cues
- A small early speed adjustment opens a gap more smoothly than late braking.
- Only move left if that lane is clearly clear; do not trade one conflict for
  another.

## Failure modes & recovery
- Ignoring the merging vehicle forces a late conflict - adjust early.
- Braking too hard disrupts following traffic - prefer a gentle gap-opening.
