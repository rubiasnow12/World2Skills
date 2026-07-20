---
name: follow-keep-distance
description: Use when the ego follows a lead vehicle in the same lane and must keep a safe time-gap headway, matching the lead vehicle's speed without collision.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Follow and Keep Distance

## When to use
The ego is behind a lead vehicle in the same lane with no intent or opportunity
to overtake, and must maintain a safe following distance.

## Procedure
1. Measure the time gap (headway) to the lead vehicle.
2. If the gap is smaller than the target, decelerate.
3. If the gap is larger than the target and below target speed, accelerate.
4. Otherwise hold speed to track the lead vehicle.

## Reasoning cues
- Headway in time, not distance, is what stays safe across speeds.
- React early and gently; late hard braking propagates instability upstream.

## Failure modes & recovery
- Tailgating leaves no room to brake - restore the target gap by decelerating.
- Over-braking on a transient gap dip wastes speed - filter brief fluctuations.
