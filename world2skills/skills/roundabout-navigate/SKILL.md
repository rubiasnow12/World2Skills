---
name: roundabout-navigate
description: Use when the ego must enter and traverse a roundabout, yielding to circulating traffic on entry and exiting at the intended offramp.
metadata:
  schema_version: "0.1"
  representation: skill.yaml
---

# Roundabout Navigate

## When to use
The ego approaches a roundabout and must yield to circulating traffic, enter
when a gap allows, circulate, and leave at the intended exit.

## Procedure
1. At the entry, observe circulating traffic and estimate the entry gap.
2. If the gap is insufficient, wait at the yield line.
3. When the gap is sufficient, enter the roundabout.
4. Circulate, keeping distance from the vehicle ahead.
5. Exit at the intended offramp.

## Reasoning cues
- Circulating traffic has priority; entry is the main conflict point.
- Keep a steady speed while circulating; hesitation inside the ring is riskier
  than a clean commit.

## Failure modes & recovery
- Entering on a marginal gap forces circulating traffic to brake - wait for a
  clear gap.
- Missing the intended exit means another loop - track the exit early.
