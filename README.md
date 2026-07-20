# World2Skills Demo

World2Skills Demo is a single-page React + Vite + TypeScript prototype for presenting a skill-centric embodied intelligence workflow:

```text
Skill Schema -> Skill Library -> Data Generation -> Rule Generation -> Policy / Execution Simulation
```

The demo is fully local. It does not require a backend service, real robot interface, or external API.

## Install

```bash
npm install
```

## Run

```bash
npm run dev
```

Open the local Vite URL printed in the terminal.

## Verify

```bash
npm test
npm run build
```

## Demo Walkthrough

1. Start at the hero workflow: Environment Experience -> Skill Abstraction -> Data / Rule Generation -> Policy Learning -> New Task Generalization.
2. Review the Skill Schema section to show the unified fields shared by downstream modules.
3. Use the Skill Library list to inspect JSON for `detect_object`, `move_to_object`, `pick_up_object`, `place_object`, `open_drawer`, and `put_object_into_container`.
4. Select a Skill and generate five synthetic training data configurations from `data_generation_hooks`.
5. Generate IF-THEN rules and inspect the simple Behavior Tree view.
6. Run a tabletop task simulation and watch procedure steps update symbolic state.
7. Close with the comparison table and related-work cards.

## Project Structure

```text
src/
  components/            React sections and visual widgets
  data/                  Static Skill library and related-work cards
  utils/                 Tested data, rule, and simulation logic
  types.ts               Shared TypeScript interfaces
  styles.css             Presentation styling and animation
```

## Notes

- `generateDataConfigs(skill)` derives scene variation fields from `data_generation_hooks`.
- `generateRules(skill)` derives precondition, effect, and recovery rules from Skill fields.
- `applySimulationStep(state, stepId)` updates symbolic task state for the CSS/SVG tabletop simulator.
