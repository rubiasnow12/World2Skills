# World2Skills Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-runnable React + Vite + TypeScript single-page demo for the World2Skills skill-centric embodied intelligence prototype.

**Architecture:** Keep domain data and reusable generation/simulation logic separate from React presentation components. The app uses local static Skill JSON, pure utility functions for data/rule/simulation behavior, and modular sections for the demo flow.

**Tech Stack:** React, Vite, TypeScript, Vitest, CSS.

---

### Task 1: Project Scaffold

**Files:**
- Create: `package.json`
- Create: `index.html`
- Create: `vite.config.ts`
- Create: `tsconfig.json`
- Create: `tsconfig.node.json`
- Create: `src/vite-env.d.ts`

- [ ] Add the Vite React TypeScript package scripts and compiler configuration.
- [ ] Install dependencies with `npm install`.

### Task 2: Tested Domain Logic

**Files:**
- Create: `src/utils/generation.test.ts`
- Create: `src/utils/simulation.test.ts`
- Create: `src/types.ts`
- Create: `src/data/skills.ts`
- Create: `src/utils/generateDataConfigs.ts`
- Create: `src/utils/generateRules.ts`
- Create: `src/utils/simulation.ts`

- [ ] Write failing tests for data config generation, rule generation, and simulation state transitions.
- [ ] Implement typed Skill data and utilities until tests pass.

### Task 3: React Demo Sections

**Files:**
- Create: `src/main.tsx`
- Create: `src/App.tsx`
- Create: `src/components/*.tsx`
- Create: `src/styles.css`
- Create: `src/data/relatedWork.ts`

- [ ] Build Hero, Skill Schema, Skill Library, Data Generation, Rule Generation, Execution Simulation, Comparison, and Related Work sections.
- [ ] Use SVG/CSS for the flow diagram, behavior tree, and tabletop simulation animation.

### Task 4: Documentation and Verification

**Files:**
- Create: `README.md`

- [ ] Document install, run, build, and demo walkthrough.
- [ ] Run `npm test`.
- [ ] Run `npm run build`.
