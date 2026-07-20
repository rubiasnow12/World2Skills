import { describe, expect, it } from 'vitest';
import {
  applySimulationStep,
  createInitialSimulationState,
  simulationTasks,
} from './simulation';

describe('simulation state transitions', () => {
  it('completes the pick and place task through ordered skill steps', () => {
    const task = simulationTasks.find((item) => item.id === 'pick_place_object');

    expect(task).toBeDefined();

    const finalState = task!.steps.reduce(
      (state, step) => applySimulationStep(state, step.id),
      createInitialSimulationState(),
    );

    expect(finalState.object_visible).toBe(true);
    expect(finalState.object_reachable).toBe(true);
    expect(finalState.gripper_empty).toBe(true);
    expect(finalState.object_in_gripper).toBe(false);
    expect(finalState.object_in_container).toBe(true);
    expect(finalState.task_success).toBe(true);
  });

  it('opens a drawer before grasping in the drawer task', () => {
    const task = simulationTasks.find((item) => item.id === 'open_drawer_pick');

    expect(task).toBeDefined();
    expect(task!.steps[0].id).toBe('open_drawer');

    const afterOpenDrawer = applySimulationStep(createInitialSimulationState(), 'open_drawer');

    expect(afterOpenDrawer.drawer_open).toBe(true);
    expect(afterOpenDrawer.object_visible).toBe(true);
  });
});
