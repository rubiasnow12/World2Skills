import type { SimulationState, SimulationStep, SimulationTask } from '../types';

export const simulationTasks: SimulationTask[] = [
  {
    id: 'pick_up_object',
    name: 'Task A: Pick up object',
    description: 'Detect, approach, grasp, and lift a visible target object.',
    steps: [
      { id: 'detect_object', label: 'detect_object', skillId: 'detect_object' },
      { id: 'move_to_object', label: 'move_to_object', skillId: 'move_to_object' },
      { id: 'estimate_grasp_pose', label: 'estimate_grasp_pose', skillId: 'pick_up_object' },
      { id: 'close_gripper', label: 'close_gripper', skillId: 'pick_up_object' },
      { id: 'lift_object', label: 'lift_object', skillId: 'pick_up_object' },
    ],
  },
  {
    id: 'pick_place_object',
    name: 'Task B: Pick and place object into container',
    description: 'Lift an object and place it into the target container.',
    steps: [
      { id: 'detect_object', label: 'detect_object', skillId: 'detect_object' },
      { id: 'move_to_object', label: 'move_to_object', skillId: 'move_to_object' },
      { id: 'estimate_grasp_pose', label: 'estimate_grasp_pose', skillId: 'pick_up_object' },
      { id: 'close_gripper', label: 'close_gripper', skillId: 'pick_up_object' },
      { id: 'lift_object', label: 'lift_object', skillId: 'pick_up_object' },
      { id: 'place_object', label: 'place_object', skillId: 'place_object' },
    ],
  },
  {
    id: 'open_drawer_pick',
    name: 'Task C: Open drawer then pick object',
    description: 'Open a drawer, detect the revealed object, and lift it.',
    steps: [
      { id: 'open_drawer', label: 'open_drawer', skillId: 'open_drawer' },
      { id: 'detect_object', label: 'detect_object', skillId: 'detect_object' },
      { id: 'move_to_object', label: 'move_to_object', skillId: 'move_to_object' },
      { id: 'estimate_grasp_pose', label: 'estimate_grasp_pose', skillId: 'pick_up_object' },
      { id: 'close_gripper', label: 'close_gripper', skillId: 'pick_up_object' },
      { id: 'lift_object', label: 'lift_object', skillId: 'pick_up_object' },
    ],
  },
];

export function createInitialSimulationState(): SimulationState {
  return {
    object_visible: false,
    object_reachable: false,
    gripper_empty: true,
    object_in_gripper: false,
    object_in_container: false,
    drawer_open: false,
    task_success: false,
  };
}

export function applySimulationStep(state: SimulationState, stepId: SimulationStep['id']): SimulationState {
  switch (stepId) {
    case 'open_drawer':
      return {
        ...state,
        drawer_open: true,
        object_visible: true,
      };
    case 'detect_object':
      return {
        ...state,
        object_visible: true,
      };
    case 'move_to_object':
      return {
        ...state,
        object_reachable: true,
      };
    case 'estimate_grasp_pose':
      return state;
    case 'close_gripper':
      return {
        ...state,
        gripper_empty: false,
        object_in_gripper: true,
      };
    case 'lift_object':
      return {
        ...state,
        object_in_gripper: true,
        task_success: true,
      };
    case 'place_object':
      return {
        ...state,
        gripper_empty: true,
        object_in_gripper: false,
        object_in_container: true,
        task_success: true,
      };
    default:
      return state;
  }
}
