export type SkillTaskType = 'perception' | 'navigation' | 'manipulation' | 'placement' | 'container_interaction';

export interface Skill {
  id: string;
  name: string;
  description: string;
  task_type: SkillTaskType;
  inputs: string[];
  preconditions: string[];
  parameters: string[];
  procedure: string[];
  effects: string[];
  success_conditions: string[];
  failure_conditions: string[];
  recovery_rules: string[];
  data_generation_hooks: string[];
  rule_generation_hooks: string[];
  executable_api: string;
}

export interface SkillSchemaField {
  field: keyof Skill;
  purpose: string;
}

export interface DataConfig {
  scene_id: string;
  skill_id: string;
  object_shape: string;
  object_pose: string;
  lighting: string;
  occlusion: string;
  distractors: string;
  surface_height: string;
  expected_success_condition: string;
  generated_instruction: string;
}

export interface RuleSet {
  preconditions: string[];
  effects: string[];
  recovery: string[];
}

export interface SimulationState {
  object_visible: boolean;
  object_reachable: boolean;
  gripper_empty: boolean;
  object_in_gripper: boolean;
  object_in_container: boolean;
  drawer_open: boolean;
  task_success: boolean;
}

export interface SimulationStep {
  id: string;
  label: string;
  skillId: string;
}

export interface SimulationTask {
  id: string;
  name: string;
  description: string;
  steps: SimulationStep[];
}
