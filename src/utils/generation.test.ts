import { describe, expect, it } from 'vitest';
import { skillLibrary } from '../data/skills';
import { generateDataConfigs } from './generateDataConfigs';
import { generateRules } from './generateRules';

describe('generateDataConfigs', () => {
  it('creates five schema-driven synthetic configurations for a selected skill', () => {
    const pickUpSkill = skillLibrary.find((skill) => skill.id === 'pick_up_object');

    expect(pickUpSkill).toBeDefined();

    const configs = generateDataConfigs(pickUpSkill!, 5);

    expect(configs).toHaveLength(5);
    expect(configs[0]).toMatchObject({
      scene_id: 'scene_pick_up_object_001',
      skill_id: 'pick_up_object',
      expected_success_condition: 'object_lifted',
    });
    expect(configs.every((config) => config.generated_instruction.length > 20)).toBe(true);
    expect(new Set(configs.map((config) => config.object_shape)).size).toBeGreaterThan(1);
    expect(new Set(configs.map((config) => config.occlusion)).size).toBeGreaterThan(1);
  });
});

describe('generateRules', () => {
  it('turns preconditions, effects, and recovery arrays into readable rule groups', () => {
    const pickUpSkill = skillLibrary.find((skill) => skill.id === 'pick_up_object');

    expect(pickUpSkill).toBeDefined();

    const rules = generateRules(pickUpSkill!);

    expect(rules.preconditions[0]).toBe(
      'IF object_visible AND object_reachable AND gripper_empty AND object_graspable THEN skill pick_up_object is applicable',
    );
    expect(rules.effects).toContain('AFTER pick_up_object succeeds SET object_in_gripper = true');
    expect(rules.effects).toContain('AFTER pick_up_object succeeds SET object_not_on_surface = true');
    expect(rules.recovery[0]).toBe('IF grasp_failed THEN retry_with_new_grasp_pose');
    expect(rules.recovery[2]).toBe('IF collision_detected THEN change_approach_direction');
  });
});
