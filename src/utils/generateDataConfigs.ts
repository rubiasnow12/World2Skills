import type { DataConfig, Skill } from '../types';

const variableBanks = {
  object_shape: ['red cup', 'small cube', 'fragile bowl', 'metal can', 'soft pouch'],
  object_pose: ['upright near table center', 'tilted at the edge', 'rotated beside a box', 'lying on its side', 'near the container'],
  lighting: ['bright overhead lighting', 'side lighting', 'warm dim lighting', 'shadowed lighting', 'high contrast lighting'],
  occlusion: ['no occlusion', 'partial occlusion', 'handle-side occlusion', 'front-edge occlusion', 'heavy clutter occlusion'],
  distractors: ['no distractors', 'two nearby blocks', 'cluttered tools', 'similar colored cups', 'mixed small objects'],
  surface_height: ['low table', 'standard table', 'raised platform', 'drawer shelf', 'uneven support surface'],
};

const hookToField: Record<string, keyof typeof variableBanks> = {
  vary_object_shape: 'object_shape',
  vary_object_pose: 'object_pose',
  vary_lighting: 'lighting',
  vary_occlusion: 'occlusion',
  vary_distractors: 'distractors',
  vary_surface_height: 'surface_height',
};

const defaults: Omit<DataConfig, 'scene_id' | 'skill_id' | 'expected_success_condition' | 'generated_instruction'> = {
  object_shape: 'red cup',
  object_pose: 'upright near table center',
  lighting: 'bright overhead lighting',
  occlusion: 'no occlusion',
  distractors: 'no distractors',
  surface_height: 'standard table',
};

function valueFor(field: keyof typeof variableBanks, index: number): string {
  const values = variableBanks[field];
  return values[index % values.length];
}

function instructionFor(skill: Skill, config: Omit<DataConfig, 'scene_id' | 'skill_id' | 'expected_success_condition' | 'generated_instruction'>): string {
  if (skill.id === 'pick_up_object') {
    if (config.object_shape.includes('fragile')) {
      return `Lift the ${config.object_shape} from the ${config.surface_height} with a reduced force limit.`;
    }

    if (config.distractors.includes('cluttered')) {
      return `Grasp the ${config.object_shape} from a cluttered surface.`;
    }

    return `Pick up the ${config.object_shape} from the ${config.surface_height} under ${config.occlusion}.`;
  }

  if (skill.id === 'open_drawer') {
    return `Open the drawer under ${config.lighting} while the handle has ${config.occlusion}.`;
  }

  if (skill.id === 'put_object_into_container') {
    return `Place the ${config.object_shape} into the container with ${config.distractors} nearby.`;
  }

  return `Execute ${skill.name.toLowerCase()} for the ${config.object_shape} at ${config.object_pose}.`;
}

export function generateDataConfigs(skill: Skill, count = 5): DataConfig[] {
  const activeFields = skill.data_generation_hooks
    .map((hook) => hookToField[hook])
    .filter((field): field is keyof typeof variableBanks => Boolean(field));

  return Array.from({ length: count }, (_, index) => {
    const generatedValues = activeFields.reduce(
      (accumulator, field) => ({
        ...accumulator,
        [field]: valueFor(field, index),
      }),
      defaults,
    );

    return {
      scene_id: `scene_${skill.id}_${String(index + 1).padStart(3, '0')}`,
      skill_id: skill.id,
      ...generatedValues,
      expected_success_condition: skill.success_conditions[0] ?? 'skill_succeeded',
      generated_instruction: instructionFor(skill, generatedValues),
    };
  });
}
