import type { RuleSet, Skill } from '../types';

export function generateRules(skill: Skill): RuleSet {
  const preconditionRule = `IF ${skill.preconditions.join(' AND ')} THEN skill ${skill.id} is applicable`;

  const effectRules = skill.effects.map(
    (effect) => `AFTER ${skill.id} succeeds SET ${effect} = true`,
  );

  const recoveryRules = skill.failure_conditions.map((failure, index) => {
    const recovery = skill.recovery_rules[index] ?? skill.recovery_rules[0] ?? 'request_human_intervention';
    return `IF ${failure} THEN ${recovery}`;
  });

  return {
    preconditions: [preconditionRule],
    effects: effectRules,
    recovery: recoveryRules,
  };
}
