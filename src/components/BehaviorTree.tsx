import type { Skill } from '../types';

interface BehaviorTreeProps {
  skill: Skill;
}

export function BehaviorTree({ skill }: BehaviorTreeProps) {
  return (
    <div className="behavior-tree" aria-label="Behavior tree">
      <div className="bt-node root">Sequence</div>
      <div className="bt-connector" />
      <div className="bt-children">
        <div className="bt-node">
          <strong>Check Preconditions</strong>
          <span>{skill.preconditions.slice(0, 3).join(' AND ')}</span>
        </div>
        <div className="bt-node active">
          <strong>Execute Skill</strong>
          <span>{skill.executable_api}</span>
        </div>
        <div className="bt-node">
          <strong>Check Success</strong>
          <span>{skill.success_conditions.join(' OR ')}</span>
        </div>
        <div className="bt-node fallback">
          <strong>Recovery Fallback</strong>
          <span>{skill.recovery_rules.slice(0, 2).join(' / ')}</span>
        </div>
      </div>
    </div>
  );
}
