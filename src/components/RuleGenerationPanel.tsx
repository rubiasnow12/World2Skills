import { useEffect, useMemo, useState } from 'react';
import type { RuleSet, Skill } from '../types';
import { generateRules } from '../utils/generateRules';
import { BehaviorTree } from './BehaviorTree';

interface RuleGenerationPanelProps {
  selectedSkill: Skill;
}

export function RuleGenerationPanel({ selectedSkill }: RuleGenerationPanelProps) {
  const generatedRules = useMemo(() => generateRules(selectedSkill), [selectedSkill]);
  const [rules, setRules] = useState<RuleSet>(generatedRules);

  useEffect(() => {
    setRules(generatedRules);
  }, [generatedRules]);

  function handleGenerate() {
    setRules(generateRules(selectedSkill));
  }

  return (
    <section className="section" id="rule-generation">
      <div className="section-heading with-action">
        <div>
          <p className="eyebrow">04 / Rule Generation</p>
          <h2>IF-THEN Rules and Behavior Tree</h2>
          <p>Symbolic fields become applicability, effect, and recovery rules for execution control.</p>
        </div>
        <button type="button" className="primary-button" onClick={handleGenerate}>
          Generate Rules
        </button>
      </div>

      <div className="rules-layout">
        <div className="rule-columns">
          <RuleGroup title="Preconditions" lines={rules.preconditions} />
          <RuleGroup title="Effects" lines={rules.effects} />
          <RuleGroup title="Recovery" lines={rules.recovery} />
        </div>
        <BehaviorTree skill={selectedSkill} />
      </div>
    </section>
  );
}

interface RuleGroupProps {
  title: string;
  lines: string[];
}

function RuleGroup({ title, lines }: RuleGroupProps) {
  return (
    <article className="rule-card">
      <h3>{title}</h3>
      {lines.map((line) => (
        <code key={line}>{line}</code>
      ))}
    </article>
  );
}
