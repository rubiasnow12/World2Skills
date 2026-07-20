import { useMemo, useState } from 'react';
import { Comparison } from './components/Comparison';
import { DataGenerationPanel } from './components/DataGenerationPanel';
import { ExecutionSimulation } from './components/ExecutionSimulation';
import { Hero } from './components/Hero';
import { RelatedWork } from './components/RelatedWork';
import { RuleGenerationPanel } from './components/RuleGenerationPanel';
import { SkillLibrary } from './components/SkillLibrary';
import { SkillSchema } from './components/SkillSchema';
import { skillLibrary } from './data/skills';
import './styles.css';

export default function App() {
  const [selectedSkillId, setSelectedSkillId] = useState('pick_up_object');
  const selectedSkill = useMemo(
    () => skillLibrary.find((skill) => skill.id === selectedSkillId) ?? skillLibrary[0],
    [selectedSkillId],
  );

  return (
    <main>
      <Hero />
      <SkillSchema />
      <SkillLibrary
        skills={skillLibrary}
        selectedSkill={selectedSkill}
        onSelectSkill={setSelectedSkillId}
      />
      <DataGenerationPanel selectedSkill={selectedSkill} />
      <RuleGenerationPanel selectedSkill={selectedSkill} />
      <ExecutionSimulation />
      <Comparison />
      <RelatedWork />
    </main>
  );
}
