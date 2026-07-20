import type { Skill } from '../types';
import { CodeBlock } from './CodeBlock';

interface SkillLibraryProps {
  skills: Skill[];
  selectedSkill: Skill;
  onSelectSkill: (skillId: string) => void;
}

export function SkillLibrary({ skills, selectedSkill, onSelectSkill }: SkillLibraryProps) {
  return (
    <section className="section" id="library">
      <div className="section-heading">
        <p className="eyebrow">02 / Skill Library</p>
        <h2>Reusable Skill Library</h2>
        <p>Browse structured Skills and inspect the JSON contract passed to downstream modules.</p>
      </div>

      <div className="library-layout">
        <aside className="skill-list" aria-label="Skill list">
          {skills.map((skill) => (
            <button
              key={skill.id}
              type="button"
              className={skill.id === selectedSkill.id ? 'skill-list-item active' : 'skill-list-item'}
              onClick={() => onSelectSkill(skill.id)}
            >
              <span>{skill.id}</span>
              <small>{skill.task_type}</small>
            </button>
          ))}
        </aside>

        <div className="library-detail">
          <div className="detail-header">
            <div>
              <p className="eyebrow">{selectedSkill.task_type}</p>
              <h3>{selectedSkill.name}</h3>
            </div>
            <code>{selectedSkill.executable_api}</code>
          </div>
          <p>{selectedSkill.description}</p>
          <CodeBlock code={JSON.stringify(selectedSkill, null, 2)} />
        </div>
      </div>
    </section>
  );
}
