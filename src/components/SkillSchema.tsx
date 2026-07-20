import { pickUpObjectJson, skillSchemaFields } from '../data/skills';
import { CodeBlock } from './CodeBlock';

export function SkillSchema() {
  return (
    <section className="section" id="schema">
      <div className="section-heading">
        <p className="eyebrow">01 / Skill Schema</p>
        <h2>Unified Skill Representation</h2>
        <p>
          Each Skill exposes one typed interface for data synthesis, rule generation, and execution modules.
        </p>
      </div>

      <div className="schema-layout">
        <div className="schema-grid">
          {skillSchemaFields.map((item) => (
            <article key={item.field} className="schema-card">
              <h3>{item.field}</h3>
              <p>{item.purpose}</p>
            </article>
          ))}
        </div>
        <CodeBlock label="Example schema instance" code={pickUpObjectJson} />
      </div>
    </section>
  );
}
