export function Comparison() {
  return (
    <section className="section" id="comparison">
      <div className="section-heading">
        <p className="eyebrow">06 / Comparison</p>
        <h2>Traditional Pipeline vs. World2Skills</h2>
        <p>Skill-centered structure separates reusable knowledge from task-specific training data.</p>
      </div>

      <div className="comparison-grid">
        <article className="comparison-card traditional">
          <h3>Traditional Embodied AI</h3>
          <p className="pipeline">Data -&gt; Model -&gt; Task Execution</p>
          <ul>
            <li>new task requires new data</li>
            <li>new environment requires retraining</li>
            <li>weak interpretability</li>
          </ul>
        </article>

        <article className="comparison-card world">
          <h3>World2Skills</h3>
          <p className="pipeline">Skill -&gt; Data Generation / Rule Generation -&gt; Policy Learning -&gt; Execution</p>
          <ul>
            <li>reusable skill knowledge</li>
            <li>automatic data template generation</li>
            <li>interpretable rule generation</li>
            <li>better task and scene generalization</li>
          </ul>
        </article>
      </div>
    </section>
  );
}
