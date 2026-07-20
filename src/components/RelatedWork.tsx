import { relatedWorkItems } from '../data/relatedWork';

export function RelatedWork() {
  return (
    <section className="section" id="related-work">
      <div className="section-heading">
        <p className="eyebrow">07 / Related Work</p>
        <h2>Paper and Source Inspirations</h2>
        <p>These cards mark the conceptual anchors used by the prototype.</p>
      </div>

      <div className="related-grid">
        {relatedWorkItems.map((item) => (
          <article key={item.name} className="related-card">
            <h3>{item.name}</h3>
            <p>{item.summary}</p>
            <div>
              <strong>Inspiration</strong>
              <span>{item.inspiration}</span>
            </div>
            <a href={item.link}>Link placeholder</a>
          </article>
        ))}
      </div>
    </section>
  );
}
