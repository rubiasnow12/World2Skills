import { FlowDiagram } from './FlowDiagram';

export function Hero() {
  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">World2Skills Demo</p>
        <h1>World2Skills: Skill-Centric Embodied Intelligence</h1>
        <p>
          A demo for Skill construction, representation, data generation, rule generation, and execution
          simulation.
        </p>
      </div>
      <FlowDiagram />
    </header>
  );
}
