const nodes = [
  { label: 'Environment Experience', x: 72, y: 90, tone: 'blue' },
  { label: 'Skill Abstraction', x: 278, y: 90, tone: 'green' },
  { label: 'Data / Rule Generation', x: 484, y: 90, tone: 'amber' },
  { label: 'Policy Learning', x: 690, y: 90, tone: 'coral' },
  { label: 'New Task Generalization', x: 896, y: 90, tone: 'purple' },
];

export function FlowDiagram() {
  return (
    <svg className="flow-diagram" viewBox="0 0 1060 220" role="img" aria-label="World2Skills workflow">
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#1f2937" />
        </marker>
      </defs>
      <path
        className="flow-backbone"
        d="M170 90 C225 35, 260 35, 315 90 S430 145, 485 90 S600 35, 655 90 S770 145, 825 90"
        fill="none"
        markerEnd="url(#arrowhead)"
      />
      {nodes.map((node, index) => (
        <g key={node.label} className={`flow-node flow-node-${node.tone}`} style={{ animationDelay: `${index * 120}ms` }}>
          <rect x={node.x} y={node.y - 42} width="154" height="84" rx="8" />
          <text x={node.x + 77} y={node.y - 5} textAnchor="middle">
            {node.label.split(' ').slice(0, 2).join(' ')}
          </text>
          <text x={node.x + 77} y={node.y + 17} textAnchor="middle">
            {node.label.split(' ').slice(2).join(' ')}
          </text>
        </g>
      ))}
      <g className="flow-loop">
        <path d="M930 145 C780 205, 265 205, 120 146" fill="none" />
        <text x="525" y="198" textAnchor="middle">
          reusable skill knowledge feeds future environments
        </text>
      </g>
    </svg>
  );
}
