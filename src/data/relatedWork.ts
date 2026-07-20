export interface RelatedWorkItem {
  name: string;
  summary: string;
  inspiration: string;
  link: string;
}

export const relatedWorkItems: RelatedWorkItem[] = [
  {
    name: 'SayCan',
    summary: 'Language models select feasible robot skills by combining task intent with affordance estimates.',
    inspiration: 'Skill can act as the interface between language-model reasoning and robot execution.',
    link: '#saycan-placeholder',
  },
  {
    name: 'ProgPrompt',
    summary: 'Prompts language models to compose executable robot plans from available programmatic actions.',
    inspiration: 'Skill can be represented as a callable program API.',
    link: '#progprompt-placeholder',
  },
  {
    name: 'Code as Policies',
    summary: 'Transforms natural-language tasks into policy code that calls perception and control primitives.',
    inspiration: 'Natural-language tasks can become executable policy code.',
    link: '#code-as-policies-placeholder',
  },
  {
    name: 'NSRT / predicators',
    summary: 'Represents neuro-symbolic skills with predicates, samplers, operators, and low-level policies.',
    inspiration: 'Skill can include preconditions, effects, sampler-like parameters, and policy hooks.',
    link: '#nsrt-predicators-placeholder',
  },
  {
    name: 'Voyager',
    summary: 'Builds an expanding skill library that can be reused for future tasks in an open-ended setting.',
    inspiration: 'Skill libraries can grow, consolidate experience, and support reuse.',
    link: '#voyager-placeholder',
  },
];
