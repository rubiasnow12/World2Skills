interface CodeBlockProps {
  code: string;
  label?: string;
}

export function CodeBlock({ code, label }: CodeBlockProps) {
  return (
    <figure className="code-block">
      {label ? <figcaption>{label}</figcaption> : null}
      <pre>
        <code>{code}</code>
      </pre>
    </figure>
  );
}
