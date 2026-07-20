#!/usr/bin/env python3
"""Structural validator driver for World2Skills skills.

Drives two off-the-shelf checks per skill directory:
  1. SKILL.md  -> skill-creator's quick_validate.py (standard Agent Skills)
  2. skill.yaml -> jsonschema library against skill.schema.json

This is a driver only. It intentionally does NOT check cross-file or
reference integrity (graph closure, primitive_map coverage, name agreement
across files) -- those are manual checks per the spec.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "skill.schema.json"
SKILLS_DIR = ROOT / "skills"

# External skill-creator validator (authoritative Agent Skills checker).
QUICK_VALIDATE = Path(
    "/home/yifanyang/.claude/plugins/marketplaces/claude-plugins-official"
    "/plugins/skill-creator/skills/skill-creator/scripts/quick_validate.py"
)


def validate_skill_md(skill_dir: Path) -> list[str]:
    if not QUICK_VALIDATE.exists():
        return [f"quick_validate.py not found at {QUICK_VALIDATE}"]
    proc = subprocess.run(
        [sys.executable, str(QUICK_VALIDATE), str(skill_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return [f"SKILL.md: {proc.stdout.strip() or proc.stderr.strip()}"]
    return []


def validate_skill_yaml(skill_dir: Path, validator: Draft202012Validator) -> list[str]:
    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        return ["skill.yaml not found"]
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"skill.yaml: invalid YAML: {exc}"]
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    return [f"skill.yaml: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not skill_dirs:
        print("no skills found")
        return 1

    total_errors = 0
    for skill_dir in skill_dirs:
        errors = validate_skill_md(skill_dir) + validate_skill_yaml(skill_dir, validator)
        if errors:
            total_errors += len(errors)
            print(f"[FAIL] {skill_dir.name}")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"[OK]   {skill_dir.name}")

    print(f"\n{len(skill_dirs)} skills checked, {total_errors} error(s).")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
