"""Load and validate World2Skills skill cards."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
import yaml

from .types import Grounding, SkillCard


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "skill.schema.json"
REQUIRED_HEADINGS = (
    "## When to use",
    "## Procedure",
    "## Reasoning cues",
    "## Failure modes & recovery",
)
_SKILL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LEVEL_TWO_HEADING = re.compile(r"^##(?:\s|$)")


def _validate_skill_id(skill_id: str) -> None:
    if not isinstance(skill_id, str) or not _SKILL_ID_PATTERN.fullmatch(skill_id):
        raise ValueError(f"invalid skill id: {skill_id!r}")


def _read_required_file(skill_dir: Path, filename: str) -> str:
    path = skill_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{filename} not found for '{skill_dir.name}' at {path}")
    return path.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("SKILL.md frontmatter must start with an exact '---' fence")

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "---"
        ),
        None,
    )
    if closing_index is None:
        raise ValueError("SKILL.md frontmatter is missing its closing '---' fence")

    try:
        frontmatter = yaml.safe_load("".join(lines[1:closing_index]))
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md frontmatter is invalid YAML: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    for field_name in ("name", "description"):
        value = frontmatter.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"SKILL.md frontmatter field '{field_name}' must be a non-empty string"
            )

    body = "".join(lines[closing_index + 1 :]).lstrip("\r\n")
    return frontmatter, body


def _validate_headings(body: str) -> None:
    headings = [
        line
        for line in body.splitlines()
        if _LEVEL_TWO_HEADING.match(line)
    ]
    if headings != list(REQUIRED_HEADINGS):
        raise ValueError(
            "SKILL.md required headings must be exactly: "
            + ", ".join(REQUIRED_HEADINGS)
        )


@lru_cache(maxsize=1)
def _skill_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _load_skill_yaml(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"skill.yaml is invalid YAML: {exc}") from exc

    errors = sorted(
        _skill_validator().iter_errors(data),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"skill.yaml validation failed at {location}: {error.message}"
        )
    return data


def _ordered_primitives(interface: dict[str, Any]) -> list[str]:
    primitives: list[str] = []
    for action in interface["actions"]:
        for primitive in action["primitives"]:
            if primitive not in primitives:
                primitives.append(primitive)
    return primitives


def load_skill(skill_id: str, skills_dir: Path = SKILLS_DIR) -> SkillCard:
    """Load one validated skill from ``skills_dir/<skill_id>``."""

    _validate_skill_id(skill_id)
    skill_dir = Path(skills_dir) / skill_id
    md_text = _read_required_file(skill_dir, "SKILL.md")
    yaml_text = _read_required_file(skill_dir, "skill.yaml")

    frontmatter, body = _split_frontmatter(md_text)
    _validate_headings(body)
    data = _load_skill_yaml(yaml_text)

    names = {
        "folder": skill_id,
        "frontmatter": frontmatter["name"],
        "skill.yaml": data["name"],
    }
    if len(set(names.values())) != 1:
        raise ValueError(
            "skill name mismatch: "
            + ", ".join(f"{source}={name!r}" for source, name in names.items())
        )

    groundings = [
        Grounding(
            backend=grounding["backend"],
            backend_version=grounding["backend_version"],
            environment=grounding["environment"],
            observation=grounding["observation"],
            action=grounding["action"],
            primitive_map=grounding["primitive_map"],
        )
        for grounding in data["groundings"]
    ]

    return SkillCard(
        name=data["name"],
        description=data["description"],
        skill_md_body=body,
        parameters=data.get("parameters", {}),
        interface=data["interface"],
        execution=data["execution"],
        preconditions=data["preconditions"],
        effects=data["effects"],
        success_criteria=data["success_criteria"],
        failure_criteria=data["failure_criteria"],
        safety_constraints=data["safety_constraints"],
        failure_modes=data["failure_modes"],
        termination=data["termination"],
        groundings=groundings,
        primitives=_ordered_primitives(data["interface"]),
    )


def select_grounding(
    card: SkillCard,
    backend: str = "highway-env",
) -> Grounding:
    """Return the grounding for ``backend`` or raise a clear contract error."""

    for grounding in card.groundings:
        if grounding.backend == backend:
            return grounding
    raise ValueError(
        f"skill '{card.name}' has no grounding for backend '{backend}'"
    )
