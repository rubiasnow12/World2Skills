"""Load and validate World2Skills skill cards."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
import yaml
from yaml.nodes import MappingNode

from .types import Grounding, SkillCard


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "skill.schema.json"
REQUIRED_HEADINGS = (
    "## When to use",
    "## Procedure",
    "## Reasoning cues",
    "## Failure modes & recovery",
)
OPTIONAL_FINAL_HEADING = "## References"
_SKILL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LEVEL_TWO_HEADING = re.compile(r"^##(?:\s|$)")
_FENCE_LINE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")
_FRONTMATTER_FIELDS = {"name", "description", "metadata"}
_METADATA_FIELDS = {"schema_version", "representation"}


def _validate_skill_id(skill_id: str) -> None:
    if not isinstance(skill_id, str) or not _SKILL_ID_PATTERN.fullmatch(skill_id):
        raise ValueError(f"invalid skill id: {skill_id!r}")


def _read_required_file(skill_dir: Path, filename: str) -> str:
    path = skill_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{filename} not found for '{skill_dir.name}' at {path}"
        )
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

    frontmatter_text = "".join(lines[1:closing_index])
    try:
        frontmatter_node = yaml.compose(frontmatter_text)
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md frontmatter is invalid YAML: {exc}") from exc
    if not isinstance(frontmatter_node, MappingNode) or not isinstance(
        frontmatter,
        dict,
    ):
        raise ValueError("SKILL.md frontmatter must be a mapping")

    field_nodes = {
        key_node.value: value_node for key_node, value_node in frontmatter_node.value
    }
    field_names = [key_node.value for key_node, _ in frontmatter_node.value]
    if (
        len(field_names) != len(_FRONTMATTER_FIELDS)
        or set(field_names) != _FRONTMATTER_FIELDS
    ):
        raise ValueError(
            "SKILL.md frontmatter fields must be exactly: "
            + ", ".join(sorted(_FRONTMATTER_FIELDS))
        )

    name = frontmatter["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md frontmatter field 'name' must be a non-empty string")

    description = frontmatter["description"]
    description_node = field_nodes["description"]
    if (
        not isinstance(description, str)
        or not description.strip()
        or "\n" in description
        or "\r" in description
        or description_node.start_mark.line != description_node.end_mark.line
    ):
        raise ValueError(
            "SKILL.md frontmatter description must be a non-empty single-line string"
        )

    metadata = frontmatter["metadata"]
    metadata_node = field_nodes["metadata"]
    if not isinstance(metadata, dict) or not isinstance(metadata_node, MappingNode):
        raise ValueError(
            "SKILL.md frontmatter metadata must be a mapping with exactly "
            "schema_version and representation"
        )
    metadata_field_names = [key_node.value for key_node, _ in metadata_node.value]
    if (
        len(metadata_field_names) != len(_METADATA_FIELDS)
        or set(metadata_field_names) != _METADATA_FIELDS
    ):
        raise ValueError(
            "SKILL.md frontmatter metadata must be a mapping with exactly "
            "schema_version and representation"
        )
    if not isinstance(metadata["schema_version"], str):
        raise ValueError(
            "SKILL.md frontmatter metadata schema_version must be a string"
        )
    if metadata["representation"] != "skill.yaml":
        raise ValueError(
            "SKILL.md frontmatter metadata representation must equal 'skill.yaml'"
        )

    body = "".join(lines[closing_index + 1 :]).lstrip("\r\n")
    return frontmatter, body


def _headings_outside_fences(body: str) -> list[str]:
    headings: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    for line in body.splitlines():
        fence_match = _FENCE_LINE.match(line)
        if fence_match:
            fence = fence_match.group("fence")
            rest = fence_match.group("rest")
            if fence_character is None:
                fence_character = fence[0]
                fence_length = len(fence)
            elif (
                fence[0] == fence_character
                and len(fence) >= fence_length
                and not rest.strip()
            ):
                fence_character = None
                fence_length = 0
            continue
        if fence_character is None and _LEVEL_TWO_HEADING.match(line):
            headings.append(line)

    return headings


def _validate_headings(body: str) -> None:
    headings = _headings_outside_fences(body)
    allowed_headings = (
        list(REQUIRED_HEADINGS),
        [*REQUIRED_HEADINGS, OPTIONAL_FINAL_HEADING],
    )
    if headings not in allowed_headings:
        raise ValueError(
            "SKILL.md required headings must be exactly "
            f"{list(REQUIRED_HEADINGS)!r}, optionally followed by "
            f"{OPTIONAL_FINAL_HEADING!r}"
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
        raise ValueError(f"skill.yaml validation failed at {location}: {error.message}")
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

    if frontmatter["metadata"]["schema_version"] != data["schema_version"]:
        raise ValueError(
            "SKILL.md frontmatter metadata schema_version mismatch: "
            f"{frontmatter['metadata']['schema_version']!r} != "
            f"{data['schema_version']!r}"
        )

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
            observation=deepcopy(grounding["observation"]),
            action=deepcopy(grounding["action"]),
            primitive_map=deepcopy(grounding["primitive_map"]),
        )
        for grounding in data["groundings"]
    ]

    return SkillCard(
        schema_version=data["schema_version"],
        name=data["name"],
        version=data["version"],
        domain=data["domain"],
        category=data["category"],
        tags=deepcopy(data.get("tags", [])),
        description=data["description"],
        entities=deepcopy(data["entities"]),
        skill_md_body=body,
        parameters=deepcopy(data.get("parameters", {})),
        interface=deepcopy(data["interface"]),
        execution=deepcopy(data["execution"]),
        preconditions=deepcopy(data["preconditions"]),
        effects=deepcopy(data["effects"]),
        success_criteria=deepcopy(data["success_criteria"]),
        failure_criteria=deepcopy(data["failure_criteria"]),
        safety_constraints=deepcopy(data["safety_constraints"]),
        failure_modes=deepcopy(data["failure_modes"]),
        termination=data["termination"],
        related_skills=deepcopy(data.get("related_skills", [])),
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
    raise ValueError(f"skill '{card.name}' has no grounding for backend '{backend}'")
