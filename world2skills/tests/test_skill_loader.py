from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import shutil

import pytest
import yaml

from world2skills.runtime.skill_loader import load_skill, select_grounding


REPO_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
REQUIRED_HEADINGS = [
    "## When to use",
    "## Procedure",
    "## Reasoning cues",
    "## Failure modes & recovery",
]
EXPECTED_PRIMITIVES = {
    "facilitate-highway-merge": [
        "change-lane-left",
        "maintain-speed",
        "accelerate",
        "decelerate",
    ],
    "follow-keep-distance": [
        "maintain-speed",
        "accelerate",
        "decelerate",
    ],
    "lane-change-overtake": [
        "change-lane-left",
        "change-lane-right",
        "maintain-speed",
        "accelerate",
        "decelerate",
    ],
    "roundabout-navigate": [
        "maintain-speed",
        "accelerate",
        "decelerate",
    ],
    "unprotected-left-turn": [
        "maintain-speed",
        "accelerate",
        "decelerate",
    ],
}


def _copy_skill(
    tmp_path: Path,
    source_id: str = "lane-change-overtake",
    destination_id: str | None = None,
) -> tuple[Path, Path]:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / (destination_id or source_id)
    shutil.copytree(REPO_SKILLS_DIR / source_id, skill_dir)
    return skills_dir, skill_dir


def _read_skill_md(skill_dir: Path) -> tuple[dict, str]:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter_text, body = text.split("---", 2)
    return yaml.safe_load(frontmatter_text), body.lstrip("\n")


def _write_skill_md(skill_dir: Path, frontmatter: dict, body: str) -> None:
    frontmatter_text = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        width=10_000,
    )
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter_text}---\n{body}",
        encoding="utf-8",
    )


@pytest.mark.parametrize("skill_id", sorted(EXPECTED_PRIMITIVES))
def test_loads_all_skills_and_preserves_required_fields(skill_id: str):
    card = load_skill(skill_id)
    source = yaml.safe_load(
        (REPO_SKILLS_DIR / skill_id / "skill.yaml").read_text(encoding="utf-8")
    )

    assert card.name == skill_id
    assert card.description == source["description"]
    for field_name in (
        "schema_version",
        "version",
        "domain",
        "category",
        "tags",
        "entities",
        "parameters",
        "interface",
        "execution",
        "preconditions",
        "effects",
        "success_criteria",
        "failure_criteria",
        "safety_constraints",
        "failure_modes",
        "termination",
        "related_skills",
    ):
        assert getattr(card, field_name) == source[field_name]
    assert [asdict(grounding) for grounding in card.groundings] == source["groundings"]
    assert card.primitives == EXPECTED_PRIMITIVES[skill_id]

    headings = [
        line for line in card.skill_md_body.splitlines() if line.startswith("## ")
    ]
    assert headings == REQUIRED_HEADINGS


def test_selects_grounding_and_rejects_missing_backend():
    card = load_skill("lane-change-overtake")

    grounding = select_grounding(card)

    assert grounding.backend == "highway-env"
    assert grounding.backend_version == ">=1.8"
    assert grounding.environment == "highway-v0"
    assert grounding.primitive_map["accelerate"] == "FASTER"
    with pytest.raises(ValueError, match="no grounding.*carla"):
        select_grounding(card, backend="carla")


@pytest.mark.parametrize("missing_name", ["SKILL.md", "skill.yaml"])
def test_rejects_missing_required_file(tmp_path: Path, missing_name: str):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    (skill_dir / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


@pytest.mark.parametrize(
    "skill_id",
    [
        "",
        ".",
        "..",
        "../lane-change-overtake",
        "lane-change-overtake/../follow-keep-distance",
        "/tmp/lane-change-overtake",
        "Lane-Change-Overtake",
        "lane_change_overtake",
    ],
)
def test_rejects_invalid_skill_id_and_path_traversal(skill_id: str):
    with pytest.raises(ValueError, match="invalid skill id"):
        load_skill(skill_id)


def test_rejects_frontmatter_without_closing_fence(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        "---\n"
        "name: lane-change-overtake\n"
        "description: missing closing fence\n"
        "# Lane Change Overtake\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frontmatter"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_invalid_frontmatter_yaml(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        "---\nname: [\n---\n# Lane Change Overtake\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frontmatter"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_non_mapping_frontmatter(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        "---\n- lane-change-overtake\n---\n# Lane Change Overtake\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frontmatter.*mapping"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_undocumented_frontmatter_key(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    frontmatter, body = _read_skill_md(skill_dir)
    frontmatter["undocumented"] = "value"
    _write_skill_md(skill_dir, frontmatter, body)

    with pytest.raises(ValueError, match="frontmatter fields"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


@pytest.mark.parametrize("missing_field", ["name", "description", "metadata"])
def test_rejects_missing_frontmatter_field(tmp_path: Path, missing_field: str):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    frontmatter, body = _read_skill_md(skill_dir)
    del frontmatter[missing_field]
    _write_skill_md(skill_dir, frontmatter, body)

    with pytest.raises(ValueError, match="frontmatter fields"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


@pytest.mark.parametrize(
    "description_yaml",
    [
        "description: >-\n  first line\n  second line\n",
        "description: first line\n  second line\n",
    ],
)
def test_rejects_multiline_frontmatter_description(
    tmp_path: Path,
    description_yaml: str,
):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    _, body = _read_skill_md(skill_dir)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: lane-change-overtake\n"
        f"{description_yaml}"
        "metadata:\n"
        '  schema_version: "0.1"\n'
        "  representation: skill.yaml\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="description.*single-line"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "skill.yaml",
        {"schema_version": "0.1"},
        {"representation": "skill.yaml"},
        {
            "schema_version": "0.1",
            "representation": "skill.yaml",
            "undocumented": "value",
        },
        {"schema_version": 0.1, "representation": "skill.yaml"},
        {"schema_version": "0.1", "representation": "other.yaml"},
    ],
)
def test_rejects_invalid_frontmatter_metadata(tmp_path: Path, metadata):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    frontmatter, body = _read_skill_md(skill_dir)
    frontmatter["metadata"] = metadata
    _write_skill_md(skill_dir, frontmatter, body)

    with pytest.raises(ValueError, match="frontmatter metadata"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_frontmatter_schema_version_mismatch(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    frontmatter, body = _read_skill_md(skill_dir)
    frontmatter["metadata"]["schema_version"] = "9.9"
    _write_skill_md(skill_dir, frontmatter, body)

    with pytest.raises(ValueError, match="schema_version mismatch"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_duplicate_frontmatter_metadata_key(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    _, body = _read_skill_md(skill_dir)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: lane-change-overtake\n"
        "description: A single-line description.\n"
        "metadata:\n"
        '  schema_version: "0.1"\n'
        '  schema_version: "0.1"\n'
        "  representation: skill.yaml\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frontmatter metadata"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_accepts_optional_final_references_heading(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        md_path.read_text(encoding="utf-8")
        + "\n## References\n"
        + "- HighwayEnv documentation\n",
        encoding="utf-8",
    )

    card = load_skill("lane-change-overtake", skills_dir=skills_dir)

    assert card.skill_md_body.rstrip().endswith(
        "## References\n- HighwayEnv documentation"
    )


@pytest.mark.parametrize(
    ("opening_fence", "closing_fence"),
    [
        ("```markdown", "```"),
        ("~~~~text", "~~~~"),
    ],
)
def test_ignores_level_two_headings_inside_fenced_code_blocks(
    tmp_path: Path,
    opening_fence: str,
    closing_fence: str,
):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    text = md_path.read_text(encoding="utf-8").replace(
        "## Reasoning cues",
        f"{opening_fence}\n## Not a real section\n{closing_fence}\n\n"
        "## Reasoning cues",
        1,
    )
    md_path.write_text(text, encoding="utf-8")

    card = load_skill("lane-change-overtake", skills_dir=skills_dir)

    assert "## Not a real section" in card.skill_md_body


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        ("## When to use", "## When to use this skill"),
        ("## Failure modes & recovery", "## Failure modes"),
        ("## Reasoning cues\n", ""),
        ("## Procedure", "## Procedure\n\n## Extra section"),
        ("## Procedure", "## References\n\n## Procedure"),
    ],
)
def test_rejects_missing_changed_or_extra_or_nonfinal_level_two_heading(
    tmp_path: Path,
    source: str,
    replacement: str,
):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        md_path.read_text(encoding="utf-8").replace(source, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required headings"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_reordered_required_headings(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    text = md_path.read_text(encoding="utf-8")
    text = text.replace("## When to use", "## TEMP", 1)
    text = text.replace("## Procedure", "## When to use", 1)
    text = text.replace("## TEMP", "## Procedure", 1)
    md_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="required headings"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_folder_name_mismatch(tmp_path: Path):
    skills_dir, _ = _copy_skill(
        tmp_path,
        destination_id="renamed-lane-change-overtake",
    )

    with pytest.raises(ValueError, match="name mismatch"):
        load_skill("renamed-lane-change-overtake", skills_dir=skills_dir)


def test_rejects_frontmatter_name_mismatch(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(
        md_path.read_text(encoding="utf-8").replace(
            "name: lane-change-overtake",
            "name: follow-keep-distance",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="name mismatch"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_yaml_name_mismatch(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    yaml_path = skill_dir / "skill.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["name"] = "follow-keep-distance"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="name mismatch"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_rejects_schema_invalid_skill_yaml(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    yaml_path = skill_dir / "skill.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    del data["effects"]
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"skill\.yaml validation failed"):
        load_skill("lane-change-overtake", skills_dir=skills_dir)


def test_breaks_yaml_aliases_between_mutable_skill_fields(tmp_path: Path):
    skills_dir, skill_dir = _copy_skill(tmp_path)
    yaml_path = skill_dir / "skill.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    shared_labels = ["shared-label"]
    data["tags"] = shared_labels
    data["related_skills"] = shared_labels

    shared_criteria = ["shared criterion"]
    data["preconditions"] = shared_criteria
    data["effects"] = shared_criteria

    shared_features = data["interface"]["observations"][0]["features"]
    data["groundings"][0]["observation"]["features"] = shared_features

    second_grounding = copy.deepcopy(data["groundings"][0])
    second_grounding["backend"] = "alternate-highway-env"
    second_grounding["observation"] = data["groundings"][0]["observation"]
    second_grounding["action"] = data["groundings"][0]["action"]
    second_grounding["primitive_map"] = data["groundings"][0]["primitive_map"]
    data["groundings"].append(second_grounding)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    card = load_skill("lane-change-overtake", skills_dir=skills_dir)
    first_grounding, second_grounding = card.groundings

    card.tags.append("tags-only")
    card.preconditions.append("preconditions-only")
    card.interface["observations"][0]["features"].append("interface-only")
    first_grounding.observation["features"].append("observation-only")
    first_grounding.action["runtime-only"] = True
    first_grounding.primitive_map["runtime-only"] = "IDLE"

    assert card.related_skills == ["shared-label"]
    assert card.effects == ["shared criterion"]
    assert "interface-only" not in first_grounding.observation["features"]
    assert "observation-only" not in second_grounding.observation["features"]
    assert "runtime-only" not in second_grounding.action
    assert "runtime-only" not in second_grounding.primitive_map
