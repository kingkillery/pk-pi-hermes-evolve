import type { ConstraintResult, SkillStructureReport } from "./types.js";

const PREFIX_LENGTH = 500;

export function buildSkillStructureReport(text: string): SkillStructureReport {
  const prefix = text.slice(0, PREFIX_LENGTH);

  // Frontmatter must start at the very beginning and be closed by another --- delimiter.
  const frontmatterMatch = /^---\r?\n([\s\S]*?)\r?\n---/.exec(prefix);
  const hasFrontmatter = frontmatterMatch !== null;
  const scope = hasFrontmatter ? frontmatterMatch![1] : prefix;

  const nameRegex = /^name:\s*\S+/m;
  const descriptionRegex = /^description:\s*\S+/m;

  const hasName = nameRegex.test(scope);
  const hasDescription = descriptionRegex.test(scope);

  // nameInFirst500 / descriptionInFirst500 are true when the keyed value is present in the
  // bounded scope (frontmatter block, or first 500 chars when no frontmatter exists).
  const nameInFirst500 = hasName;
  const descriptionInFirst500 = hasDescription;

  const errors: string[] = [];
  if (!hasFrontmatter) {
    errors.push("missing YAML frontmatter (--- ... ---) at start of SKILL.md");
  }
  if (!nameInFirst500) {
    errors.push("missing `name:` field in first 500 chars of SKILL.md");
  }
  if (!descriptionInFirst500) {
    errors.push("missing `description:` field in first 500 chars of SKILL.md");
  }

  return {
    hasFrontmatter,
    hasName,
    hasDescription,
    nameInFirst500,
    descriptionInFirst500,
    errors,
  };
}

export function checkSkillStructure(candidateFullText: string): ConstraintResult {
  const report = buildSkillStructureReport(candidateFullText);
  const passed = report.nameInFirst500 && report.descriptionInFirst500;
  return {
    name: "skill_structure",
    passed,
    message: passed
      ? "skill_structure OK"
      : "missing name/description in first 500 chars of SKILL.md",
    details: JSON.stringify(report),
  };
}
