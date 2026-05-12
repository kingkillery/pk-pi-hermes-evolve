---
name: smoke-skill
description: A minimal skill artifact used as the input target for the Hermes Phase 1 runtime smoke pipeline. It exercises the frontmatter, placeholder, heading, size, and skill_structure constraints without depending on a real domain.
---

# Smoke Skill

This skill is the runtime smoke fixture. It is intentionally small and deterministic so
verifier lanes can assert against stable byte counts and constraint outcomes. It is not
meant to be useful in production; do not import it into agent runtimes.

## When to use

Use this skill when you need a synthetic target artifact for the self-evolution engine
to chew on. The body has just enough structure to trip every constraint that depends on
real markdown shape (top heading, frontmatter, placeholder, growth limit).

## How it works

1. The engine reads this file as the target artifact.
2. The mock LLM driver returns canned candidate bodies that mutate the body below
   while preserving the `{{ITERATION_NAME}}` placeholder and the `# Smoke Skill` heading.
3. The judge mock returns deterministic scores so the iterative loop can promote a winner.

## Operating notes

- Always keep the `{{ITERATION_NAME}}` placeholder so the placeholder constraint passes.
- Keep the top-level `# Smoke Skill` heading so the heading constraint passes.
- Never expand the body past 1.2x the original size (the growth limit will reject it).
- Current iteration label: {{ITERATION_NAME}}
