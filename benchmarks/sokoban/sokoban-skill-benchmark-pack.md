# Sokoban Skill-Improvement Benchmark Pack

## Goal
Test whether your harness can improve an agent's reusable game-playing skill across repeated attempts, then transfer that improvement to a fresh held-out attempt.

## Recommended game target
Primary target: https://www.sokobanonline.com/help/how-to-play
Playable game family: https://www.sokobanonline.com/
Alternative simple target: https://paskhaver.github.io/sokoban/

Why this game works:
- Binary success condition: solve or fail.
- Small action space.
- Clear, reusable heuristics.
- Reset and undo support on Sokoban Online.

## Game facts to encode into the benchmark
Classic Sokoban rules:
- Move with arrow keys.
- Only one box can be pushed at a time.
- Boxes cannot be pulled.
- The player cannot walk through walls or boxes.
- The puzzle is solved when all boxes are on goals.

Sokoban Online default controls:
- Arrow keys = move
- U = undo
- R = reset
- P or ESC = pause

Paskhaver Sokoban notes:
- Use arrow keys to move.
- The next level loads when all boxes are on checkpoints.
- Reset Level restarts the board.

## Core experiment design
- Attempts 1-4: training attempts.
- Attempt 5: fresh evaluation attempt on an unseen level from the same source and difficulty bucket.
- After each training attempt, the agent may update a persistent skill file.
- The agent may store reusable heuristics, but must not store exact move sequences for the held-out level.
- The benchmark passes if attempt 5 solves the held-out board faster or more reliably than the baseline condition.

## Best proof design
Run two arms:
1. Baseline arm: no skill updates allowed between attempts.
2. Improvement arm: skill updates allowed after attempts 1-4.

Evaluate both arms on the same held-out attempt-5 level set.

## Harness prompt
```text
You are being evaluated on your ability to improve a reusable Sokoban-playing skill across repeated attempts.

Game: Sokoban
Objective: Push all boxes onto goals/checkpoints.
Allowed controls: Only the game's supported controls.
Assume the game may support movement, reset, undo, and pause.

Protocol:
- You will complete 5 total attempts.
- Attempts 1-4 are training attempts on levels from the same game family.
- After each of attempts 1-4, you may update your persistent skill, playbook, or memory.
- Attempt 5 is an evaluation attempt on a fresh unseen level from the same source and difficulty bucket.
- On attempt 5, apply the improved skill, but do not rely on any memorized move sequence for that exact board.

For every attempt:
1. Inspect the board.
2. Write a short plan before acting.
3. Play the level.
4. If the board becomes irrecoverable, you may use undo or reset if the game supports them.
5. Finish with a short postmortem.

After attempts 1-4, update the skill using only transferable heuristics.
Do not save level-specific move sequences except as temporary notes that are discarded before attempt 5.

Metrics to record each attempt:
- solved: true/false
- wall_clock_seconds
- total_actions
- movement_actions
- undo_count
- reset_count
- irrecoverable_state_count
- dead_box_events
- plan_length_tokens
- postmortem_length_tokens

Postmortem questions:
- What mistake caused the most wasted actions?
- Which boxes became dangerous and why?
- Which heuristic would have prevented that?
- What concise reusable rule should be added to the skill?

Skill-update rules:
- Prefer compact general rules.
- Prefer state-based heuristics over board-specific scripts.
- Keep only rules that are likely to transfer to unseen levels.
- Remove rules that did not help.

Evaluation goal:
On attempt 5, solve the fresh level faster and with fewer resets, undos, and dead-end states than earlier attempts and than the baseline arm.
```

## Suggested skill schema
```yaml
name: sokoban_solver
version: 1
principles:
  - Never push a box into a corner unless that corner is a goal.
  - Before any push, check whether the box will still have a path to a goal.
  - Prefer moves that preserve walking space around boxes.
  - Clear boxes from walls early unless the wall path ends at a goal.
  - Use undo immediately after a push that reduces future mobility.
  - Reset when multiple boxes become frozen and recovery cost exceeds restart cost.
pre_push_checklist:
  - Is the destination square legal?
  - Does the push trap the box against a wall or corner?
  - Will the player still be able to reach the useful side of the box?
  - Does this push reduce the number of reachable goals?
failure_signals:
  - Box frozen off-goal
  - Two boxes mutually blocking a corridor
  - Goal lane blocked by wrong box order
update_policy:
  - Add only rules supported by at least one observed failure.
  - Merge duplicate rules.
  - Keep the total rule set short enough to apply during play.
```

## Level split recommendation
Use a small fixed set.

Example:
- Training attempt 1: beginner level A
- Training attempt 2: beginner level B
- Training attempt 3: beginner/intermediate level C
- Training attempt 4: intermediate level D
- Evaluation attempt 5: unseen beginner/intermediate level E

Better evaluation:
- Repeat the full 5-attempt protocol over 5-20 different held-out attempt-5 levels.
- Report averages and variance, not one anecdotal run.

## Success criteria
Primary:
- Attempt 5 solve rate on unseen levels exceeds baseline.
- Attempt 5 median time is lower than baseline.

Secondary:
- Fewer resets.
- Fewer undos.
- Fewer dead-box events.
- Lower total actions to solve.

## Logging schema
```json
{
  "run_id": "string",
  "arm": "baseline|improvement",
  "attempt": 1,
  "level_id": "string",
  "held_out": false,
  "solved": true,
  "wall_clock_seconds": 0,
  "total_actions": 0,
  "movement_actions": 0,
  "undo_count": 0,
  "reset_count": 0,
  "irrecoverable_state_count": 0,
  "dead_box_events": 0,
  "notes": "string"
}
```

## Analysis plan
Compute these comparisons:
- Attempt 1 vs attempt 5 within the improvement arm.
- Baseline attempt 5 vs improvement attempt 5.
- Held-out solve rate delta.
- Held-out median time delta.
- Held-out median total actions delta.

A convincing result is not just one faster attempt 5. The strongest result is a repeated held-out advantage for the improvement arm.

## Practical warning
One held-out level can be noisy. Use several held-out attempt-5 boards if you want strong evidence that the framework really improves a transferable skill.
