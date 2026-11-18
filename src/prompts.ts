export interface OnlinePromptVars {
  direction_guidance?: string;
  node_dir: string;
  history_dir: string;
  baseline_dir: string;
  eval_program: string;
  problem_file: string;
  parent_dir: string;
}

/**
 * Listing 1: Prompt used for online exploration (Appendix B.1).
 */
export function buildOnlineExplorationPrompt(vars: OnlinePromptVars): string {
  return `You must read every historical proposal before proposing or implementing a new solution.

${vars.direction_guidance ?? ""}

Variables ('node_dir'="${vars.node_dir}", 'parent_dir'="${vars.parent_dir}", 'history_dir'="${vars.history_dir}", 'baseline_dir'="${vars.baseline_dir}", 'eval_program'="${vars.eval_program}", 'problem_file'="${vars.problem_file}") are filled in by the calling system. '${vars.node_dir}' is your own attempt directory — exclude it when scanning sibling 'attempt_*/' dirs.

## 1. Read the complete history first
Before proposing anything, read every 'proposal.md' under sibling 'attempt_*/' dirs, '${vars.history_dir}', and '${vars.baseline_dir}' in full — not a sample, not just recent cycles or the current branch. For each, read its matching 'eval/score.json' (and 'error.txt' if it failed). Trust the measured result over what the proposal claims about itself.

## 2. Learn from both successes and failures
For every past attempt, note the mechanism and how it did. For failures, figure out *why*: a flawed core idea, or a good idea let down by a bug, bad parameters, or an implementation slip? Don't repeat the former. The latter is worth retrying — but only once you've actually located the bug in the code (not just guessed from the proposal), and only with a specific fix in hand.

## 3. Don't converge into a local optimum
Look at the shape of what's been tried. If most attempts cluster around small variations of one mechanism with flattening returns, that's a local optimum — resist proposing another small tweak there. Deliberately favor a structurally different mechanism or an untried combination over a safer marginal refinement. Exploration diversity matters as much as the next incremental gain.

## 4. Propose and implement
The new idea must be a genuinely new mechanism, a new combination of previously-successful pieces, or a targeted fix to a specific bug found in step 2 — never a repeat or rename of something already tried. Implement it in '${vars.node_dir}/${vars.eval_program}'. Don't claim it compiles, is correct, or beats SOTA until it's actually evaluated.

## Files
Write only '${vars.node_dir}/proposal.md' (mechanism, evidence from history, why it's not a repeat, expected benefit/risk) and '${vars.node_dir}/${vars.eval_program}'. Everything else is read-only.

## Note:
Never execute pkill, kill, killall, or terminate unrelated processes.`;
}

export interface PolicyImprovementPromptVars {
  method_file: string;
  history_dir: string;
  trace_pool: string;
  current_sweep_summary: string;
}

/**
 * Listing 2: Prompt used for replay-based improvement of the exploration policy (Appendix B.2).
 */
export function buildReplayPolicyImprovementPrompt(vars: PolicyImprovementPromptVars): string {
  return `You are improving one **prefix-only exploration policy**. Edit only '${vars.method_file}' and implement 'OptimalPolicy' implementing 'ExplorationPolicy' ('select_batch(question, closed)' and 'plan_grid(context)').
Do not solve the scientific task and do not edit any other program.

## Objective: quality, work, and parallelism
The environment is a frozen, irregular branch attempt grid. A policy opens a root or refines the next cell of an already-open branch. Each revealed cell costs one probe. The policy sees only the cells it has revealed so far; unrevealed scores are unknown.

The evaluator sweeps your single 'beta' knob and ranks the resulting curve by:
  pareto.reward = pareto.auc - lambda * parallel_penalty
and Equation (1):
  V_i^m = max_{v} s_v - beta_1 * N_i^m + beta_2 * (N_i^m / max(1, k_i^{m,*}))

Choose only promising probes, but batch independent promising probes up to 'question.max_parallelism' whenever possible. A local implementation failure does not by itself prove that its parent direction is poor. Weigh recovery value against new roots and ordinary refinements while keeping batches parallel.

## Current Replay Evaluation Summary
${vars.current_sweep_summary}

## Hard constraints
1. Keep 'name = "OptimalPolicy"' and export 'class OptimalPolicy' in '${vars.method_file}' only.
2. **Prefix-only:** decisions may use revealed observations ('question.observed()'), 'question.baseline_score', 'question.legal_actions()', 'question.legal_roots()', 'question.meta(cell_id)', and signal helpers. Never read unrevealed scores, hardcoded cell IDs, or absolute score targets.
3. Every prune, widen, deepen, batch, and stop decision must be explainable from the current prefix.
4. Implement 'plan_grid(context: GridPlanningContext): GridPlan' returning a non-null 'GridPlan' with 'branch_count', 'refine_count', and a factual 'reason' derived from '${vars.trace_pool}/iter*/live_cycle_manifest.json' and '${vars.history_dir}/r*/beta_sweep.json'.`;
}
