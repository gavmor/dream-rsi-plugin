export type FailClass =
  | "ok"
  | "compile_other"
  | "output_mismatch"
  | "resource_limit"
  | "shape_layout_error"
  | "runtime_error"
  | "env_dependency"
  | "timeout";

export interface Observation {
  cell_id: string; // "b{branch}_a{attempt}" or "root"
  parent_id: string;
  branch: number;
  attempt: number;
  score: number;
  evaluated: boolean;
  valid: boolean;
  fail_class: FailClass;
  error: string | null;
  delta_vs_baseline: number;
  delta_vs_parent: number;
  n_valid: number;
  n_total: number;
  proposal_summary?: string;
  artifact_path?: string;
}

export interface CellMeta {
  cell_id: string;
  parent_id: string;
  branch: number;
  attempt: number;
  seq: number;
  tags: string[];
}

export interface DiscoveryNode {
  id: string;
  parent_id: string | null;
  branch: number;
  attempt: number;
  created_order: number;
  workspace_dir: string;
  observation: Observation;
  meta: CellMeta;
  children_ids: string[];
}

/**
 * A recorded discovery tree T_i collected during Online Rollout t=i.
 */
export interface DiscoveryTree {
  tree_id: string;
  iteration: number;
  root_id: string;
  baseline_score: number;
  max_parallelism: number; // W
  planned_branch_count: number;
  planned_refine_count: number;
  baked_beta: number;
  nodes: Record<string, DiscoveryNode>;
  root_children_ordered: string[]; // Ordered by creation time for deterministic root expansion
}

export interface ReplayRoundTrace {
  round: number;
  selected_batch: string[];
  revealed_observations: Observation[];
  best_score_so_far: number;
  total_probes: number;
}

export interface ReplayScore {
  tree_id: string;
  beta: number;
  best_quality: number;       // max_{v in T_i^{m,*}} s_v
  total_probes: number;       // N_i^m = |T_i^{m,*}| - 1
  decision_rounds: number;    // k_i^{m,*}
  effective_seq_rounds: number;
  parallelism_ratio: number;  // N_i^m / max(1, k_i^{m,*})
  equation1_score: number;    // V_i^m = max s_v - beta_1 * N_i^m + beta_2 * (N_i^m / max(1, k_i^{m,*}))
  curve: Array<{ probes: number; round: number; best_score: number }>;
  traces: ReplayRoundTrace[];
}

export interface BetaSweepSummary {
  policy_version: string;
  betas: number[];
  per_beta: Record<
    number,
    {
      mean_eq1_score: number;
      mean_best_quality: number;
      mean_probes: number;
      parallel_penalty: number;
    }
  >;
  pareto: {
    auc: number;
    parallel_penalty: number;
    reward: number; // pareto.auc - lambda * parallel_penalty
  };
  mean_v_score: number; // V^m = (1/t) * sum_{i=1}^t V_i^m
}

export interface LiveCycleManifest {
  iteration: number;
  tree_id: string;
  policy_version: string;
  baked_beta: number;
  planned_grid: GridPlan;
  opened_width: number;
  max_depth_reached: number;
  probe_work: number;
  decision_rounds: number;
  baseline_score: number;
  final_best_score: number;
  timestamp: string;
}

export interface GridPlan {
  branch_count: number; // W_grid
  refine_count: number; // R_grid
  reason: string;
}

export interface GridPlanningContext {
  history: LiveCycleManifest[];
  hard_max_branch_count: number;
  hard_max_refine_count: number;
  fallback_branch_count: number;
  fallback_refine_count: number;
  max_parallelism: number;
  trace_branch_count?: number;
  trace_refine_count?: number;
}

/**
 * Shared prefix-only decision environment interface for both Offline Replay and Online Rollouts.
 */
export interface ReplayQuestion {
  readonly baseline_score: number;
  readonly max_parallelism: number;
  reset(): void;
  observed(): Record<string, Observation>;
  legal_actions(): string[];
  legal_roots(): string[];
  opened_branches(): number[];
  meta(cell_id: string): CellMeta;
  probe_batch(cells: string[], on_reveal?: () => void): Observation[];
}

export interface ExplorationPolicy {
  readonly name: string;
  readonly beta: number;
  plan_grid(context: GridPlanningContext): GridPlan;
  select_batch(question: ReplayQuestion, closed: Set<number>): string[];
}
