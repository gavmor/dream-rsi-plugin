## Implement Dream-RSI as an Oh My Pi harness plugin

Implementing **DREAM-RSI** (*Recursive Self-Improvement through Evolving Worlds*) as an **Oh My Pi (`omp`)** harness plugin turns `omp` into a meta-exploration orchestrator. It treats completed online discovery trees ($\mathcal{T}_t$) as zero-execution-cost offline replay simulators, allowing the agent to synthesize, evaluate, and select improved executable TypeScript exploration policies ($\pi_t^m$) across a $\beta$-grid sweep before deploying $\pi_{t+1}$ online.

### Plugin Package Layout

Place this plugin in `.omp/extensions/dream-rsi/` (project-local) or install it as a local package via `omp plugin install ./omp-plugin-dream-rsi`.

```text
omp-plugin-dream-rsi/
├── package.json
└── src/
    ├── types.ts         # DiscoveryTree, prefix-only ReplayWorld, Policy & GridPlan contracts
    ├── simulator.ts     # Replay simulator, Eq. (1) objective, & Pareto beta-sweep evaluator
    ├── policy.ts        # Prefix-only signal helpers & baseline OptimalPolicy (pi_1)
    ├── prompts.ts       # Listing 1 (Online Exploration) & Listing 2 (Policy Improvement)
    └── index.ts         # ExtensionAPI registration: tools, /dream-rsi command, & RSI loop

```

---

### 1. `package.json`

Defines the `omp` (and legacy `pi`) extension manifest so `omp` loads the TypeScript entrypoint directly without a build step.

```json
{
  "name": "omp-plugin-dream-rsi",
  "version": "0.1.0",
  "description": "DREAM-RSI recursive self-improvement meta-exploration harness plugin for Oh My Pi",
  "type": "module",
  "omp": {
    "extensions": ["./src/index.ts"]
  },
  "pi": {
    "extensions": ["./src/index.ts"]
  },
  "peerDependencies": {
    "@oh-my-pi/pi-coding-agent": ">=18.0.0"
  },
  "peerDependenciesMeta": {
    "@oh-my-pi/pi-coding-agent": {
      "optional": true
    }
  }
}

```

---

### 2. `src/types.ts`

Encodes the mathematical formulation from Section 3 ($A(\mathcal{T}; W)$, prefix-observable `Observation`, and $V_i^m$) alongside the prefix-only `ReplayQuestion` and `GridPlan` interfaces specified in Appendix B.2.

```typescript
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

```

---

### 3. `src/simulator.ts`

Implements the deterministic replay world transition $\mathcal{T}_i^{m,k+1} = \mathcal{T}_i^{m,k} \cup \bigcup_{v \in C_i^{m,k}} \text{Child}(v; \mathcal{T}_i, \mathcal{T}_i^{m,k})$, Equation (1):


$$V_i^m = \max_{v \in \mathcal{T}_i^{m,*}} s_v - \beta_1 N_i^m + \beta_2 \frac{N_i^m}{\max\{1, k_i^{m,*}\}}$$


and the $\beta$-sweep Pareto evaluator (`pareto.reward = pareto.auc - lambda * parallel_penalty`) from Appendix B.2.

```typescript
import type {
  BetaSweepSummary,
  CellMeta,
  DiscoveryTree,
  ExplorationPolicy,
  Observation,
  ReplayQuestion,
  ReplayRoundTrace,
  ReplayScore,
} from "./types.ts";

export class HistoricalReplayWorld implements ReplayQuestion {
  private tree: DiscoveryTree;
  private revealedIds: Set<string> = new Set();
  private openedRootIds: Set<string> = new Set();

  constructor(tree: DiscoveryTree) {
    this.tree = tree;
    this.reset();
  }

  get baseline_score(): number {
    return this.tree.baseline_score;
  }

  get max_parallelism(): number {
    return this.tree.max_parallelism;
  }

  reset(): void {
    this.revealedIds.clear();
    this.openedRootIds.clear();
    this.revealedIds.add(this.tree.root_id);
  }

  /**
   * Prefix-only view: returns ONLY observations for nodes revealed so far.
   */
  observed(): Record<string, Observation> {
    const out: Record<string, Observation> = {};
    for (const id of this.revealedIds) {
      if (id === this.tree.root_id) continue;
      const node = this.tree.nodes[id];
      if (node) {
        out[id] = structuredClone(node.observation);
      }
    }
    return out;
  }

  /**
   * Unopened root continuation slots (earliest-created children of root outside T_i^{m,k}).
   */
  legal_roots(): string[] {
    const unopened: string[] = [];
    for (const rootChildId of this.tree.root_children_ordered) {
      if (!this.openedRootIds.has(rootChildId)) {
        unopened.push(rootChildId);
      }
    }
    return unopened;
  }

  /**
   * Eligible actions A(T_i^{m,k}) = unopened root branches U unrevealed unique children of current leaves.
   */
  legal_actions(): string[] {
    const actions: string[] = [...this.legal_roots()];
    for (const id of this.revealedIds) {
      if (id === this.tree.root_id) continue;
      const node = this.tree.nodes[id];
      if (!node) continue;
      // A revealed non-root node is a leaf of T_i^{m,k} if none of its children are revealed yet
      const hasRevealedChild = node.children_ids.some((cid) => this.revealedIds.has(cid));
      if (!hasRevealedChild && node.children_ids.length > 0) {
        // Child(v; T_i, T_i^{m,k}) is v's unique recorded child
        actions.push(node.children_ids[0]);
      }
    }
    return actions;
  }

  opened_branches(): number[] {
    const branches = new Set<number>();
    for (const id of this.revealedIds) {
      if (id === this.tree.root_id) continue;
      const node = this.tree.nodes[id];
      if (node) branches.add(node.branch);
    }
    return Array.from(branches).sort((a, b) => a - b);
  }

  meta(cell_id: string): CellMeta {
    const node = this.tree.nodes[cell_id];
    if (!node) {
      throw new Error(`Unknown cell_id in replay world: ${cell_id}`);
    }
    return structuredClone(node.meta);
  }

  probe_batch(cells: string[], on_reveal?: () => void): Observation[] {
    const uniqueCells = Array.from(new Set(cells));
    if (uniqueCells.length > this.max_parallelism) {
      throw new Error(
        `Batch size ${uniqueCells.length} exceeds max_parallelism W=${this.max_parallelism}`,
      );
    }

    const currentlyLegal = new Set(this.legal_actions());
    for (const cellId of uniqueCells) {
      if (!currentlyLegal.has(cellId)) {
        throw new Error(`Illegal or non-frontier cell selected in batch: ${cellId}`);
      }
    }

    const revealed: Observation[] = [];
    for (const cellId of uniqueCells) {
      const node = this.tree.nodes[cellId];
      if (!node) continue;
      this.revealedIds.add(cellId);
      if (node.parent_id === this.tree.root_id) {
        this.openedRootIds.add(cellId);
      }
      revealed.push(structuredClone(node.observation));
      if (on_reveal) on_reveal();
    }
    return revealed;
  }
}

export interface SimulatorHyperparams {
  K2_max_rounds: number;
  beta1_cost_weight: number;
  beta2_parallel_weight: number;
  lambda_parallel_penalty: number;
  beta_grid: number[];
}

export const DEFAULT_SIM_PARAMS: SimulatorHyperparams = {
  K2_max_rounds: 64,
  beta1_cost_weight: 0.002,
  beta2_parallel_weight: 0.015,
  lambda_parallel_penalty: 0.08,
  beta_grid: [0.2, 0.4, 0.6, 0.8, 1.0],
};

/**
 * Evaluates a single (policy, tree) trajectory under fixed beta.
 */
export function evaluatePolicyOnTree(
  policy: ExplorationPolicy,
  tree: DiscoveryTree,
  params: SimulatorHyperparams = DEFAULT_SIM_PARAMS,
): ReplayScore {
  const world = new HistoricalReplayWorld(tree);
  const closed = new Set<number>();
  const traces: ReplayRoundTrace[] = [];
  const curve: Array<{ probes: number; round: number; best_score: number }> = [];

  let bestQuality = tree.baseline_score;
  let totalProbes = 0;
  let decisionRounds = 0;
  let effectiveSeqRounds = 0;

  while (decisionRounds < params.K2_max_rounds) {
    const legal = world.legal_actions();
    if (legal.length === 0) break; // T_i^{m,k} == T_i

    const rawBatch = policy.select_batch(world, closed);
    if (!rawBatch || rawBatch.length === 0) break; // Policy terminated

    // Enforce legality and capacity W
    const legalSet = new Set(legal);
    const batch = Array.from(new Set(rawBatch))
      .filter((id) => legalSet.has(id))
      .slice(0, world.max_parallelism);

    if (batch.length === 0) break;

    decisionRounds += 1;
    effectiveSeqRounds += Math.ceil(batch.length / world.max_parallelism);

    const revealed = world.probe_batch(batch, () => {
      totalProbes += 1;
    });

    for (const obs of revealed) {
      if (obs.evaluated && obs.fail_class === "ok" && obs.score > bestQuality) {
        bestQuality = obs.score;
      }
    }

    curve.push({
      probes: totalProbes,
      round: decisionRounds,
      best_score: bestQuality,
    });

    traces.push({
      round: decisionRounds,
      selected_batch: batch,
      revealed_observations: revealed,
      best_score_so_far: bestQuality,
      total_probes: totalProbes,
    });
  }

  const parallelismRatio = totalProbes / Math.max(1, decisionRounds);
  const equation1Score =
    bestQuality -
    params.beta1_cost_weight * totalProbes +
    params.beta2_parallel_weight * parallelismRatio;

  return {
    tree_id: tree.tree_id,
    beta: policy.beta,
    best_quality: bestQuality,
    total_probes: totalProbes,
    decision_rounds: decisionRounds,
    effective_seq_rounds: effectiveSeqRounds,
    parallelism_ratio: parallelismRatio,
    equation1_score: equation1Score,
    curve,
    traces,
  };
}

/**
 * Evaluates a policy factory across all historical trees H_t and sweeps the beta grid.
 */
export function runBetaSweep(
  policyVersion: string,
  createPolicy: (beta: number) => ExplorationPolicy,
  history: DiscoveryTree[],
  params: SimulatorHyperparams = DEFAULT_SIM_PARAMS,
): { summary: BetaSweepSummary; allRuns: ReplayScore[] } {
  const perBeta: BetaSweepSummary["per_beta"] = {};
  const allRuns: ReplayScore[] = [];

  for (const beta of params.beta_grid) {
    const policy = createPolicy(beta);
    const scores = history.map((tree) => evaluatePolicyOnTree(policy, tree, params));
    allRuns.push(...scores);

    const n = Math.max(1, scores.length);
    const meanEq1 = scores.reduce((acc, s) => acc + s.equation1_score, 0) / n;
    const meanQuality = scores.reduce((acc, s) => acc + s.best_quality, 0) / n;
    const meanProbes = scores.reduce((acc, s) => acc + s.total_probes, 0) / n;
    const meanPenalty =
      scores.reduce(
        (acc, s) => acc + (s.total_probes > 0 ? s.effective_seq_rounds / s.total_probes : 1),
        0,
      ) / n;

    perBeta[beta] = {
      mean_eq1_score: meanEq1,
      mean_best_quality: meanQuality,
      mean_probes: meanProbes,
      parallel_penalty: meanPenalty,
    };
  }

  // Compute Pareto AUC of attainment vs normalized probes across the beta sweep
  const betas = [...params.beta_grid].sort((a, b) => a - b);
  const auc =
    betas.reduce((acc, b) => acc + perBeta[b].mean_best_quality, 0) / Math.max(1, betas.length);
  const parallelPenalty =
    betas.reduce((acc, b) => acc + perBeta[b].parallel_penalty, 0) / Math.max(1, betas.length);
  const reward = auc - params.lambda_parallel_penalty * parallelPenalty;
  const meanVScore =
    betas.reduce((acc, b) => acc + perBeta[b].mean_eq1_score, 0) / Math.max(1, betas.length);

  return {
    summary: {
      policy_version: policyVersion,
      betas,
      per_beta: perBeta,
      pareto: {
        auc,
        parallel_penalty: parallelPenalty,
        reward,
      },
      mean_v_score: meanVScore,
    },
    allRuns,
  };
}

```

---

### 4. `src/policy.ts`

Implements the prefix-only observation signal helpers (`branch_promising`, `branch_failed_hard`, `probe_improved_vs_parent`, `probe_improved_vs_baseline`, and trajectory classification) and the adaptive dynamic-portfolio `OptimalPolicy` class described in Appendix B.2.

```typescript
import type {
  ExplorationPolicy,
  GridPlan,
  GridPlanningContext,
  Observation,
  ReplayQuestion,
} from "./types.ts";

/**
 * Observation signal helpers matching `see.policy.observation_signal` from Appendix B.2.
 */
export function isSuccessfulEval(obs: Observation): boolean {
  return obs.evaluated && obs.error === null && obs.fail_class === "ok";
}

export function branchPromising(obs: Observation, baselineScore: number): boolean {
  return isSuccessfulEval(obs) && obs.score >= baselineScore;
}

export function branchFailedHard(obs: Observation): boolean {
  return obs.evaluated && obs.fail_class === "env_dependency";
}

export function isRepairableFailure(obs: Observation): boolean {
  if (!obs.evaluated || isSuccessfulEval(obs)) return false;
  return (
    obs.fail_class === "output_mismatch" ||
    obs.fail_class === "resource_limit" ||
    obs.fail_class === "shape_layout_error" ||
    obs.fail_class === "compile_other" ||
    obs.fail_class === "runtime_error"
  );
}

export function probeImprovedVsParent(obs: Observation): boolean {
  return isSuccessfulEval(obs) && obs.delta_vs_parent > 0;
}

export function probeImprovedVsBaseline(obs: Observation): boolean {
  return isSuccessfulEval(obs) && obs.delta_vs_baseline > 0;
}

export interface BranchTrajectory {
  branch: number;
  attempts: Observation[];
  latest: Observation;
  frontierActionId: string | null;
  successfulAnchor: number | null;
  scoreTrend: number;
  consecutiveFailures: number;
  status: "promising" | "repairable" | "underexplored" | "hard_unrecoverable" | "unpromising";
}

export interface BetaSchedule {
  maxConsecutiveFailures: number;
  patienceDepth: number;
  exploreQuotaRatio: number;
  minRelativeGain: number;
}

export class OptimalPolicy implements ExplorationPolicy {
  readonly name = "OptimalPolicy";
  readonly beta: number;

  constructor(config: { beta?: number } = {}) {
    this.beta = Number(config.beta ?? 0.6);
  }

  /**
   * Routes all behavioral thresholds through one deterministic schedule(beta).
   */
  private _schedule(beta: number): BetaSchedule {
    const clamped = Math.max(0.05, Math.min(1.0, beta));
    return {
      maxConsecutiveFailures: clamped >= 0.7 ? 3 : clamped >= 0.4 ? 2 : 1,
      patienceDepth: Math.max(2, Math.round(2 + clamped * 4)),
      exploreQuotaRatio: 0.25 + 0.35 * clamped,
      minRelativeGain: (1 - clamped) * 0.02,
    };
  }

  /**
   * Prefix-safe next-cycle grid planner (Appendix B.2 lines 190-243).
   */
  plan_grid(context: GridPlanningContext): GridPlan {
    const clampW = (w: number) =>
      Math.max(1, Math.min(context.hard_max_branch_count, Math.round(w)));
    const clampR = (r: number) =>
      Math.max(1, Math.min(context.hard_max_refine_count, Math.round(r)));

    if (!context.history || context.history.length < 2) {
      return {
        branch_count: clampW(context.fallback_branch_count),
        refine_count: clampR(context.fallback_refine_count),
        reason: "Insufficient live history (<2 cycles); using conservative bootstrap grid plan.",
      };
    }

    const recent = context.history.slice(-3);
    const prev = recent[recent.length - 2];
    const curr = recent[recent.length - 1];
    const scoreImproved = curr.final_best_score > prev.final_best_score;
    const avgDepthRatio =
      curr.max_depth_reached / Math.max(1, curr.planned_grid.refine_count);

    if (scoreImproved && avgDepthRatio >= 0.75) {
      return {
        branch_count: clampW(curr.planned_grid.branch_count),
        refine_count: clampR(curr.planned_grid.refine_count + 2),
        reason: "High gains arriving late on active branches; holding width and increasing refinement depth.",
      };
    }

    if (!scoreImproved) {
      return {
        branch_count: clampW(curr.planned_grid.branch_count + 2),
        refine_count: clampR(Math.max(3, curr.planned_grid.refine_count - 1)),
        reason: "Live best plateaued across recent depth; increasing branch width to uncover distinct directions.",
      };
    }

    return {
      branch_count: clampW(curr.planned_grid.branch_count),
      refine_count: clampR(curr.planned_grid.refine_count),
      reason: "Steady live progression; maintaining balanced width and depth.",
    };
  }

  /**
   * Reconstructs ordered prefix trajectories per opened branch and builds a dynamic portfolio batch.
   */
  select_batch(question: ReplayQuestion, closed: Set<number>): string[] {
    const prefix = question.observed();
    const sched = this._schedule(this.beta);
    const legalActions = question.legal_actions();
    if (legalActions.length === 0) return [];

    const legalRoots = new Set(question.legal_roots());
    const legalFrontiersByBranch = new Map<number, string>();
    for (const act of legalActions) {
      if (!legalRoots.has(act)) {
        const meta = question.meta(act);
        legalFrontiersByBranch.set(meta.branch, act);
      }
    }

    // Group revealed observations by branch ordered by attempt index
    const byBranch = new Map<number, Observation[]>();
    for (const obs of Object.values(prefix)) {
      const list = byBranch.get(obs.branch) ?? [];
      list.push(obs);
      byBranch.set(obs.branch, list);
    }

    const trajectories: BranchTrajectory[] = [];
    let globalBestAnchor = question.baseline_score;

    for (const [branch, list] of byBranch.entries()) {
      list.sort((a, b) => a.attempt - b.attempt);
      let anchor: number | null = null;
      let consecutiveFailures = 0;

      for (const obs of list) {
        if (isSuccessfulEval(obs)) {
          anchor = anchor === null ? obs.score : Math.max(anchor, obs.score);
          if (anchor > globalBestAnchor) globalBestAnchor = anchor;
          consecutiveFailures = 0; // Later success cancels earlier failure episode
        } else {
          consecutiveFailures += 1;
        }
      }

      const latest = list[list.length - 1];
      const scoreTrend = latest.delta_vs_parent;
      const frontierActionId = legalFrontiersByBranch.get(branch) ?? null;

      let status: BranchTrajectory["status"] = "promising";
      if (branchFailedHard(latest) && consecutiveFailures >= 2) {
        status = "hard_unrecoverable";
        closed.add(branch);
      } else if (consecutiveFailures > sched.maxConsecutiveFailures) {
        status = "unpromising";
        closed.add(branch);
      } else if (isRepairableFailure(latest)) {
        status = "repairable";
        closed.delete(branch);
      } else if (list.length < sched.patienceDepth) {
        status = "underexplored";
        closed.delete(branch);
      } else if (
        anchor !== null &&
        anchor < globalBestAnchor * (1 - sched.minRelativeGain) &&
        scoreTrend <= 0
      ) {
        status = "unpromising";
        closed.add(branch);
      }

      trajectories.push({
        branch,
        attempts: list,
        latest,
        frontierActionId,
        successfulAnchor: anchor,
        scoreTrend,
        consecutiveFailures,
        status,
      });
    }

    // Build dynamic portfolio queues: Exploitation, Exploration, and at most 1 Recovery
    const exploitationQueue: Array<{ id: string; priority: number }> = [];
    const explorationQueue: Array<{ id: string; priority: number }> = [];
    const recoveryQueue: Array<{ id: string; priority: number }> = [];

    for (const traj of trajectories) {
      if (closed.has(traj.branch) || !traj.frontierActionId) continue;

      if (traj.status === "repairable") {
        const anchorBonus = traj.successfulAnchor ?? question.baseline_score;
        recoveryQueue.push({
          id: traj.frontierActionId,
          priority: anchorBonus - traj.consecutiveFailures * 0.05,
        });
      } else if (traj.status === "underexplored") {
        const scoreSignal = traj.successfulAnchor ?? question.baseline_score;
        explorationQueue.push({
          id: traj.frontierActionId,
          priority: scoreSignal + (sched.patienceDepth - traj.attempts.length) * 0.1,
        });
      } else if (traj.status === "promising") {
        const anchor = traj.successfulAnchor ?? question.baseline_score;
        exploitationQueue.push({
          id: traj.frontierActionId,
          priority: anchor + Math.max(0, traj.scoreTrend),
        });
      }
    }

    // Add unopened roots to exploration queue
    for (const rootActionId of question.legal_roots()) {
      const meta = question.meta(rootActionId);
      explorationQueue.push({
        id: rootActionId,
        priority: question.baseline_score + this.beta * 0.2 - meta.branch * 1e-4,
      });
    }

    exploitationQueue.sort((a, b) => b.priority - a.priority);
    explorationQueue.sort((a, b) => b.priority - a.priority);
    recoveryQueue.sort((a, b) => b.priority - a.priority);

    const W = question.max_parallelism;
    const batch: string[] = [];
    const chosen = new Set<string>();

    const pushCandidate = (id: string) => {
      if (!chosen.has(id) && batch.length < W) {
        chosen.add(id);
        batch.push(id);
      }
    };

    // 1. Ensure representation for exploration and at most one justified recovery
    const targetExploreCount = Math.max(1, Math.floor(W * sched.exploreQuotaRatio));
    for (let i = 0; i < Math.min(targetExploreCount, explorationQueue.length); i++) {
      pushCandidate(explorationQueue[i].id);
    }

    if (recoveryQueue.length > 0 && batch.length < W) {
      pushCandidate(recoveryQueue[0].id);
    }

    // 2. Fill remaining parallel slots with top exploitation refinements
    for (const item of exploitationQueue) {
      pushCandidate(item.id);
    }

    // 3. Backfill any still-idle workers with remaining exploration or recovery actions
    for (const item of explorationQueue) {
      pushCandidate(item.id);
    }
    for (const item of recoveryQueue) {
      pushCandidate(item.id);
    }

    return batch;
  }
}

```

---

### 5. `src/prompts.ts`

Provides the exact prompt templates from **Appendix B.1 (Listing 1)** for online subagent worker execution and **Appendix B.2 (Listing 2)** for replay-based policy synthesis.

```typescript
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

```

---

### 6. `src/index.ts`

Wires DREAM-RSI into Oh My Pi's `ExtensionAPI`:

1. **`dream_rsi_init`**: Initializes `.dream-rsi/` workspace state, baseline policy $\pi_1$, and replay history $\mathcal{H}_0 = ()$.
2. **`dream_rsi_online_step`**: Steps or executes the online discovery rollout $\mathcal{T}_t$ using $\pi_t$ to emit parallel worker batches $C_t^k \in A(\mathcal{T}_t^k; W)$ and records evaluator outcomes (`eval/score.json`) into the live discovery tree.


3. **`dream_rsi_dream`**: Executes the offline dreaming loop across $\mathcal{H}_t$, running the $\beta$-sweep and Equation (1) across all recorded replay worlds, persisting `beta_sweep.json` and `policy_execution_traces.jsonl`, and selecting $\pi_{t+1} = \pi_t^{m^*}$ where $m^* \in \arg\max_m V^m$ ($V^{m^*} \ge V^0$).


4. **`/dream-rsi`**: Slash command for inspecting the simulator pool, triggering an offline dreaming sweep, or advancing the outer RSI loop.

```typescript
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { OptimalPolicy } from "./policy.ts";
import {
  buildOnlineExplorationPrompt,
  buildReplayPolicyImprovementPrompt,
} from "./prompts.ts";
import {
  DEFAULT_SIM_PARAMS,
  HistoricalReplayWorld,
  runBetaSweep,
} from "./simulator.ts";
import type {
  BetaSweepSummary,
  DiscoveryNode,
  DiscoveryTree,
  ExplorationPolicy,
  FailClass,
  GridPlanningContext,
  LiveCycleManifest,
  Observation,
} from "./types.ts";

interface DreamRsiWorkspaceConfig {
  workspace_root: string;
  eval_program: string;
  problem_file: string;
  eval_command: string;
  baseline_score: number;
  max_parallelism: number; // W
  K1_max_online_rounds: number;
  M_dream_revisions: number;
  hard_max_branch_count: number;
  hard_max_refine_count: number;
  current_iteration: number;
  active_policy_file: string;
  baked_beta: number;
}

export default function dreamRsiExtension(pi: ExtensionAPI) {
  const z = pi.zod;

  function getDreamRoot(cwd: string): string {
    return path.join(cwd, ".dream-rsi");
  }

  function loadConfig(cwd: string): DreamRsiWorkspaceConfig {
    const cfgPath = path.join(getDreamRoot(cwd), "config.json");
    if (!fs.existsSync(cfgPath)) {
      throw new Error(
        `DREAM-RSI workspace not initialized. Run dream_rsi_init first (missing ${cfgPath}).`,
      );
    }
    return JSON.parse(fs.readFileSync(cfgPath, "utf8")) as DreamRsiWorkspaceConfig;
  }

  function saveConfig(cwd: string, cfg: DreamRsiWorkspaceConfig): void {
    const root = getDreamRoot(cwd);
    fs.mkdirSync(root, { recursive: true });
    fs.writeFileSync(path.join(root, "config.json"), JSON.stringify(cfg, null, 2));
  }

  function loadHistoryTrees(cwd: string): DiscoveryTree[] {
    const poolDir = path.join(getDreamRoot(cwd), "trace_pool");
    if (!fs.existsSync(poolDir)) return [];
    const entries = fs.readdirSync(poolDir).sort();
    const trees: DiscoveryTree[] = [];
    for (const entry of entries) {
      const treeFile = path.join(poolDir, entry, "discovery_tree.json");
      if (fs.existsSync(treeFile)) {
        trees.push(JSON.parse(fs.readFileSync(treeFile, "utf8")) as DiscoveryTree);
      }
    }
    return trees;
  }

  function loadManifests(cwd: string): LiveCycleManifest[] {
    const poolDir = path.join(getDreamRoot(cwd), "trace_pool");
    if (!fs.existsSync(poolDir)) return [];
    const entries = fs.readdirSync(poolDir).sort();
    const manifests: LiveCycleManifest[] = [];
    for (const entry of entries) {
      const mf = path.join(poolDir, entry, "live_cycle_manifest.json");
      if (fs.existsSync(mf)) {
        manifests.push(JSON.parse(fs.readFileSync(mf, "utf8")) as LiveCycleManifest);
      }
    }
    return manifests;
  }

  async function loadPolicyFactory(
    policyFilePath: string,
  ): Promise<(beta: number) => ExplorationPolicy> {
    if (!fs.existsSync(policyFilePath)) {
      return (beta: number) => new OptimalPolicy({ beta });
    }
    try {
      const fileUrl = `${pathToFileURL(policyFilePath).href}?t=${Date.now()}`;
      const mod = await import(fileUrl);
      const PolicyClass = mod.OptimalPolicy ?? mod.default;
      if (typeof PolicyClass === "function") {
        return (beta: number) => new PolicyClass({ beta });
      }
    } catch {
      // Fallback to built-in OptimalPolicy if dynamic load fails
    }
    return (beta: number) => new OptimalPolicy({ beta });
  }

  /**
   * Pre-populates a skeleton grid of legal branches (0..W_grid-1) and refinement depths (0..R_grid)
   * so that online execution and offline replay share the exact same ReplayQuestion interface.
   */
  function createOnlineSkeletonTree(
    iteration: number,
    cfg: DreamRsiWorkspaceConfig,
    branchCount: number,
    refineCount: number,
    iterDir: string,
  ): DiscoveryTree {
    const nodes: Record<string, DiscoveryNode> = {};
    const rootId = "root";
    nodes[rootId] = {
      id: rootId,
      parent_id: null,
      branch: -1,
      attempt: -1,
      created_order: 0,
      workspace_dir: path.join(getDreamRoot(cfg.workspace_root), "baseline"),
      observation: {
        cell_id: rootId,
        parent_id: "",
        branch: -1,
        attempt: -1,
        score: cfg.baseline_score,
        evaluated: true,
        valid: true,
        fail_class: "ok",
        error: null,
        delta_vs_baseline: 0,
        delta_vs_parent: 0,
        n_valid: 1,
        n_total: 1,
      },
      meta: {
        cell_id: rootId,
        parent_id: "",
        branch: -1,
        attempt: -1,
        seq: 0,
        tags: ["root"],
      },
      children_ids: [],
    };

    const rootChildrenOrdered: string[] = [];
    let seq = 1;

    for (let b = 0; b < branchCount; b++) {
      let parentId = rootId;
      for (let a = 0; a <= refineCount; a++) {
        const cellId = `b${b}_a${a}`;
        const nodeDir = path.join(iterDir, `branch_${b}`, `attempt_${a}`);
        nodes[cellId] = {
          id: cellId,
          parent_id: parentId,
          branch: b,
          attempt: a,
          created_order: seq,
          workspace_dir: nodeDir,
          observation: {
            cell_id: cellId,
            parent_id: parentId,
            branch: b,
            attempt: a,
            score: cfg.baseline_score,
            evaluated: false,
            valid: false,
            fail_class: "ok",
            error: null,
            delta_vs_baseline: 0,
            delta_vs_parent: 0,
            n_valid: 0,
            n_total: 0,
          },
          meta: {
            cell_id: cellId,
            parent_id: parentId,
            branch: b,
            attempt: a,
            seq,
            tags: a === 0 ? ["root_branch"] : ["refinement"],
          },
          children_ids: [],
        };
        nodes[parentId].children_ids.push(cellId);
        if (a === 0) {
          rootChildrenOrdered.push(cellId);
        }
        parentId = cellId;
        seq++;
      }
    }

    return {
      tree_id: `T_${iteration}`,
      iteration,
      root_id: rootId,
      baseline_score: cfg.baseline_score,
      max_parallelism: cfg.max_parallelism,
      planned_branch_count: branchCount,
      planned_refine_count: refineCount,
      baked_beta: cfg.baked_beta,
      nodes,
      root_children_ordered: rootChildrenOrdered,
    };
  }

  // ---------------------------------------------------------------------------
  // Tool 1: dream_rsi_init
  // ---------------------------------------------------------------------------
  pi.registerTool({
    name: "dream_rsi_init",
    label: "DREAM-RSI Initialize Workspace",
    description:
      "Initialize a DREAM-RSI meta-exploration workspace (.dream-rsi/) with baseline score, parallel worker budget W, target eval_program, and initial exploration policy pi_1.",
    parameters: z.object({
      eval_program: z.string().describe("Relative path to target code file to evolve (e.g. solver.py)"),
      problem_file: z.string().default("PROBLEM.md").describe("Problem specification file"),
      eval_command: z
        .string()
        .describe("Shell command to evaluate a candidate; writes eval/score.json in node_dir"),
      baseline_score: z.number().describe("Initial baseline score s_r at the root node"),
      max_parallelism: z.number().int().min(1).default(8).describe("Number of parallel workers W"),
      K1_max_online_rounds: z.number().int().min(1).default(12).describe("Max decision rounds K1 per online rollout"),
      M_dream_revisions: z.number().int().min(1).default(5).describe("Number of offline policy revisions M per dreaming phase"),
      hard_max_branch_count: z.number().int().min(1).default(32).describe("Hard cap on branches W_grid"),
      hard_max_refine_count: z.number().int().min(1).default(20).describe("Hard cap on refinement steps R_grid"),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const root = getDreamRoot(ctx.cwd);
      const policiesDir = path.join(root, "policies", "baseline");
      const poolDir = path.join(root, "trace_pool");
      const baselineDir = path.join(root, "baseline");

      fs.mkdirSync(policiesDir, { recursive: true });
      fs.mkdirSync(poolDir, { recursive: true });
      fs.mkdirSync(baselineDir, { recursive: true });

      const activePolicyFile = path.join(policiesDir, "method.ts");
      const localPolicySource = path.join(path.dirname(new URL(import.meta.url).pathname), "policy.ts");
      if (fs.existsSync(localPolicySource)) {
        fs.copyFileSync(localPolicySource, activePolicyFile);
      }

      const cfg: DreamRsiWorkspaceConfig = {
        workspace_root: ctx.cwd,
        eval_program: params.eval_program,
        problem_file: params.problem_file,
        eval_command: params.eval_command,
        baseline_score: params.baseline_score,
        max_parallelism: params.max_parallelism,
        K1_max_online_rounds: params.K1_max_online_rounds,
        M_dream_revisions: params.M_dream_revisions,
        hard_max_branch_count: params.hard_max_branch_count,
        hard_max_refine_count: params.hard_max_refine_count,
        current_iteration: 1,
        active_policy_file: activePolicyFile,
        baked_beta: 0.6,
      };
      saveConfig(ctx.cwd, cfg);

      return {
        content: [
          {
            type: "text",
            text: `Initialized DREAM-RSI workspace at ${root} (W=${cfg.max_parallelism}, K1=${cfg.K1_max_online_rounds}, M=${cfg.M_dream_revisions}, baseline_score=${cfg.baseline_score}).`,
          },
        ],
        details: cfg,
      };
    },
  });

  // ---------------------------------------------------------------------------
  // Tool 2: dream_rsi_plan_online_batch
  // ---------------------------------------------------------------------------
  pi.registerTool({
    name: "dream_rsi_plan_online_batch",
    label: "DREAM-RSI Plan Next Online Batch",
    description:
      "Run the active exploration policy pi_t on the current online discovery tree T_t^k to select the next parallel batch C_t^k of up to W worker nodes, and generate Listing 1 subagent prompts for each selected node.",
    parameters: z.object({
      direction_guidance: z
        .string()
        .optional()
        .describe("Optional high-level directional guidance (leave empty for unguided exploration)"),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const cfg = loadConfig(ctx.cwd);
      const root = getDreamRoot(ctx.cwd);
      const iterTag = `iter_${String(cfg.current_iteration).padStart(4, "0")}`;
      const iterDir = path.join(root, "trace_pool", iterTag);
      fs.mkdirSync(iterDir, { recursive: true });

      const createPolicy = await loadPolicyFactory(cfg.active_policy_file);
      const policy = createPolicy(cfg.baked_beta);

      const stateFile = path.join(iterDir, "online_state.json");
      let tree: DiscoveryTree;
      let revealedCells: string[] = [];
      let closedBranches: number[] = [];
      let round = 0;

      if (!fs.existsSync(stateFile)) {
        const gridCtx: GridPlanningContext = {
          history: loadManifests(ctx.cwd),
          hard_max_branch_count: cfg.hard_max_branch_count,
          hard_max_refine_count: cfg.hard_max_refine_count,
          fallback_branch_count: Math.min(10, cfg.hard_max_branch_count),
          fallback_refine_count: Math.min(11, cfg.hard_max_refine_count),
          max_parallelism: cfg.max_parallelism,
        };
        const plan = policy.plan_grid(gridCtx);
        tree = createOnlineSkeletonTree(
          cfg.current_iteration,
          cfg,
          plan.branch_count,
          plan.refine_count,
          iterDir,
        );
      } else {
        const saved = JSON.parse(fs.readFileSync(stateFile, "utf8"));
        tree = saved.tree;
        revealedCells = saved.revealedCells;
        closedBranches = saved.closedBranches ?? [];
        round = saved.round;
      }

      if (round >= cfg.K1_max_online_rounds) {
        return {
          content: [
            {
              type: "text",
              text: `Online rollout iteration ${cfg.current_iteration} reached K1=${cfg.K1_max_online_rounds} decision rounds. Call dream_rsi_finalize_rollout to commit T_${cfg.current_iteration} to the replay simulator pool.`,
            },
          ],
          details: { terminated: true, reason: "K1_limit_reached", round },
        };
      }

      // Reconstruct prefix-only ReplayQuestion state up to current round k
      const world = new HistoricalReplayWorld(tree);
      for (const cid of revealedCells) {
        world.probe_batch([cid]);
      }

      const closedSet = new Set<number>(closedBranches);
      const selectedBatch = policy.select_batch(world, closedSet);

      if (selectedBatch.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `Policy ${policy.name} (beta=${cfg.baked_beta}) selected an empty batch C_t^k = ∅ at round ${round}. Call dream_rsi_finalize_rollout to commit T_${cfg.current_iteration}.`,
            },
          ],
          details: { terminated: true, reason: "policy_stopped", round },
        };
      }

      // Prepare workspace directories and Listing 1 prompts for each worker in C_t^k
      const workerAssignments = selectedBatch.map((cellId) => {
        const node = tree.nodes[cellId];
        const parentNode = tree.nodes[node.parent_id ?? tree.root_id];
        fs.mkdirSync(path.join(node.workspace_dir, "eval"), { recursive: true });

        // Copy parent's candidate artifact into child workspace as starting point
        const parentArtifact = path.join(parentNode.workspace_dir, cfg.eval_program);
        const childArtifact = path.join(node.workspace_dir, cfg.eval_program);
        if (fs.existsSync(parentArtifact) && !fs.existsSync(childArtifact)) {
          fs.copyFileSync(parentArtifact, childArtifact);
        }

        const prompt = buildOnlineExplorationPrompt({
          direction_guidance: params.direction_guidance,
          node_dir: node.workspace_dir,
          parent_dir: parentNode.workspace_dir,
          history_dir: path.join(root, "trace_pool"),
          baseline_dir: path.join(root, "baseline"),
          eval_program: cfg.eval_program,
          problem_file: cfg.problem_file,
        });

        return {
          cell_id: cellId,
          branch: node.branch,
          attempt: node.attempt,
          parent_id: node.parent_id,
          workspace_dir: node.workspace_dir,
          subagent_prompt: prompt,
        };
      });

      fs.writeFileSync(
        stateFile,
        JSON.stringify(
          {
            tree,
            revealedCells,
            closedBranches: Array.from(closedSet),
            round,
            pendingBatch: selectedBatch,
          },
          null,
          2,
        ),
      );

      return {
        content: [
          {
            type: "text",
            text: `Round ${round + 1}/${cfg.K1_max_online_rounds}: Policy selected batch of ${workerAssignments.length} parallel attempts: [${selectedBatch.join(", ")}]. Fan out subagents using 'task' for each worker assignment, then call 'dream_rsi_record_online_batch'.`,
          },
        ],
        details: {
          iteration: cfg.current_iteration,
          round: round + 1,
          batch: selectedBatch,
          worker_assignments: workerAssignments,
        },
      };
    },
  });

  // ---------------------------------------------------------------------------
  // Tool 3: dream_rsi_record_online_batch
  // ---------------------------------------------------------------------------
  pi.registerTool({
    name: "dream_rsi_record_online_batch",
    label: "DREAM-RSI Record Online Batch Outcomes",
    description:
      "Evaluate and record the outcomes of the current parallel batch C_t^k into the discovery tree T_t^{k+1}, or commit T_t to the replay simulator pool if the rollout is complete.",
    parameters: z.object({
      outcomes: z
        .array(
          z.object({
            cell_id: z.string(),
            score: z.number(),
            valid: z.boolean(),
            fail_class: z
              .enum([
                "ok",
                "compile_other",
                "output_mismatch",
                "resource_limit",
                "shape_layout_error",
                "runtime_error",
                "env_dependency",
                "timeout",
              ])
              .default("ok"),
            error: z.string().nullable().default(null),
            n_valid: z.number().int().default(1),
            n_total: z.number().int().default(1),
          }),
        )
        .describe("Evaluation outcomes for each cell in the executed batch"),
      finalize_rollout: z
        .boolean()
        .default(false)
        .describe("Set true to finalize T_t and append it to the replay simulator history H_t"),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const cfg = loadConfig(ctx.cwd);
      const root = getDreamRoot(ctx.cwd);
      const iterTag = `iter_${String(cfg.current_iteration).padStart(4, "0")}`;
      const iterDir = path.join(root, "trace_pool", iterTag);
      const stateFile = path.join(iterDir, "online_state.json");

      if (!fs.existsSync(stateFile)) {
        throw new Error(`No active online rollout state found at ${stateFile}`);
      }

      const saved = JSON.parse(fs.readFileSync(stateFile, "utf8"));
      const tree: DiscoveryTree = saved.tree;
      const revealedCells: string[] = saved.revealedCells;

      for (const item of params.outcomes) {
        const node = tree.nodes[item.cell_id];
        if (!node) continue;
        const parentNode = tree.nodes[node.parent_id ?? tree.root_id];
        const parentScore =
          parentNode && parentNode.observation.evaluated
            ? parentNode.observation.score
            : tree.baseline_score;

        const obs: Observation = {
          cell_id: item.cell_id,
          parent_id: node.parent_id ?? tree.root_id,
          branch: node.branch,
          attempt: node.attempt,
          score: item.score,
          evaluated: true,
          valid: item.valid,
          fail_class: item.fail_class as FailClass,
          error: item.error,
          delta_vs_baseline: item.score - tree.baseline_score,
          delta_vs_parent: item.score - parentScore,
          n_valid: item.n_valid,
          n_total: item.n_total,
        };

        node.observation = obs;
        if (!revealedCells.includes(item.cell_id)) {
          revealedCells.push(item.cell_id);
        }

        // Write eval/score.json and error.txt inside the node directory per Listing 1 contract
        const evalDir = path.join(node.workspace_dir, "eval");
        fs.mkdirSync(evalDir, { recursive: true });
        fs.writeFileSync(path.join(evalDir, "score.json"), JSON.stringify(obs, null, 2));
        if (obs.error) {
          fs.writeFileSync(path.join(evalDir, "error.txt"), obs.error);
        }
      }

      const nextRound = Number(saved.round) + 1;
      fs.writeFileSync(
        stateFile,
        JSON.stringify(
          {
            tree,
            revealedCells,
            closedBranches: saved.closedBranches,
            round: nextRound,
            pendingBatch: [],
          },
          null,
          2,
        ),
      );

      if (params.finalize_rollout || nextRound >= cfg.K1_max_online_rounds) {
        // Prune unrevealed skeleton nodes so T_t contains only the realized discovery tree
        const realizedSet = new Set([tree.root_id, ...revealedCells]);
        const realizedNodes: Record<string, DiscoveryNode> = {};
        for (const [id, n] of Object.entries(tree.nodes)) {
          if (realizedSet.has(id)) {
            realizedNodes[id] = {
              ...n,
              children_ids: n.children_ids.filter((cid) => realizedSet.has(cid)),
            };
          }
        }
        tree.nodes = realizedNodes;
        tree.root_children_ordered = tree.root_children_ordered.filter((cid) =>
          realizedSet.has(cid),
        );

        fs.writeFileSync(
          path.join(iterDir, "discovery_tree.json"),
          JSON.stringify(tree, null, 2),
        );

        // Compute LiveCycleManifest sidecar
        let bestScore = tree.baseline_score;
        let maxDepth = 0;
        const branchesOpened = new Set<number>();
        for (const id of revealedCells) {
          const n = tree.nodes[id];
          if (!n) continue;
          branchesOpened.add(n.branch);
          if (n.attempt + 1 > maxDepth) maxDepth = n.attempt + 1;
          if (n.observation.evaluated && n.observation.fail_class === "ok") {
            if (n.observation.score > bestScore) bestScore = n.observation.score;
          }
        }

        const manifest: LiveCycleManifest = {
          iteration: cfg.current_iteration,
          tree_id: tree.tree_id,
          policy_version: path.basename(path.dirname(cfg.active_policy_file)),
          baked_beta: cfg.baked_beta,
          planned_grid: {
            branch_count: tree.planned_branch_count,
            refine_count: tree.planned_refine_count,
            reason: "Recorded live rollout grid",
          },
          opened_width: branchesOpened.size,
          max_depth_reached: maxDepth,
          probe_work: revealedCells.length,
          decision_rounds: nextRound,
          baseline_score: tree.baseline_score,
          final_best_score: bestScore,
          timestamp: new Date().toISOString(),
        };

        fs.writeFileSync(
          path.join(iterDir, "live_cycle_manifest.json"),
          JSON.stringify(manifest, null, 2),
        );

        return {
          content: [
            {
              type: "text",
              text: `Committed discovery tree ${tree.tree_id} (${revealedCells.length} evaluated probes, best_score=${bestScore}) to replay simulator pool. Run 'dream_rsi_dream' to evaluate and evolve exploration policies over H_${cfg.current_iteration}.`,
            },
          ],
          details: { finalized: true, manifest },
        };
      }

      return {
        content: [
          {
            type: "text",
            text: `Recorded ${params.outcomes.length} outcomes for round ${nextRound}. Call 'dream_rsi_plan_online_batch' for round ${nextRound + 1}.`,
          },
        ],
        details: { finalized: false, round: nextRound, total_revealed: revealedCells.length },
      };
    },
  });

  // ---------------------------------------------------------------------------
  // Tool 4: dream_rsi_dream (Offline Replay Evaluation & Policy Selection)
  // ---------------------------------------------------------------------------
  pi.registerTool({
    name: "dream_rsi_dream",
    label: "DREAM-RSI Offline Dreaming Simulator & Policy Selector",
    description:
      "Evaluate a candidate exploration policy file (or all M candidate policy revisions) against the accumulated replay simulator pool H_t using Equation (1) and the beta-grid Pareto sweep. Automatically selects pi_{t+1} = argmax_m V^m.",
    parameters: z.object({
      candidate_policy_file: z
        .string()
        .optional()
        .describe("Path to a revised policy file pi_t^m to evaluate in the replay simulator"),
      revision_index: z
        .number()
        .int()
        .min(0)
        .default(0)
        .describe("Policy revision index m in {0, ..., M-1}"),
      select_and_advance: z
        .boolean()
        .default(false)
        .describe("If true, selects pi_{t+1} = argmax_m V^m across all evaluated revisions and advances outer iteration t -> t+1"),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const cfg = loadConfig(ctx.cwd);
      const root = getDreamRoot(ctx.cwd);
      const history = loadHistoryTrees(ctx.cwd);

      if (history.length === 0) {
        throw new Error(
          "Replay simulator pool H_t is empty. Complete at least one online rollout before dreaming.",
        );
      }

      const policyPath = params.candidate_policy_file
        ? path.resolve(ctx.cwd, params.candidate_policy_file)
        : cfg.active_policy_file;

      const revTag = `r${String(cfg.current_iteration).padStart(2, "0")}_m${params.revision_index}`;
      const revDir = path.join(root, "policies", revTag, "proposal_results");
      fs.mkdirSync(revDir, { recursive: true });

      // Archive policy snapshot into history directory
      const archivedPolicyPath = path.join(root, "policies", revTag, "method.ts");
      if (fs.existsSync(policyPath) && policyPath !== archivedPolicyPath) {
        fs.copyFileSync(policyPath, archivedPolicyPath);
      }

      const createPolicy = await loadPolicyFactory(archivedPolicyPath);
      const { summary, allRuns } = runBetaSweep(revTag, createPolicy, history, DEFAULT_SIM_PARAMS);

      // Save beta_sweep.json and policy_execution_traces.jsonl per Appendix B.2
      fs.writeFileSync(
        path.join(revDir, "beta_sweep.json"),
        JSON.stringify(summary, null, 2),
      );
      const jsonlTraces = allRuns.map((r) => JSON.stringify(r)).join("\n") + "\n";
      fs.writeFileSync(path.join(revDir, "policy_execution_traces.jsonl"), jsonlTraces);

      // Prepare Listing 2 prompt for generating the next policy revision pi_t^{m+1}
      const nextRevTag = `r${String(cfg.current_iteration).padStart(2, "0")}_m${params.revision_index + 1}`;
      const nextPolicyFile = path.join(root, "policies", nextRevTag, "method.ts");
      fs.mkdirSync(path.dirname(nextPolicyFile), { recursive: true });
      if (fs.existsSync(archivedPolicyPath) && !fs.existsSync(nextPolicyFile)) {
        fs.copyFileSync(archivedPolicyPath, nextPolicyFile);
      }

      const improvementPrompt = buildReplayPolicyImprovementPrompt({
        method_file: nextPolicyFile,
        history_dir: path.join(root, "policies"),
        trace_pool: path.join(root, "trace_pool"),
        current_sweep_summary: JSON.stringify(summary, null, 2),
      });

      if (params.select_and_advance) {
        // Scan all evaluated revisions m in {0..M-1} for current outer iteration t and select m* = argmax_m V^m
        const policiesRoot = path.join(root, "policies");
        const prefixTag = `r${String(cfg.current_iteration).padStart(2, "0")}_m`;
        const revDirs = fs
          .readdirSync(policiesRoot)
          .filter((d) => d.startsWith(prefixTag))
          .sort();

        let bestRev = revTag;
        let bestVScore = summary.mean_v_score;
        let bestPolicyPath = archivedPolicyPath;
        let bestSweep: BetaSweepSummary = summary;

        for (const d of revDirs) {
          const sweepFile = path.join(policiesRoot, d, "proposal_results", "beta_sweep.json");
          const candidateMethod = path.join(policiesRoot, d, "method.ts");
          if (fs.existsSync(sweepFile) && fs.existsSync(candidateMethod)) {
            const s = JSON.parse(fs.readFileSync(sweepFile, "utf8")) as BetaSweepSummary;
            if (s.mean_v_score >= bestVScore) {
              bestVScore = s.mean_v_score;
              bestRev = d;
              bestPolicyPath = candidateMethod;
              bestSweep = s;
            }
          }
        }

        // Cross-cycle default-beta rule (Appendix B.2 lines 169-188)
        const manifests = loadManifests(ctx.cwd);
        let nextBeta = cfg.baked_beta;
        if (manifests.length >= 2) {
          const prevBest = manifests[manifests.length - 2].final_best_score;
          const currBest = manifests[manifests.length - 1].final_best_score;
          if (currBest <= prevBest) {
            nextBeta = Math.min(1.0, Number((cfg.baked_beta + 0.15).toFixed(2)));
          }
        }

        cfg.active_policy_file = bestPolicyPath;
        cfg.baked_beta = nextBeta;
        cfg.current_iteration += 1;
        saveConfig(ctx.cwd, cfg);

        return {
          content: [
            {
              type: "text",
              text: `Selected optimal policy pi_${cfg.current_iteration} = ${bestRev} (V^{m*} = ${bestVScore.toFixed(4)}, pareto.reward = ${bestSweep.pareto.reward.toFixed(4)}, next baked_beta = ${nextBeta}) across ${history.length} replay worlds. Advanced to outer iteration t=${cfg.current_iteration}.`,
            },
          ],
          details: {
            selected_revision: bestRev,
            selected_policy_file: bestPolicyPath,
            mean_v_score: bestVScore,
            next_iteration: cfg.current_iteration,
            next_baked_beta: nextBeta,
          },
        };
      }

      return {
        content: [
          {
            type: "text",
            text: `Evaluated policy revision ${revTag} across ${history.length} replay world(s): V^m = ${summary.mean_v_score.toFixed(4)}, pareto.reward = ${summary.pareto.reward.toFixed(4)}. To synthesize revision m=${params.revision_index + 1}, apply the returned policy improvement prompt to '${nextPolicyFile}', then call 'dream_rsi_dream' with revision_index=${params.revision_index + 1}.`,
          },
        ],
        details: {
          revision: revTag,
          summary,
          next_policy_file: nextPolicyFile,
          policy_improvement_prompt: improvementPrompt,
        },
      };
    },
  });

  // ---------------------------------------------------------------------------
  // Slash Command: /dream-rsi
  // ---------------------------------------------------------------------------
  pi.registerCommand("dream-rsi", {
    description:
      "DREAM-RSI controller: '/dream-rsi status', '/dream-rsi dream', or '/dream-rsi rollout'",
    handler: async (args: string, ctx: ExtensionContext) => {
      const subcmd = args.trim().split(/\s+/)[0] || "status";
      const root = getDreamRoot(ctx.cwd);

      if (!fs.existsSync(path.join(root, "config.json"))) {
        ctx.ui.notify(
          "DREAM-RSI not initialized in this workspace. Ask the agent to call dream_rsi_init.",
          "warning",
        );
        return;
      }

      const cfg = loadConfig(ctx.cwd);
      const history = loadHistoryTrees(ctx.cwd);
      const manifests = loadManifests(ctx.cwd);

      if (subcmd === "status") {
        const bestEver = manifests.reduce(
          (acc, m) => Math.max(acc, m.final_best_score),
          cfg.baseline_score,
        );
        const totalProbes = manifests.reduce((acc, m) => acc + m.probe_work, 0);
        ctx.ui.notify(
          `DREAM-RSI | iter t=${cfg.current_iteration} | worlds |H|=${history.length} | best=${bestEver.toFixed(6)} | total_probes=${totalProbes} | beta=${cfg.baked_beta}`,
          "info",
        );
        return;
      }

      if (subcmd === "dream") {
        pi.sendMessage(
          {
            customType: "dream-rsi-directive",
            content: `Run the DREAM-RSI offline dreaming phase (M=${cfg.M_dream_revisions} revisions) over the ${history.length} recorded replay worlds in .dream-rsi/trace_pool/:
1. Call 'dream_rsi_dream' with revision_index=0 to evaluate the current policy pi_${cfg.current_iteration}^0.
2. For m = 0..${cfg.M_dream_revisions - 2}, inspect 'beta_sweep.json' and 'policy_execution_traces.jsonl', edit the next candidate policy 'method.ts' following the Listing 2 constraints, and evaluate it with 'dream_rsi_dream'.
3. On revision m=${cfg.M_dream_revisions - 1}, pass 'select_and_advance: true' to deploy pi_${cfg.current_iteration + 1} = argmax_m V^m.`,
            display: true,
            attribution: "user",
          },
          { triggerTurn: true },
        );
        return;
      }

      if (subcmd === "rollout") {
        pi.sendMessage(
          {
            customType: "dream-rsi-directive",
            content: `Execute DREAM-RSI online rollout iteration t=${cfg.current_iteration}:
1. Call 'dream_rsi_plan_online_batch' to obtain the parallel node batch C_t^k (up to W=${cfg.max_parallelism} workers) and Listing 1 prompts.
2. Fan out parallel subagent workers via 'task' for each selected node in C_t^k, run '${cfg.eval_command}', and record scores via 'dream_rsi_record_online_batch'.
3. Repeat until the policy returns an empty batch or K1=${cfg.K1_max_online_rounds} rounds complete, finalizing T_${cfg.current_iteration} into the replay simulator pool.`,
            display: true,
            attribution: "user",
          },
          { triggerTurn: true },
        );
      }
    },
  });
}

```

---

### How the 3-Stage Loop Operates in `omp`

1. **Initialize**: Run `dream_rsi_init` with your target `eval_program`, `eval_command`, `baseline_score`, and worker concurrency $W$ (`max_parallelism`).
2. **Stage 1 & 2 — Online Exploration & Simulator Construction (`/dream-rsi rollout`)**:
* `dream_rsi_plan_online_batch` invokes `policy.plan_grid(context)` before the live episode to size the branch/refinement grid $(W_{\text{grid}}, R_{\text{grid}})$, then queries `policy.select_batch(question, closed)` at each round $k \le K_1$ to emit a batch $C_t^k \in A(\mathcal{T}_t^k; W)$.


* `omp` fans out its native parallel `task` subagents across the batch using the **Listing 1** prompt, evaluates each candidate, and calls `dream_rsi_record_online_batch` to persist `eval/score.json`, `discovery_tree.json`, and `live_cycle_manifest.json` into `.dream-rsi/trace_pool/iter_XXXX/`.




3. **Stage 3 — Offline Dreaming (`/dream-rsi dream`)**:
* `dream_rsi_dream` loads all historical trees $\mathcal{H}_t = \{\mathcal{T}_1, \dots, \mathcal{T}_t\}$ into `HistoricalReplayWorld` instances.


* For each candidate policy revision $\pi_t^m$ ($m = 0, \dots, M-1$), it sweeps $\beta \in [0.2, 0.4, 0.6, 0.8, 1.0]$ at zero execution cost, scoring both Equation (1) ($V_i^m$) and the Pareto attainment/parallelism objective, and deploys $\pi_{t+1} = \pi_t^{m^*}$ ($V^{m^*} \ge V^0$) for the next online rollout.





