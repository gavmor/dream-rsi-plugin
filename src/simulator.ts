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
