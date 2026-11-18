import { describe, expect, it } from "bun:test";
import {
  DEFAULT_SIM_PARAMS,
  HistoricalReplayWorld,
  evaluatePolicyOnTree,
  runBetaSweep,
} from "../src/simulator.ts";
import type { ExplorationPolicy } from "../src/types.ts";
import { buildTree, greedyFirstPolicy, scriptedPolicy } from "./fixtures.ts";

// b2_a0 failed with a high score: failures must never count as quality.
const tree = () =>
  buildTree({
    W: 2,
    branches: [
      [{ score: 1.2 }, { score: 1.5 }],
      [{ score: 0.9 }],
      [{ score: 9.9, fail_class: "runtime_error" }],
    ],
  });

describe("HistoricalReplayWorld", () => {
  it("starts with nothing observed and only root branches legal", () => {
    const world = new HistoricalReplayWorld(tree());
    expect(world.observed()).toEqual({});
    expect(world.legal_roots()).toEqual(["b0_a0", "b1_a0", "b2_a0"]);
    expect(world.legal_actions()).toEqual(["b0_a0", "b1_a0", "b2_a0"]);
    expect(world.opened_branches()).toEqual([]);
  });

  it("reveals only probed cells and exposes each leaf's next refinement", () => {
    const world = new HistoricalReplayWorld(tree());
    world.probe_batch(["b0_a0"]);

    expect(Object.keys(world.observed())).toEqual(["b0_a0"]);
    expect(world.legal_roots()).toEqual(["b1_a0", "b2_a0"]);
    expect(world.legal_actions()).toEqual(["b1_a0", "b2_a0", "b0_a1"]);
    expect(world.opened_branches()).toEqual([0]);
  });

  it("does not let callers mutate the recorded tree through observations", () => {
    const t = tree();
    const world = new HistoricalReplayWorld(t);
    world.probe_batch(["b0_a0"]);
    world.observed().b0_a0.score = -1;
    expect(world.observed().b0_a0.score).toBe(1.2);
    expect(t.nodes.b0_a0.observation.score).toBe(1.2);
  });

  it("rejects cells that are not on the frontier", () => {
    const world = new HistoricalReplayWorld(tree());
    expect(() => world.probe_batch(["b0_a1"])).toThrow(/non-frontier/);
  });

  it("rejects batches wider than max_parallelism", () => {
    const world = new HistoricalReplayWorld(tree());
    expect(() => world.probe_batch(["b0_a0", "b1_a0", "b2_a0"])).toThrow(/exceeds max_parallelism/);
  });

  it("reset() forgets every revealed cell", () => {
    const world = new HistoricalReplayWorld(tree());
    world.probe_batch(["b0_a0", "b1_a0"]);
    world.reset();
    expect(world.observed()).toEqual({});
    expect(world.legal_actions()).toEqual(["b0_a0", "b1_a0", "b2_a0"]);
  });
});

describe("evaluatePolicyOnTree", () => {
  it("scores a trajectory with Equation (1) and ignores failed cells' scores", () => {
    const policy = scriptedPolicy([
      ["b0_a0", "b1_a0"],
      ["b0_a1", "b2_a0"],
    ]);
    const score = evaluatePolicyOnTree(policy, tree());

    expect(score.total_probes).toBe(4);
    expect(score.decision_rounds).toBe(2);
    expect(score.best_quality).toBe(1.5);
    expect(score.parallelism_ratio).toBe(2);
    // 1.5 - 0.002 * 4 + 0.015 * 2
    expect(score.equation1_score).toBeCloseTo(1.522, 10);
    expect(score.curve).toEqual([
      { probes: 2, round: 1, best_score: 1.2 },
      { probes: 4, round: 2, best_score: 1.5 },
    ]);
  });

  it("drops duplicate, illegal and over-capacity cells instead of failing", () => {
    const policy = scriptedPolicy([["b0_a1", "b0_a0", "b0_a0", "b1_a0", "b2_a0"]]);
    const score = evaluatePolicyOnTree(policy, tree());
    expect(score.traces[0].selected_batch).toEqual(["b0_a0", "b1_a0"]);
    expect(score.total_probes).toBe(2);
  });

  it("returns the baseline when the policy stops immediately", () => {
    const score = evaluatePolicyOnTree(scriptedPolicy([]), tree());
    expect(score.total_probes).toBe(0);
    expect(score.decision_rounds).toBe(0);
    expect(score.best_quality).toBe(1.0);
  });

  it("stops at K2_max_rounds", () => {
    const score = evaluatePolicyOnTree(greedyFirstPolicy(), tree(), {
      ...DEFAULT_SIM_PARAMS,
      K2_max_rounds: 1,
    });
    expect(score.decision_rounds).toBe(1);
    expect(score.total_probes).toBe(1);
  });

  it("stops once the recorded tree is exhausted", () => {
    const score = evaluatePolicyOnTree(greedyFirstPolicy(), tree());
    expect(score.total_probes).toBe(4);
    expect(score.best_quality).toBe(1.5);
  });
});

describe("runBetaSweep", () => {
  // Explores greedily at low beta, refuses to explore at high beta.
  const betaGated = (beta: number): ExplorationPolicy =>
    beta < 0.5 ? greedyFirstPolicy(beta) : scriptedPolicy([], beta);

  it("evaluates every beta against every tree and aggregates per beta", () => {
    const history = [tree(), buildTree({ W: 2, tree_id: "T_2", branches: [[{ score: 2.0 }]] })];
    const params = { ...DEFAULT_SIM_PARAMS, beta_grid: [0.8, 0.2] };
    const { summary, allRuns } = runBetaSweep("v0", betaGated, history, params);

    expect(allRuns).toHaveLength(4);
    expect(summary.betas).toEqual([0.2, 0.8]);
    expect(summary.per_beta[0.2].mean_best_quality).toBeCloseTo((1.5 + 2.0) / 2, 10);
    expect(summary.per_beta[0.2].mean_probes).toBe((4 + 1) / 2);
    expect(summary.per_beta[0.8].mean_best_quality).toBe(1.0);
    expect(summary.per_beta[0.8].mean_probes).toBe(0);
    // Zero-probe runs count as fully sequential.
    expect(summary.per_beta[0.8].parallel_penalty).toBe(1);
    expect(summary.pareto.auc).toBeCloseTo((1.75 + 1.0) / 2, 10);
    expect(summary.pareto.reward).toBeCloseTo(
      summary.pareto.auc - params.lambda_parallel_penalty * summary.pareto.parallel_penalty,
      10,
    );
  });
});
