import { describe, expect, it } from "bun:test";
import {
  OptimalPolicy,
  branchFailedHard,
  branchPromising,
  isRepairableFailure,
  isSuccessfulEval,
} from "../src/policy.ts";
import { HistoricalReplayWorld } from "../src/simulator.ts";
import type { GridPlanningContext, LiveCycleManifest } from "../src/types.ts";
import { buildTree } from "./fixtures.ts";

describe("observation signal helpers", () => {
  const tree = buildTree({
    branches: [
      [{ score: 1.3 }],
      [{ score: 0.7 }],
      [{ score: 0, fail_class: "runtime_error" }],
      [{ score: 0, fail_class: "env_dependency" }],
    ],
  });
  const obs = (id: string) => tree.nodes[id].observation;

  it("treats only clean evaluations as successful", () => {
    expect(isSuccessfulEval(obs("b0_a0"))).toBe(true);
    expect(isSuccessfulEval(obs("b2_a0"))).toBe(false);
    expect(isSuccessfulEval({ ...obs("b0_a0"), evaluated: false })).toBe(false);
  });

  it("calls a branch promising only at or above baseline", () => {
    expect(branchPromising(obs("b0_a0"), 1.0)).toBe(true);
    expect(branchPromising(obs("b1_a0"), 1.0)).toBe(false);
  });

  it("separates repairable code failures from hard environment failures", () => {
    expect(isRepairableFailure(obs("b2_a0"))).toBe(true);
    expect(branchFailedHard(obs("b2_a0"))).toBe(false);
    expect(isRepairableFailure(obs("b3_a0"))).toBe(false);
    expect(branchFailedHard(obs("b3_a0"))).toBe(true);
    expect(isRepairableFailure(obs("b0_a0"))).toBe(false);
  });
});

describe("OptimalPolicy.plan_grid", () => {
  const ctx = (history: LiveCycleManifest[], over: Partial<GridPlanningContext> = {}) => ({
    history,
    hard_max_branch_count: 32,
    hard_max_refine_count: 20,
    fallback_branch_count: 10,
    fallback_refine_count: 11,
    max_parallelism: 4,
    ...over,
  });
  const manifest = (best: number, branches: number, refine: number, depth: number) =>
    ({
      final_best_score: best,
      planned_grid: { branch_count: branches, refine_count: refine, reason: "" },
      max_depth_reached: depth,
    }) as LiveCycleManifest;
  const plan = (history: LiveCycleManifest[], over?: Partial<GridPlanningContext>) => {
    const { branch_count, refine_count } = new OptimalPolicy().plan_grid(ctx(history, over));
    return [branch_count, refine_count];
  };

  it("bootstraps from the fallback grid with fewer than two cycles", () => {
    expect(plan([])).toEqual([10, 11]);
    expect(plan([manifest(2, 4, 4, 4)])).toEqual([10, 11]);
  });

  it("clamps every plan to the hard caps", () => {
    expect(plan([], { fallback_branch_count: 50, fallback_refine_count: 50 })).toEqual([32, 20]);
    expect(plan([manifest(1, 10, 20, 20), manifest(2, 10, 20, 20)])).toEqual([10, 20]);
  });

  it("deepens when gains arrive late in the last cycle", () => {
    expect(plan([manifest(1.0, 10, 8, 8), manifest(1.2, 10, 8, 6)])).toEqual([10, 10]);
  });

  it("widens when the best score plateaus", () => {
    expect(plan([manifest(1.0, 10, 8, 8), manifest(1.0, 10, 8, 8)])).toEqual([12, 7]);
  });

  it("holds the grid on steady shallow progress", () => {
    expect(plan([manifest(1.0, 10, 8, 8), manifest(1.2, 10, 8, 2)])).toEqual([10, 8]);
  });
});

describe("OptimalPolicy.select_batch", () => {
  it("only ever proposes legal, unique batches within W until it stops", () => {
    const branches = [0, 1, 2, 3, 4].map((b) =>
      [0, 1, 2].map((a) =>
        (a + b) % 4 === 3
          ? { score: 0, fail_class: "runtime_error" as const }
          : { score: 1 + 0.1 * b - 0.05 * a },
      ),
    );
    for (const beta of [0.2, 0.6, 1.0]) {
      const world = new HistoricalReplayWorld(buildTree({ W: 3, branches }));
      const policy = new OptimalPolicy({ beta });
      const closed = new Set<number>();
      let probes = 0;
      for (let round = 0; round < 100; round++) {
        const batch = policy.select_batch(world, closed);
        if (batch.length === 0) break;
        expect(batch.length).toBeLessThanOrEqual(3);
        expect(new Set(batch).size).toBe(batch.length);
        world.probe_batch(batch); // throws on any non-frontier cell
        probes += batch.length;
      }
      expect(probes).toBeGreaterThan(0);
      expect(policy.select_batch(world, closed)).toEqual([]);
    }
  });

  it("decides only from the revealed prefix", () => {
    const make = (hidden: number) =>
      new HistoricalReplayWorld(
        buildTree({
          W: 3,
          branches: [
            [{ score: 1.2 }, { score: hidden }, { score: hidden }],
            [{ score: 0.8 }, { score: hidden }],
            [{ score: hidden }],
          ],
        }),
      );
    const low = make(-100);
    const high = make(100);
    for (const world of [low, high]) world.probe_batch(["b0_a0", "b1_a0"]);

    expect(new OptimalPolicy().select_batch(high, new Set())).toEqual(
      new OptimalPolicy().select_batch(low, new Set()),
    );
  });

  describe("a branch with two consecutive repairable failures", () => {
    const world = () => {
      const w = new HistoricalReplayWorld(
        buildTree({
          W: 2,
          branches: [
            [
              { score: 0, fail_class: "runtime_error" },
              { score: 0, fail_class: "runtime_error" },
              { score: 2.0 },
            ],
            [{ score: 1.1 }],
          ],
        }),
      );
      w.probe_batch(["b0_a0"]);
      w.probe_batch(["b0_a1"]);
      return w;
    };

    it("is closed at low beta", () => {
      const closed = new Set<number>();
      const batch = new OptimalPolicy({ beta: 0.2 }).select_batch(world(), closed);
      expect(closed.has(0)).toBe(true);
      expect(batch).not.toContain("b0_a2");
      expect(batch).toContain("b1_a0");
    });

    it("gets one recovery attempt at high beta", () => {
      const closed = new Set<number>([0]);
      const batch = new OptimalPolicy({ beta: 0.8 }).select_batch(world(), closed);
      expect(closed.has(0)).toBe(false);
      expect(batch).toEqual(["b1_a0", "b0_a2"]);
    });
  });
});
