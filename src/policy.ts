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
