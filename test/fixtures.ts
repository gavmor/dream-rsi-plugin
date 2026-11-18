import type {
  DiscoveryNode,
  DiscoveryTree,
  ExplorationPolicy,
  FailClass,
  Observation,
  ReplayQuestion,
} from "../src/types.ts";

export interface AttemptSpec {
  score: number;
  fail_class?: FailClass;
  error?: string | null;
}

/**
 * Builds a recorded discovery tree: `branches[b][a]` is attempt `a` of root branch `b`,
 * stored as cell `b{b}_a{a}` with attempt `a` the unique child of attempt `a-1`.
 */
export function buildTree(opts: {
  branches: AttemptSpec[][];
  baseline?: number;
  W?: number;
  tree_id?: string;
}): DiscoveryTree {
  const baseline = opts.baseline ?? 1.0;
  const rootObs: Observation = {
    cell_id: "root",
    parent_id: "",
    branch: -1,
    attempt: -1,
    score: baseline,
    evaluated: true,
    valid: true,
    fail_class: "ok",
    error: null,
    delta_vs_baseline: 0,
    delta_vs_parent: 0,
    n_valid: 1,
    n_total: 1,
  };
  const nodes: Record<string, DiscoveryNode> = {
    root: {
      id: "root",
      parent_id: null,
      branch: -1,
      attempt: -1,
      created_order: 0,
      workspace_dir: "/nonexistent/root",
      observation: rootObs,
      meta: { cell_id: "root", parent_id: "", branch: -1, attempt: -1, seq: 0, tags: ["root"] },
      children_ids: [],
    },
  };
  const rootChildren: string[] = [];
  let seq = 1;

  opts.branches.forEach((attempts, b) => {
    let parentId = "root";
    let parentScore = baseline;
    attempts.forEach((spec, a) => {
      const id = `b${b}_a${a}`;
      const failClass = spec.fail_class ?? "ok";
      nodes[id] = {
        id,
        parent_id: parentId,
        branch: b,
        attempt: a,
        created_order: seq,
        workspace_dir: `/nonexistent/${id}`,
        observation: {
          cell_id: id,
          parent_id: parentId,
          branch: b,
          attempt: a,
          score: spec.score,
          evaluated: true,
          valid: failClass === "ok",
          fail_class: failClass,
          error: spec.error ?? (failClass === "ok" ? null : failClass),
          delta_vs_baseline: spec.score - baseline,
          delta_vs_parent: spec.score - parentScore,
          n_valid: failClass === "ok" ? 1 : 0,
          n_total: 1,
        },
        meta: {
          cell_id: id,
          parent_id: parentId,
          branch: b,
          attempt: a,
          seq,
          tags: a === 0 ? ["root_branch"] : ["refinement"],
        },
        children_ids: [],
      };
      nodes[parentId].children_ids.push(id);
      if (a === 0) rootChildren.push(id);
      parentId = id;
      parentScore = spec.score;
      seq++;
    });
  });

  return {
    tree_id: opts.tree_id ?? "T_test",
    iteration: 1,
    root_id: "root",
    baseline_score: baseline,
    max_parallelism: opts.W ?? 2,
    planned_branch_count: opts.branches.length,
    planned_refine_count: Math.max(0, ...opts.branches.map((a) => a.length - 1)),
    baked_beta: 0.6,
    nodes,
    root_children_ordered: rootChildren,
  };
}

/** Policy that replays a fixed list of batches, then stops. */
export function scriptedPolicy(batches: string[][], beta = 0.5): ExplorationPolicy {
  let round = 0;
  return {
    name: "Scripted",
    beta,
    plan_grid: () => ({ branch_count: 1, refine_count: 1, reason: "scripted" }),
    select_batch: () => batches[round++] ?? [],
  };
}

/** Policy that probes the first legal action each round until the tree is exhausted. */
export function greedyFirstPolicy(beta = 0.5): ExplorationPolicy {
  return {
    name: "GreedyFirst",
    beta,
    plan_grid: () => ({ branch_count: 1, refine_count: 1, reason: "greedy" }),
    select_batch: (q: ReplayQuestion) => q.legal_actions().slice(0, 1),
  };
}
