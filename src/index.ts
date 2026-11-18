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
  const initParams = z.object({
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
  });
  pi.registerTool<typeof initParams>({
    name: "dream_rsi_init",
    label: "DREAM-RSI Initialize Workspace",
    description:
      "Initialize a DREAM-RSI meta-exploration workspace (.dream-rsi/) with baseline score, parallel worker budget W, target eval_program, and initial exploration policy pi_1.",
    parameters: initParams,
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
  const planParams = z.object({
      direction_guidance: z
        .string()
        .optional()
        .describe("Optional high-level directional guidance (leave empty for unguided exploration)"),
  });
  pi.registerTool<typeof planParams>({
    name: "dream_rsi_plan_online_batch",
    label: "DREAM-RSI Plan Next Online Batch",
    description:
      "Run the active exploration policy pi_t on the current online discovery tree T_t^k to select the next parallel batch C_t^k of up to W worker nodes, and generate Listing 1 subagent prompts for each selected node.",
    parameters: planParams,
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
  const recordParams = z.object({
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
  });
  pi.registerTool<typeof recordParams>({
    name: "dream_rsi_record_online_batch",
    label: "DREAM-RSI Record Online Batch Outcomes",
    description:
      "Evaluate and record the outcomes of the current parallel batch C_t^k into the discovery tree T_t^{k+1}, or commit T_t to the replay simulator pool if the rollout is complete.",
    parameters: recordParams,
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
  const dreamParams = z.object({
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
  });
  pi.registerTool<typeof dreamParams>({
    name: "dream_rsi_dream",
    label: "DREAM-RSI Offline Dreaming Simulator & Policy Selector",
    description:
      "Evaluate a candidate exploration policy file (or all M candidate policy revisions) against the accumulated replay simulator pool H_t using Equation (1) and the beta-grid Pareto sweep. Automatically selects pi_{t+1} = argmax_m V^m.",
    parameters: dreamParams,
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