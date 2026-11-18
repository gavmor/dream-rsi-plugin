import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { type Tool, validateToolArguments } from "@oh-my-pi/pi-ai";
import type { Extension, ExtensionCommandContext } from "@oh-my-pi/pi-coding-agent";
import {
  ExtensionRuntime,
  loadExtensions,
} from "@oh-my-pi/pi-coding-agent/extensibility/extensions/loader";
import type { BetaSweepSummary, DiscoveryTree, LiveCycleManifest, Observation } from "../src/types.ts";

// Shapes of the `details` each tool returns (see src/index.ts).
interface WorkspaceConfig {
  current_iteration: number;
  max_parallelism: number;
  K1_max_online_rounds: number;
  active_policy_file: string;
}
interface PlanDetails {
  batch: string[];
  worker_assignments: Array<{ cell_id: string; workspace_dir: string; subagent_prompt: string }>;
}
interface RecordDetails {
  finalized: boolean;
  round?: number;
  total_revealed?: number;
}
interface DreamDetails {
  summary: BetaSweepSummary;
  next_policy_file: string;
  policy_improvement_prompt: string;
}
interface AdvanceDetails {
  next_iteration: number;
  selected_policy_file: string;
}
interface SentMessage {
  message: { customType: string; content: string };
  options: { triggerTurn?: boolean };
}

const PLUGIN = path.resolve(import.meta.dir, "../src/index.ts");

let cwd: string;
let root: string;
let sent: SentMessage[];
let notes: Array<{ message: string; type?: string }>;
let plugin: Extension;
let ctx: ExtensionCommandContext;

beforeEach(async () => {
  cwd = fs.mkdtempSync(path.join(os.tmpdir(), "dream-rsi-test-"));
  root = path.join(cwd, ".dream-rsi");
  sent = [];
  notes = [];
  // Only the fields the plugin reads; the real context is built by the host session.
  ctx = {
    cwd,
    ui: { notify: (message: string, type?: string) => notes.push({ message, type }) },
  } as unknown as ExtensionCommandContext;

  // The host wires action methods onto the runtime after loading (runner.ts); do the same.
  const runtime = new ExtensionRuntime();
  Object.assign(runtime, {
    sendMessage: (message: SentMessage["message"], options: SentMessage["options"]) =>
      sent.push({ message, options }),
  });
  const result = await loadExtensions([PLUGIN], cwd, undefined, runtime);
  expect(result.errors).toEqual([]);
  plugin = result.extensions[0];
});

afterEach(() => {
  fs.rmSync(cwd, { recursive: true, force: true });
});

/** Runs a tool the way omp does: validate the arguments (applying defaults), then execute. */
async function call<T>(name: string, args: Record<string, unknown>): Promise<T> {
  const tool = plugin.tools.get(name);
  if (!tool) throw new Error(`tool ${name} not registered`);
  // Extension ToolDefinitions carry the name/description/parameters the validator reads.
  const params = validateToolArguments(tool.definition as unknown as Tool, {
    type: "toolCall",
    id: "call-1",
    name,
    arguments: args,
  });
  const result = await tool.definition.execute("call-1", params, undefined, undefined, ctx);
  // `details` is an in-process value whose shape the tool defines; see interfaces above.
  return result.details as T;
}

function readJson<T>(...parts: string[]): T {
  return JSON.parse(fs.readFileSync(path.join(root, ...parts), "utf8"));
}

function init() {
  return call<WorkspaceConfig>("dream_rsi_init", {
    eval_program: "solver.py",
    eval_command: "python eval.py",
    baseline_score: 1.0,
    max_parallelism: 2,
    M_dream_revisions: 2,
    hard_max_branch_count: 3,
    hard_max_refine_count: 2,
  });
}

/** Plans and records one round, scoring every selected cell with `score`. */
async function round(score: number, finalize_rollout = false) {
  const plan = await call<PlanDetails>("dream_rsi_plan_online_batch", {});
  const recorded = await call<RecordDetails>("dream_rsi_record_online_batch", {
    outcomes: plan.batch.map((cell_id) => ({ cell_id, score, valid: true })),
    finalize_rollout,
  });
  return { plan, recorded };
}

describe("registration", () => {
  it("registers the four tools and the /dream-rsi command", () => {
    expect([...plugin.tools.keys()].sort()).toEqual([
      "dream_rsi_dream",
      "dream_rsi_init",
      "dream_rsi_plan_online_batch",
      "dream_rsi_record_online_batch",
    ]);
    expect([...plugin.commands.keys()]).toEqual(["dream-rsi"]);
  });
});

describe("tools", () => {
  it("refuses to plan before the workspace is initialized", async () => {
    await expect(call("dream_rsi_plan_online_batch", {})).rejects.toThrow(/not initialized/);
  });

  it("refuses to dream with an empty replay pool", async () => {
    await init();
    await expect(call("dream_rsi_dream", {})).rejects.toThrow(/pool H_t is empty/);
  });

  it("initializes config and seeds the baseline policy", async () => {
    const cfg = await init();
    expect(cfg.current_iteration).toBe(1);
    expect(cfg.max_parallelism).toBe(2);
    expect(cfg.K1_max_online_rounds).toBe(12); // schema default
    expect(readJson<WorkspaceConfig>("config.json")).toEqual(cfg);
    expect(fs.existsSync(path.join(root, "policies", "baseline", "method.ts"))).toBe(true);
  });

  it("runs a full rollout, dreams over it, and advances to the next iteration", async () => {
    await init();
    fs.writeFileSync(path.join(root, "baseline", "solver.py"), "print('baseline')\n");

    // Round 1: the first batch opens root branches and seeds each worker from its parent.
    const first = await round(1.4);
    expect(first.plan.batch).toHaveLength(2);
    for (const w of first.plan.worker_assignments) {
      expect(w.cell_id).toMatch(/^b\d+_a0$/);
      expect(w.subagent_prompt).toContain(w.workspace_dir);
      expect(fs.readFileSync(path.join(w.workspace_dir, "solver.py"), "utf8")).toBe(
        "print('baseline')\n",
      );
      const score: Observation = JSON.parse(
        fs.readFileSync(path.join(w.workspace_dir, "eval", "score.json"), "utf8"),
      );
      expect(score.delta_vs_baseline).toBeCloseTo(0.4, 10);
    }
    expect(first.recorded).toMatchObject({ finalized: false, round: 1, total_revealed: 2 });

    // Round 2 never re-selects a revealed cell; finalizing commits T_1 to the pool.
    const second = await round(1.7, true);
    for (const cell of second.plan.batch) expect(first.plan.batch).not.toContain(cell);
    expect(second.recorded.finalized).toBe(true);

    const revealed = [...first.plan.batch, ...second.plan.batch];
    const tree = readJson<DiscoveryTree>("trace_pool", "iter_0001", "discovery_tree.json");
    expect(Object.keys(tree.nodes).sort()).toEqual(["root", ...revealed].sort());
    const manifest = readJson<LiveCycleManifest>("trace_pool", "iter_0001", "live_cycle_manifest.json");
    expect(manifest).toMatchObject({ probe_work: revealed.length, final_best_score: 1.7 });

    // Dreaming replays the pool across the beta grid and stages the next revision.
    const dream = await call<DreamDetails>("dream_rsi_dream", { revision_index: 0 });
    const sweep = readJson<BetaSweepSummary>("policies", "r01_m0", "proposal_results", "beta_sweep.json");
    expect(sweep.betas).toEqual([0.2, 0.4, 0.6, 0.8, 1.0]);
    expect(sweep.pareto.auc).toBeGreaterThan(1.0);
    const traces = fs
      .readFileSync(
        path.join(root, "policies", "r01_m0", "proposal_results", "policy_execution_traces.jsonl"),
        "utf8",
      )
      .trim()
      .split("\n");
    expect(traces).toHaveLength(5); // 5 betas x 1 tree
    expect(fs.existsSync(dream.next_policy_file)).toBe(true);
    expect(dream.policy_improvement_prompt).toContain(dream.next_policy_file);

    const advanced = await call<AdvanceDetails>("dream_rsi_dream", {
      revision_index: 1,
      select_and_advance: true,
    });
    expect(advanced.next_iteration).toBe(2);
    const cfg = readJson<WorkspaceConfig>("config.json");
    expect(cfg.current_iteration).toBe(2);
    expect(cfg.active_policy_file).toBe(advanced.selected_policy_file);
    expect(fs.existsSync(cfg.active_policy_file)).toBe(true);
  });

  it("replays an edited candidate policy, not the built-in one", async () => {
    await init();
    await round(1.5, true);

    fs.writeFileSync(
      path.join(cwd, "candidate.ts"),
      `export class OptimalPolicy {
  name = "OptimalPolicy";
  beta: number;
  constructor(cfg: { beta?: number } = {}) { this.beta = cfg.beta ?? 0.6; }
  plan_grid() { return { branch_count: 1, refine_count: 1, reason: "stub" }; }
  select_batch() { return []; }
}
`,
    );

    const builtIn = await call<DreamDetails>("dream_rsi_dream", { revision_index: 0 });
    const stub = await call<DreamDetails>("dream_rsi_dream", {
      revision_index: 1,
      candidate_policy_file: "candidate.ts",
    });

    expect(builtIn.summary.per_beta[0.6].mean_probes).toBeGreaterThan(0);
    for (const beta of stub.summary.betas) {
      expect(stub.summary.per_beta[beta].mean_probes).toBe(0);
      expect(stub.summary.per_beta[beta].mean_best_quality).toBe(1.0);
    }
  });
});

describe("/dream-rsi command", () => {
  const run = (args: string) => plugin.commands.get("dream-rsi")!.handler(args, ctx);

  it("warns when the workspace is not initialized", async () => {
    await run("status");
    expect(notes).toEqual([{ message: expect.stringContaining("not initialized"), type: "warning" }]);
    expect(sent).toEqual([]);
  });

  it("reports status after init", async () => {
    await init();
    await run("");
    expect(notes[0].type).toBe("info");
    expect(notes[0].message).toContain("iter t=1");
    expect(notes[0].message).toContain("|H|=0");
  });

  it("asks the agent to run a rollout or a dreaming phase", async () => {
    await init();
    await run("rollout");
    await run("dream");
    expect(sent.map((s) => s.message.customType)).toEqual([
      "dream-rsi-directive",
      "dream-rsi-directive",
    ]);
    expect(sent[0].message.content).toContain("dream_rsi_plan_online_batch");
    expect(sent[1].message.content).toContain("dream_rsi_dream");
    expect(sent.every((s) => s.options.triggerTurn)).toBe(true);
  });
});
