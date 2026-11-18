> [!WARNING]
> # THIS IS UNPROVEN VIBE-CODED EXPERIMENT

# omp-plugin-dream-rsi

DREAM-RSI (recursive self-improvement through replayed discovery trees) as an [Oh My Pi](https://omp.sh) (`omp`) extension. Online rollouts record discovery trees; offline "dreaming" replays them at zero execution cost to evaluate and select improved exploration policies across a beta sweep.

Generated from the conversation in [`docs/conversation.md`](docs/conversation.md).

## Install

```sh
omp plugin install github:gavmor/dream-rsi-plugin#v0.1.0
```

## Usage

- Tools: `dream_rsi_init`, `dream_rsi_plan_online_batch`, `dream_rsi_record_online_batch`, `dream_rsi_dream`
- Command: `/dream-rsi status | rollout | dream`

State lives in `.dream-rsi/` in the workspace.

## Development

```sh
bun install
bun run typecheck
bun test
```

Releases: bump `version` in `package.json`, then push a matching `v*` tag; the Release workflow reruns CI and attaches the packed tarball.
