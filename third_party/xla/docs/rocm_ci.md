# AMD ROCm Continuous Integration

This document describes the AMD ROCm continuous integration (CI) implementation.
It covers the GitHub Actions workflow, runner pool, Docker containers, Bazel
configurations, tag filters, GPU locking, and the upstream-fork relationship
with [`ROCm/xla`](https://github.com/ROCm/xla).

The setup comprises two distinct paths:

1.  **ROCm CI workflow** (`.github/workflows/rocm_ci.yml`): Runs physical XLA
    and JAX tests on a self-hosted AMD Instinct GPU runner.
2.  **Compile-only ROCm CI** (`.github/workflows/ci.yml`): Builds XLA against a
    hermetic ROCm toolchain on a generic Linux x86 runner without a GPU.

These paths ensure both runtime verification on hardware and compilation
reliability on every commit.

## Trigger and concurrency

`rocm_ci.yml` runs on:

-   `pull_request` against `main` (the `jax` job is additionally gated to
    `github.base_ref == 'main'`).
-   `workflow_dispatch` (manual).

The compile-only ROCm path inside `ci.yml` runs on the same `pull_request`
events as the rest of the CI matrix and at post-submit.

Both workflows share a concurrency group keyed by workflow + head ref, so a new
push to a PR cancels any in-flight run on that PR but does not cancel runs on
`main`:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: ${{ github.ref != 'main' }}
```

Permissions are restricted to `contents: read`. No write scopes or artifact
uploads are enabled.

## Runner and container configuration

### Runner pool

The ROCm jobs use a self-hosted GitHub Actions runner labelled
`linux-x86-64-1gpu-amd`. This is a Linux x86_64 host with a single AMD Instinct
GPU attached, currently targeting `gfx950`. The label links to the AMD-owned RBE
runner fleet, where the build and tests run. The status of this cluster can be
monitored on (this dashboard)[https://wardite.cluster.engflow.com/].

### Docker image

The jobs run in a Docker container. The Docker image is provided by AMD and
derived from the official Tensorflow build images.

**Key points:**

-   `--device=/dev/kfd` and `--device=/dev/dri` expose the AMD GPU kernel fusion
    driver and DRI render nodes to the container. Without these the container
    cannot see the GPU.
-   `--group-add video` puts the container's user in the `video` group, which is
    the convention on Ubuntu for users that may access GPU device nodes.
-   `--ipc=host` and `--shm-size=64G` are needed because HIP/ROCm collectives
    (RCCL) and several stream-executor pathways use shared memory for
    inter-process communication on a single node.
-   `--cap-add=SYS_PTRACE` and `--security-opt=seccomp=unconfined` enable
    sanitizer-instrumented test binaries to attach debuggers and bypass seccomp
    filters that would block sanitizer runtime calls.
-   `--tmpfs /root/.cache/bazel:rw,exec,size=80g` puts the Bazel disk cache in
    RAM. `exec` is mandatory because Bazel runs `bazel-out` binaries directly
    out of the cache.
-   The two `ci-cert.*` files mounted in from `/data` on the host are the client
    TLS certificate and private key used to authenticate to the EngFlow remote
    build cluster.

## Job topology

`rocm_ci.yml` defines three jobs:

| Job           | Purpose                                    | Depends on    |
| ------------- | ------------------------------------------ | ------------- |
| `rocm-config` | Publish the pinned Docker image digest.    | —             |
| `jax`         | Build and test JAX against changes to XLA, | `rocm-config` |
:               : on a single AMD GPU.                       :               :
| `xla`         | Build and test XLA itself, on a single AMD | `rocm-config` |
:               : GPU, plus a CPU-only XLA suite.            :               :

`jax` and `xla` run in parallel once `rocm-config` finishes.

The `xla` job is *not* gated on `github.base_ref == 'main'`; the `jax` job is.
This means the JAX suite only runs for PRs targeted at `main`, while the XLA
suite runs for any PR plus `workflow_dispatch`.

`timeout-minutes` is set at the job level, with a tighter `timeout-minutes: 60`
(XLA single-GPU, JAX) or 80 (XLA CPU suite) on each test step.

## XLA CI job

### Step sequence

1.  Check out `openxla/xla` at the PR head.
2.  Download the AMD-maintained driver script (`execute_ci_build_upstream.sh`)
    from the `ROCm/xla` fork.
3.  Print CPU (`lscpu`) and GPU (`rocminfo`) information for debuggability.
4.  Run the test phases (Single-GPU and CPU) using the downloaded script.

### What `--config=ci_single_gpu` and `--config=ci_rocm_cpu` mean

These are defined in `build_tools/rocm/rocm_xla.bazelrc`. The single-GPU suite
uses the parallel GPU helper, retries flakes three times, and runs tests in a
process-isolated way. The CPU suite uses 200-wide local parallelism and the
sanitizer wrapper:

```
build:ci_single_gpu --run_under=//build_tools/rocm:parallel_gpu_execute
build:ci_single_gpu --flaky_test_attempts=3

build:ci_rocm_cpu --run_under=//build_tools/rocm:sanitizer_wrapper
build:ci_rocm_cpu --local_test_jobs=200
build:ci_rocm_cpu --strategy=TestRunner=local
```

A third config, `ci_multi_gpu`, exists in the bazelrc but is not currently
triggered from `rocm_ci.yml` — it is used by the standalone helper
`run_xla_multi_gpu.sh` on hosts with ≥4 GPUs.

### Tag filtering

`execute_ci_build_upstream.sh` (in the `ROCm/xla` fork) composes a tag filter
list using the helper at `build_tools/rocm/rocm_tag_filters.sh`. The legacy
in-tree wrapper `build_tools/rocm/run_xla_ci_build.sh` shows the same pattern:

```bash
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)
for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_rocm_cpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-requires-gpu-rocm,-requires-gpu-amd"
    fi
done
```

The base list from `rocm_tag_filters.sh` is the "things ROCm CI should *never*
attempt":

```
-no_gpu
-requires-gpu-intel
-requires-gpu-nvidia
-requires-gpu-cuda
-cuda-only
-oneapi-only
-requires-gpu-sm60
-requires-gpu-sm60-only
-requires-gpu-sm70
-requires-gpu-sm70-only
-requires-gpu-sm80
-requires-gpu-sm80-only
-requires-gpu-sm86
-requires-gpu-sm86-only
-requires-gpu-sm89
-requires-gpu-sm89-only
-requires-gpu-sm90
-requires-gpu-sm90-only
-skip_rocprofiler_sdk
-no_oss
-oss_excluded
-oss_serial
```

The per-config additions then narrow the set:

Config          | Adds
--------------- | -----------------------------------------------------
`ci_single_gpu` | `requires-gpu-rocm`, `requires-gpu-amd`, `-multi_gpu`
`ci_multi_gpu`  | `requires-gpu-rocm`, `requires-gpu-amd`, `multi_gpu`
`ci_rocm_cpu`   | `gpu`, `-requires-gpu-rocm`, `-requires-gpu-amd`

Note: The `gpu` tag includes tests that touch GPU code paths but can execute on
CPU. The CPU portion excludes tests requiring a physical AMD/ROCm GPU, allowing
it to exercise GPU codegen and the autotuner harness against the host CPU.

### Tests excluded by name

Collectives, sharding propagation, and distributed PJRT require multiple AMD
GPUs and are filtered from single-GPU runs. These must be executed on a
dedicated multi-GPU host.

## JAX CI job

The JAX job runs in the same container but with a different driver script,
provided by JAX itself:

```bash
./ci/run_bazel_test_rocm_rbe.sh \
  --override_repository=xla="${GITHUB_WORKSPACE}" \
  --override_module=xla="${GITHUB_WORKSPACE}" \
  --config=single_gpu \
  --//jax:build_jaxlib=wheel \
  --//jax:build_jax=true \
  --local_test_jobs=1 \
  --action_env=JAX_ENABLE_X64=0 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.14 \
  --repo_env=TF_ROCM_RBE_DOCKER_IMAGE="${DOCKER_IMAGE}"
```

**Flag details:**

-   `--override_repository=xla=...` and `--override_module=xla=...` redirect
    JAX's `xla` dependency at the in-tree checkout of the PR. Without these the
    JAX job would compile against the pinned XLA commit recorded in JAX's
    WORKSPACE/MODULE.bazel, which would defeat the point of running JAX from an
    XLA PR.
-   `--//jax:build_jaxlib=wheel` and `--//jax:build_jax=true` build jaxlib as a
    wheel artifact and rebuild jax itself from source — this is how JAX
    currently bootstraps the C++ extension for tests.
-   `HERMETIC_PYTHON_VERSION=3.14` selects the Python toolchain hermetically
    (the build is reproducible regardless of the host Python).
-   `--local_test_jobs=1` keeps the GPU contention low; with only one AMD GPU on
    the runner there is no benefit to running JAX tests in parallel.
-   `JAX_ENABLE_X64=0` matches the JAX team's default test posture for fast CI.
-   The same `TF_ROCM_RBE_DOCKER_IMAGE` is passed down so that any remote worker
    the build leases from EngFlow runs the same ROCm image.

## Remote build execution (EngFlow)

ROCm CI uses an EngFlow remote build cluster
(`grpcs://wardite.cluster.engflow.com`) for both remote caching and remote
execution. The configuration lives in `build_tools/rocm/rocm_xla.bazelrc`.

Authentication to this cluster is handled via mounted certificates
(`/data/ci-cert.crt`, `/data/ci-cert.key`).

Key configurations:

-   `REMOTE_GPU_TESTING=1`: Enables remote-execution optimizations (e.g.,
    restricted filesystem access).
-   `@local_config_rocm//rocm:linux_x64`: Specifies the platform, assuming
    system-level ROCm installation on workers to resolve the
    `@local_config_rocm` repository.

There is also a `rocm_rbe_dynamic` config that builds locally but tests under
Bazel's dynamic scheduler (running each test action concurrently locally and
remotely, taking whichever finishes first):

```
build:rocm_rbe_dynamic --config=rocm_rbe
build:rocm_rbe_dynamic --spawn_strategy=local
test:rocm_rbe_dynamic --experimental_spawn_scheduler
test:rocm_rbe_dynamic --strategy=TestRunner=dynamic
test:rocm_rbe_dynamic --dynamic_mode=default
test:rocm_rbe_dynamic --dynamic_local_strategy=worker,standalone,local
test:rocm_rbe_dynamic --dynamic_remote_strategy=remote
test:rocm_rbe_dynamic --experimental_local_execution_delay=1000
test:rocm_rbe_dynamic --local_resources=cpu=HOST_CPUS*0.5
```

The XLA single-GPU step in `rocm_ci.yml` uses essentially this approach by
passing `--internal_spawn_scheduler --strategy=TestRunner=dynamic` directly on
the command line on top of `--config=rocm_rbe`.

The third "umbrella" config, `rocm_ci`, is loaded via:

```
# rocm_xla_ci.bazelrc
try-import /usertools/rocm.bazelrc
try-import %workspace%/build_tools/rocm/rocm_xla.bazelrc
```

`/usertools/rocm.bazelrc` is provided by the AMD-built Docker image (it encodes
the host's ROCm install paths); the in-tree bazelrc is layered on top.
`--config=rocm_ci` therefore means "use the in-image ROCm toolchain plus the
in-tree XLA configs".

## GPU locking

When a GPU test is run on the AMD host, Bazel invokes it via
`--run_under=//build_tools/rocm:parallel_gpu_execute`. The script
(`build_tools/rocm/parallel_gpu_execute.sh`) does three things:

1.  Figure out how many GPUs the runner actually has by calling `rocminfo` and
    counting `Name: *gfx*` lines.

    ```bash
    ROCMINFO=$(find -L "${TEST_SRCDIR:-.}" -name "rocminfo" -path "*/bin/rocminfo" | head -n 1)
    TF_GPU_COUNT=$($ROCMINFO | grep "Name: *gfx*" | wc -l)
    ```

    `rocminfo` is brought into the action's runfiles via the Bazel target's
    `data = ["//tensorflow/third_party/rocm/google:rocminfo"]` in
    `build_tools/rocm/BUILD`, so it is discoverable inside the remote-execution
    sandbox.

2.  If `TF_GPU_COUNT == 0` (e.g. on the RBE default pool, which has no GPU) the
    script just `exec "$@"` — the test still runs, but without any per-GPU
    isolation. Tests that genuinely need a GPU are expected to self-fail; this
    branch exists so that GPU-tagged tests that *can* actually run on CPU still
    execute on cheap workers.

3.  Otherwise, acquire a `flock`-based slot, one per `(gpu, slot_within_gpu)`
    pair, and export `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` to that
    GPU index for the duration of the test:

    ```bash
    for j in `seq 0 $((TF_TESTS_PER_GPU-1))`; do
      for i in `seq 0 $((TF_GPU_COUNT-1))`; do
        exec {lock_fd}>/var/lock/gpulock${i}_${j} || exit 1
        if flock -n "$lock_fd"; then
          (
            export CUDA_VISIBLE_DEVICES=$i
            export HIP_VISIBLE_DEVICES=$i
            "$TEST_BINARY" "$@"
          )
          …
        fi
      done
    done
    ```

Slots are filled across GPUs sequentially (e.g., Slot 0 on all GPUs first, then
Slot 1) to minimize contention before oversubscribing memory.

There is also a near-identical copy at `build_tools/ci/parallel_gpu_execute.sh`,
which is the CUDA-oriented sibling used by the NVIDIA jobs. They differ mainly
in that the ROCm version actually consults `rocminfo` at runtime, whereas the
CUDA version trusts `TF_GPU_COUNT=4` by default.

A second helper, `//build_tools/rocm:sanitizer_wrapper`, is used by
`ci_rocm_cpu` and `ci_multi_gpu`. It is a tiny generated shell script (`echo
'#!/bin/bash' > $@; echo 'exec "$$@"' >> $@`) whose only job is to declare the
sanitizer ignore lists as runfiles, so that changes to those ignore lists force
the affected test actions to re-run.

## Standalone helpers

Three scripts in `build_tools/rocm/` are designed to be run by hand against a
local AMD machine and follow the same conventions as CI:

-   **`run_xla.sh`** — single-GPU XLA test sweep. Detects GPU count via
    `rocm-smi`, computes `N_TEST_JOBS = TF_GPU_COUNT * TF_TESTS_PER_GPU`, pulls
    `gfx*` ID from `rocminfo` to set `TF_ROCM_AMDGPU_TARGETS`, and invokes
    `bazel test --config=rocm_ci --config=xla_sgpu …` with a similar tag filter
    recipe as CI (though they differ slightly in implementation details). Uses
    `--run_under=//build_tools/ci:parallel_gpu_execute` (the CUDA-style
    wrapper).
-   **`run_xla_multi_gpu.sh`** — multi-GPU XLA test sweep. Requires
    `TF_GPU_COUNT ≥ 4`, otherwise it exits silently. Uses `--config=xla_mgpu`,
    sets `NCCL_MAX_NCHANNELS=1`, and notably does *not* pass
    `--run_under=//build_tools/ci:parallel_gpu_execute` — collective tests need
    to see all the GPUs at once, so per-test GPU pinning would break them.
-   **`run_xla_ci_build.sh`** — the in-tree legacy version of what the CI now
    fetches from `ROCm/xla`. Useful as documentation for how `--config=…`
    arguments map to tag filters.

These three exist for AMD developers reproducing CI behaviour locally.

## Compile-only ROCm CI

Independently of the GPU runner, `.github/workflows/ci.yml` runs an `XLA Linux
x86 GPU ROCm` job on a generic Linux x86 runner with no GPU and no ROCm system
install. This ensures compilation reliability on all PRs independently of GPU
runner availability.

The runner pool and container for this job are:

```yaml
{
  pool: "linux-x86-n2-16",
  container: "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest",
  name: "XLA Linux x86 GPU ROCm",
  repo: "openxla/xla",
}
```

A single matrix entry; no `--device=/dev/kfd`; no AMD certificate volumes. The
job dispatches to `build_tools/ci/build.py`, which is the configuration-as-data
driver shared with every other CI matrix entry:

```yaml
- run: |
    "$GITHUB_WORKSPACE"/openxla/xla/build_tools/ci/build.py \
      --build="${{ matrix.job_info.name }}_github_actions"
```

The `XLA_LINUX_X86_GPU_ROCM_GITHUB_ACTIONS` build is defined in
`build_tools/ci/build.py` as:

```python
Build(
    type_=BuildType.XLA_LINUX_X86_GPU_ROCM_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "rbe_linux_cpu", "rocm_clang_hermetic"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
    build_tag_filters=rocm_tag_filter,
    test_tag_filters=rocm_tag_filter,
    options={**_DEFAULT_BAZEL_OPTIONS, "//xla/tsl:ci_build": True},
    subcommand="build",
)
```

**Configuration details:**

-   `subcommand="build"` — no tests are run.
-   `configs=("warnings", "rbe_linux_cpu", "rocm_clang_hermetic")` — the build
    uses the CPU RBE pool (no GPU workers needed) and the `rocm_clang_hermetic`
    config, which pulls in a hermetic ROCm + Clang toolchain rather than
    depending on `/opt/rocm` being present.
-   `rocm_tag_filter` mirrors the runtime filter set used by `rocm_ci.yml`
    *plus* an explicit `"gpu"` include, since `build.py` does not have the
    `rocm_tag_filters.sh` script in scope.

`build.py` is just a config-as-data factory: it emits three Bazel commands per
build — dry-run (`--nobuild`) retry, real build, and `bazel analyze-profile
profile.json.gz`:

```python
def commands(self) -> List[List[str]]:
    cmds = []
    cmds.extend(self.extra_setup_commands)
    if not (macos_build or windows_build):
      cmds.append(retry(self.bazel_command(subcommand="build",
                                           extra_options=("--nobuild",))))
    cmds.append(self.bazel_command(subcommand=self.subcommand))
    cmds.append(["bazel", "analyze-profile", "profile.json.gz"])
    return cmds
```

The dry-run-then-build pattern allows retrying transient package-fetch failures
without rebuilding.

This phase ensures compilation regressions are caught even if GPU runners are
unavailable.

## Sanitizer support

`build_tools/rocm/rocm_xla.bazelrc` also declares ASan and TSan configs:

```
build:tsan --strip=never
build:tsan --copt -fsanitize=thread
build:tsan --copt -g
build:tsan --copt -fno-omit-frame-pointer
build:tsan --linkopt -fsanitize=thread
build:tsan --linkopt -g
build:tsan --//build_tools/rocm:sanitizer=tsan
build:tsan --test_env=TSAN_OPTIONS=suppressions=build_tools/rocm/tsan_ignore_list.txt:…
build:tsan --run_under=//build_tools/rocm:sanitizer_wrapper

build:asan --test_env=ASAN_OPTIONS=suppressions=build_tools/rocm/asan_ignore_list.txt:use_sigaltstack=0
build:asan --test_env=LSAN_OPTIONS=suppressions=build_tools/rocm/lsan_ignore_list.txt:use_sigaltstack=0
build:asan --//build_tools/rocm:sanitizer=asan
build:asan --run_under=//build_tools/rocm:sanitizer_wrapper
```

The `//build_tools/rocm:sanitizer` string flag is consumed by `select({"asan":
[...], "tsan": [...]})` in the sanitizer_wrapper filegroup to bring the
appropriate suppression file into the test action's runfiles.

These configs are not engaged by `rocm_ci.yml` today; they are available for
ad-hoc invocations and for downstream pipelines.

## Execution flow

For a typical PR landing on `openxla/xla`:

```
PR commit
   │
   ├─► .github/workflows/ci.yml ───────────────► matrix of CPU/GPU/ROCm builds
   │       │
   │       └─► XLA Linux x86 GPU ROCm
   │             └─► build_tools/ci/build.py
   │                   └─► bazel build (nobuild → real)
   │                         configs: warnings + rbe_linux_cpu +
   │                                  rocm_clang_hermetic
   │                         tag filter: rocm_tag_filter (compile-only)
   │
   └─► .github/workflows/rocm_ci.yml ──────────► AMD GPU runner
           │
           ├─► rocm-config: pin Docker image digest
           │
           ├─► jax (1 GPU)
           │     └─► jax/ci/run_bazel_test_rocm_rbe.sh
           │           └─► bazel test --config=single_gpu …
           │                 override_module=xla=<this PR>
           │                 EngFlow RBE
           │
           └─► xla (1 GPU)
                 ├─► wget execute_ci_build_upstream.sh from ROCm/xla
                 ├─► bazel test --config=rocm_ci --config=rocm_rbe
                 │                --config=ci_single_gpu
                 │       (dynamic scheduling, parallel_gpu_execute lock,
                 │        flock per CUDA_VISIBLE_DEVICES, 3 retries)
                 └─► bazel test --config=rocm_ci --config=rocm_rbe
                                --config=ci_rocm_cpu
                         (200-way local, sanitizer_wrapper, no GPU
                          required but exercises GPU-tagged tests on CPU)
```

If anything fails:

-   An EngFlow link (`https://wardite.cluster.engflow.com/invocation/<UUID>`) is
    printed by Bazel; it is the canonical place to look at action logs, cache
    hits, and per-test stdout.
-   Profile data goes to `/tf/pkg/profile.json.gz` (this path is hardcoded in
    the AMD scripts; the container `--tmpfs` plus the `/tf/pkg` directory is
    where Bazel writes timing data for `bazel analyze-profile`).
-   `lscpu` and `rocminfo` output are at the top of each job log, which is often
    enough to triage "is the GPU actually visible to the container".

## Maintenance and updates

Common edits and where to make them:

| Goal                     | File                                              |
| ------------------------ | ------------------------------------------------- |
| Bump ROCm version        | `.github/workflows/rocm_ci.yml`, `rocm-config`    |
:                          : step — update the sha256 digest                   :
| Add/remove a base tag    | `build_tools/rocm/rocm_tag_filters.sh`            |
: filter                   :                                                   :
| Re-tune the compile-only | `build_tools/ci/build.py`,                        |
: ROCm phase               : `XLA_LINUX_X86_GPU_ROCM_GITHUB_ACTIONS` block +   :
:                          : `rocm_tag_filter` tuple                           :
| Add/remove tests from    | `build_tools/rocm/rocm_xla.bazelrc`,              |
: single-GPU sweep         : `test\:xla_sgpu`                                  :
| Promote a test to        | `build_tools/rocm/rocm_xla.bazelrc`,              |
: multi-GPU                : `test\:xla_mgpu`, and ensure it has the           :
:                          : `multi_gpu` tag                                   :
| Change EngFlow endpoint  | `build_tools/rocm/rocm_xla.bazelrc`,              |
: or auth path             : `build\:rocm_rbe` lines + the `volumes\:` in      :
:                          : `rocm_ci.yml`                                     :
| Tighten per-test         | `build_tools/rocm/parallel_gpu_execute.sh`        |
: isolation                :                                                   :
| Adjust JAX-side build    | The `jax` step in `rocm_ci.yml`, calling JAX's    |
: flags                    : `ci/run_bazel_test_rocm_rbe.sh`                   :
| Adjust upstream driver   | This lives in `ROCm/xla` on the `rocm-dev-infra`  |
: script                   : branch                                            :
:                          : (`build_tools/rocm/execute_ci_build_upstream.sh`) :
:                          : — not in this repo                                :

A change to `build.py` should be paired with
`build_tools/ci/golden_commands.txt` regeneration (the readme explains how:
`PYTHONDONTWRITEBYTECODE=1 python3 build.py --dump_commands >
golden_commands.txt`). Goldens are documentation, not enforcement — CI does not
diff against them — but reviewers do read them to see the materialised command
lines.

## References

-   Workflows: `.github/workflows/rocm_ci.yml`, `.github/workflows/ci.yml`
-   ROCm-specific scripts and bazelrc: `build_tools/rocm/`
-   Shared CI driver: `build_tools/ci/build.py`
-   Tag-filter helper: `build_tools/rocm/rocm_tag_filters.sh`
-   GPU lock helpers: `build_tools/rocm/parallel_gpu_execute.sh`,
    `build_tools/ci/parallel_gpu_execute.sh`
-   Sanitizer ignore lists: `build_tools/rocm/{asan,lsan,tsan}_ignore_list.txt`
-   Upstream-fork driver script (external): `ROCm/xla` on branch
    `rocm-dev-infra`, file `build_tools/rocm/execute_ci_build_upstream.sh`
-   EngFlow cluster (RBE + BES): `grpcs://wardite.cluster.engflow.com`
