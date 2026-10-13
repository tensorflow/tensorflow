---
name: gpu-aot-compatibility-review
description: >-
  Verifies 6-month backward and 2-week forward compatibility for XLA GPU
  AOT-compiled executables (GpuExecutableProto and ThunkProto) and produces a
  structured AOT compatibility review summary. Use when authoring or reviewing
  any change touching XLA GPU runtime files (xla/backends/gpu/runtime/...,
  including thunks, collective/kernel helpers, and constants), GPU collective
  backends (xla/backends/gpu/collectives/...), GpuExecutable and runtime
  helpers (xla/service/gpu/gpu_executable*, stream_executor_util*,
  dense_data_intermediate*), DebugOptions defaults (xla/debug_options_flags.cc),
  or transitive proto dependencies of GpuExecutableProto, even when a change
  does not directly modify proto serialization. Don't use for CPU/TPU-only
  changes or HLO passes outside the GPU runtime/executable scope that do not
  affect GPU AOT artifacts.
---

# XLA GPU Runtime & GpuExecutable AOT Compatibility Review

Use this skill when reviewing a change or validating local modifications that
touch any file under `xla/backends/gpu/runtime/...`,
`xla/backends/gpu/collectives/...`, `GpuExecutable`
(`xla/service/gpu/gpu_executable*`), `xla/debug_options_flags.cc`, or related
GPU runtime serialization and execution helpers.

## Context & Compatibility Guarantees

Ahead-Of-Time (AOT) compiled GPU executables (serialized as
`xla::gpu::GpuExecutableProto` in `xla/service/gpu/gpu_executable.proto`) exist
in persistent storage and production services where compiler and runtime
binaries are deployed independently:

1. **Backward Compatibility (Newer Runtime + Older AOT Artifact - 6-Month
   Window)**:
   We guarantee **6 months (180 days)** of backward compatibility. Newer runtime
   versions must load and execute AOT artifacts compiled and stored by compilers
   up to **6 months older**.
2. **Forward Compatibility (Older Runtime + Newer AOT Artifact - 2-Week
   Window)**:
   We guarantee **2 weeks (14 days)** of forward compatibility. Older runtime
   versions must load and execute AOT artifacts produced by a compiler that is
   up to **2 weeks newer**. Supporting the old format in the new runtime is
   **not sufficient** if the compiler immediately starts emitting a new proto
   feature that older runtimes cannot handle.

The contributor-facing version of these rules (change triage, safe vs. breaking
changes, staged rollout) is the GPU AOT compatibility guide at
`docs/gpu_aot_compatibility.md` (https://openxla.org/xla/gpu_aot_compatibility).
Keep this skill consistent with it, and link to it in every blocking finding.

## Step 1: Scope the Diff & Identify Changes Affecting `GpuExecutableProto`

1. **Obtain only the target change's diff**:
   - Inspect only the delta introduced by the change or commit under review
     (e.g., `git diff` or `git diff main...HEAD`), avoiding changes from parent
     or unrelated commits in a stack.

2. **Check that `DebugOptions` fields read after compilation are listed**:
   `kRuntimeFields` in `xla/debug_options_runtime_fields.cc` lists the
   `DebugOptions` fields that can change how a loaded executable behaves. AOT
   golden verification relies on it, and nothing else enforces it. If the diff
   adds a `DebugOptions` field to `xla/xla.proto`, or adds a read of a
   `DebugOptions` field (`.xla_*()`) on a load or execution path of any backend
   (e.g. `GpuExecutable::FromProto`, `RunThunkPasses`, `CpuExecutable` loading,
   thunk execution) or via `GetDebugOptionsFromFlags()`, apply the criteria in
   `xla/debug_options_runtime_fields.h`. If the field qualifies and is missing,
   raise an advisory (non-blocking) finding asking the author to add
   `DebugOptions::k<FieldName>FieldNumber,  // <number>` to `kRuntimeFields` in
   field-number order, and mention the result in the Step 4 summary.

3. **Trace transitive dependencies of `GpuExecutableProto`**:
   The file paths that trigger this skill are entry points, **not** an
   exhaustive list of protos in an AOT artifact. `GpuExecutableProto`
   transitively embeds many proto messages across XLA and StreamExecutor (e.g.,
   `ThunkProto` and all specific thunk sub-protos in
   `xla/backends/gpu/runtime/thunk.proto`, `CustomKernelProto`,
   `KernelSpecTableProto`, `HloModuleProtoWithConfig`, `BufferAllocationProto` /
   `BufferAssignmentProto`, `DeviceDescriptionProto`,
   `ExecutableAbiVersionProto`, `ProgramShapeProto`,
   `DenseDataIntermediateProto`, collective/kernel configs).
   Determine:
   - Which `.proto` changes in the diff are direct or transitive dependencies of
     `GpuExecutableProto`.
   - Which C++ serialization (`ToProto`), deserialization (`FromProto`),
     split-proto writer (`split_gpu_executable_writer`), compiler
     thunk-emission, or runtime execution paths affect serialized AOT artifacts.

4. **Handle in-scope changes that do not affect serialized AOT artifacts**:
   - **Never skip Steps 2 and 3 when runtime or collective execution changes**:
     Do **not** treat a change as a trivial AOT no-op if it modifies thunk
     initialization or execution, collective communicator or symmetric-memory /
     window setup (`xla/backends/gpu/collectives/...`), stream or kernel launch
     behavior, buffer allocation, load-time thunk passes (`RunThunkPasses`),
     `DebugOptions` defaults (`xla/debug_options_flags.cc`), or `FromProto` /
     `ToProto` option propagation - even when the commit itself touches zero
     `.proto`, `ToProto`, or `FromProto` lines.
   - **Trace stacked-change parameters and flags back to `FromProto` and
     `DebugOptions` defaults**: Authors frequently split changes across stacked
     commits, landing a new `DebugOptions` flag or plumbed parameter as a no-op
     in a parent change and switching runtime behavior on that parameter in a
     small child change. Whenever a change alters runtime or collective behavior
     based on a parameter, struct field, or flag, trace that value upward
     through callers in the active workspace all the way to
     `GpuExecutable::FromProto`, `GpuExecutable::ExecuteThunks`,
     `Thunk::FromProto`, `xla/xla.proto`, and `xla/debug_options_flags.cc`, and
     then run **Steps 2 and 3**.
   - **Trivial fast path**: Only when none of the modified hunks touch `.proto`
     definitions, `ToProto` / `FromProto` serialization, `DebugOptions`
     defaults, or runtime/collective execution behavior for existing serialized
     AOT artifacts (for example, changes strictly limited to compile-time HLO
     eligibility checks, size thresholds, logging, or comments) is the change
     trivially AOT-compatible. Proceed directly to **Step 4** and include a
     brief AOT compatibility summary stating why the change does not impact
     `GpuExecutableProto` or `ThunkProto` compatibility.

## Step 2: Check Backward Compatibility (6-Month Window)

Static `.proto` linters only catch surface schema violations in `.proto` files
and cannot detect semantic breaks in C++ `FromProto`/`ToProto` logic. Check both
`.proto` wire rules and C++ deserialization semantics:

- **Protobuf wire compatibility**: Flag removed or renamed in-use fields without
  `reserved` numbers and names, changed field numbers, incompatible field type
  changes, or altered default semantics. Moving an existing field into a `oneof`
  with the same tag number is wire-compatible only if `FromProto` still handles
  that field when the new `oneof` branches are unset.
- **Graceful handling of missing fields and absent load-time overrides**:
  - Production AOT callers frequently load executables via
    `LoadSerializedExecutable` with `options = std::nullopt` (no load-time
    `CompileOptions` / `DebugOptions` overrides). Consequently,
    `GpuExecutable::FromProto` and `GpuExecutable::ExecuteThunks` run with the
    `HloModuleConfig` and `ThunkProto` stored inside the up-to-6-month-older
    `GpuExecutableProto`, where any `DebugOptions` or `ThunkProto` field added
    within the last 6 months is unset (`false`, `0`, or empty).
  - Verify that `FromProto` and thunk/collective execution paths gracefully
    handle older serialized protos with unset fields. Never assume that a
    `false` default on a newly added or recently plumbed flag matches historical
    behavior: inspect the pre-change code directly to check what the runtime did
    unconditionally before the flag existed (e.g., communicator or memory
    registration flags, buffer alignment guarantees, or fallback kernel
    selection), and flag any change where an unset (`false`) field switches
    legacy AOT artifacts onto a different or weaker runtime mode.
- **Preserved execution semantics across all serialized thunk modes**:
  - Verify that changes to thunk initialization/execution, collective
    communicator or symmetric-memory setup (`xla/backends/gpu/collectives/...`),
    buffer allocation (`gpu_executable_buffer_allocator`), run options
    (`gpu_executable_run_options`), or load-time passes preserve required
    runtime preconditions for **all** serialized thunk variants and modes in
    `xla/backends/gpu/runtime/...` and `xla/backends/gpu/runtime/thunk.proto`
    (search `thunk.proto` and thunk implementations for optional collective
    buffer modes, non-default execution stream IDs, command-buffer thunks, and
    custom kernels that rely on that setup).
  - Even if standard HLO lowering does not emit a given thunk mode by default,
    AOT artifacts compiled within the last 6 months (including models compiled
    with experimental or workload-specific flags supported at compile time) may
    contain it and must continue to initialize and execute validly when loaded
    without new flags.
- **`FromProto` / `ToProto` and `HloModuleConfig` option propagation
  correctness**:
  - `GpuExecutable::ExecuteThunks` reads runtime flags from
    `module_config().debug_options()` on the deserialized `debug_module`
    (populated in `GpuExecutable::FromProto` from
    `proto.hlo_module_with_config()` and any load-time `DebugOptions`
    overrides).
  - Verify that `GpuExecutable::FromProto` and `Thunk::FromProto` / `ToProto`
    propagate fields and overrides with the intended value - never inverted (for
    example, calling `set_foo(false)` inside `if (debug_options.foo())`) and
    never clobbering values serialized in `proto.hlo_module_with_config()` when
    no load-time override was supplied.
  - Treat any bug in `FromProto`, `ToProto`, or runtime `DebugOptions` /
    `HloModuleConfig` propagation as a **blocking AOT defect** even if the
    change description states that the new flag or parameter is not yet consumed
    in the current change, because follow-up changes in a stack will rely on
    that plumbing without touching `FromProto` again.
- **Formal verification when removing legacy / backward-compatibility paths**:
  If the change deletes a fallback branch in `FromProto`/runtime execution or
  removes handling for a legacy proto field, follow these steps to verify the
  6-month window:
  1. **Find the commit that disabled default compiler emission**: Check the
     change description or run `git log -S "<field_or_flag>" -p xla/...` (on the
     `ToProto` serializer or `xla/debug_options_flags.cc`) to identify the
     commit where the compiler stopped emitting the legacy representation by
     default.
  2. **Check the commit date**: Run `git show -s --format=%ci <commit>` to
     inspect the commit timestamp.
  3. **Enforce the 180-day rule**:
     - If the compiler stopped emitting the legacy path **less than 6 months
       (180 days) ago**, flag the removal as a backward-compatibility violation.
     - If repository history shows the compiler still emits the legacy path
       under some configurations or the cutoff commit cannot be determined, ask
       the author to cite the commit that turned off emission > 6 months ago.

## Step 3: Check Forward Compatibility (2-Week Window)

When a change adds a new feature, field, or thunk variant to the serialized AOT
format, verify that AOT artifacts produced after this change will preserve
correct behavior when loaded into an older runtime (up to **2 weeks / 14 days
older**):

- **Acceptable mechanisms**:
  - Any mechanism that preserves runtime behavior when loaded into an
    up-to-2-week-older runtime is valid:
    1. **Wire-compatible fallback / dual-writing**: Populating the legacy field
       by default (or dual-writing both legacy and new fields) so older runtimes
       ignore the unknown new field number and execute the legacy field
       normally.
    2. **Purely informational / optional additive fields**: Adding fields that
       older runtimes can safely ignore without affecting execution correctness.
    3. **Two-phase rollout with a default-off flag**: When the new
       representation replaces a legacy field or requires new runtime execution
       logic:
       - **Phase 1 (Runtime First)**: Land `FromProto` deserialization and
         runtime execution support first while keeping compiler emission off by
         default (most commonly via a flag in `DebugOptions` in `xla/xla.proto`
         defaulted to `false` in `xla/debug_options_flags.cc`, enabled
         explicitly in tests).
       - **Phase 2 (Enable in Compiler After >= 2 Weeks)**: Flip the default in
         `xla/debug_options_flags.cc` (or enable default emission in `ToProto`)
         in a follow-up change at least **14 days** after Phase 1 landed.
- **Formal verification when enabling emission by default**:
  - Flag any change that causes `ToProto` / compiler emission to immediately
    stop populating a field required by up-to-2-week-older runtimes (or emit a
    new thunk/field variant that older runtimes reject) without a default-off
    flag or legacy fallback.
  - If a change flips an AOT-related `DebugOptions` flag to `true` by default:
    1. Check `git log -p` on the corresponding `FromProto` / runtime file to
       find the commit that landed runtime deserialization/execution support,
       and run `git show -s --format=%ci <commit>` to check its timestamp.
    2. Flag the change if the runtime support landed **less than 14 days ago**.

## Step 4: Report Findings & AOT Compatibility Summary

- **AOT compatibility checks pass**:
  - If both backward compatibility (Step 2, including `FromProto` / `ToProto`
    and runtime option propagation correctness) and forward compatibility
    (Step 3) checks pass (or if the change does not affect serialized AOT
    artifacts per Step 1.4), and no breaking changes to `GpuExecutableProto` or
    runtime execution are detected, provide a summary under the heading
    `### XLA GPU Runtime & AOT Compatibility Review`:
    - State which modified files, protos, serializers/deserializers, thunks, or
      runtime helpers were analyzed (and note the Step 1.2 `kRuntimeFields`
      check result when `DebugOptions` fields were added or read).
    - If serialization or thunk execution was touched, explain why 6-month
      backward compatibility (Step 2) and 2-week forward compatibility (Step 3)
      are preserved (e.g., older serialized protos missing new fields are
      handled gracefully with safe defaults, new fields are optional/additive,
      wire-compatible fallback/dual-writing is used, or new compiler emission is
      gated on a default-off flag).
    - If no serialization or AOT execution paths were affected (Step 1.4), state
      that explicitly.
- **Blocking AOT compatibility issues found (Changes Requested)**:
  - If any backward compatibility (6-month window) violation, forward
    compatibility (2-week window) violation, or bug in `FromProto` / `ToProto`
    serialization or `GpuExecutable::FromProto` runtime option propagation (even
    for a newly added flag or parameter not yet consumed in the current change)
    is detected:
    - Report each compatibility or deserialization defect with its `file:line`
      location, citing the required compatibility window where applicable and
      pointing to acceptable mitigation mechanisms from Steps 2 or 3 (e.g.,
      dual-writing, gating compiler emission on a default-off flag, maintaining
      legacy `FromProto` / runtime fallback paths, or fixing `FromProto` option
      propagation).
    - End the finding with a link to the contributor-facing guide so the author
      can self-serve the fix (change triage, safe vs. breaking changes, staged
      rollout): `See the GPU AOT compatibility guide:`
      `https://openxla.org/xla/gpu_aot_compatibility`
      `(source: docs/gpu_aot_compatibility.md).`
