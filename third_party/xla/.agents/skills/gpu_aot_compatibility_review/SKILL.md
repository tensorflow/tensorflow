---
name: gpu-aot-compatibility-review
description: >-
  Verifies 6-month backward and 2-week forward compatibility for XLA GPU
  AOT-compiled executables (GpuExecutableProto and ThunkProto) and produces a
  structured AOT compatibility review summary. Use when authoring or reviewing
  any change touching XLA GPU runtime files (xla/backends/gpu/runtime/...,
  including thunks, collective/kernel helpers, and constants), GpuExecutable
  and runtime helpers (xla/service/gpu/gpu_executable*, stream_executor_util*,
  dense_data_intermediate*), or transitive proto dependencies of
  GpuExecutableProto, even when a change does not directly modify proto
  serialization. Don't use for CPU/TPU-only changes or HLO passes outside the
  GPU runtime/executable scope that do not affect GPU AOT artifacts.
---

# XLA GPU Runtime & GpuExecutable AOT Compatibility Review

Use this skill when reviewing a change or validating local modifications that
touch any file under `xla/backends/gpu/runtime/...`, `GpuExecutable`
(`xla/service/gpu/gpu_executable*`), or related GPU runtime serialization and
execution helpers.

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

2. **Trace transitive dependencies of `GpuExecutableProto`**:
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

3. **Handle in-scope changes that do not affect serialized AOT artifacts**:
   If none of the modified hunks touch `.proto` definitions, `ToProto` /
   `FromProto` serialization, or runtime execution behavior for existing
   serialized AOT artifacts (for example, changes limited to compile-time
   eligibility checks, size thresholds, logging, or comments), the change is
   trivially AOT-compatible. Proceed directly to **Step 4** and include a brief
   AOT compatibility summary stating why the change does not impact
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
- **Graceful handling of missing fields**: Verify that `FromProto` and thunk
  constructors/execution paths gracefully handle older serialized protos (up to
  6 months old) where newly added fields are unset or default-initialized.
- **Preserved execution semantics**: Verify that changes to thunk
  initialization, buffer allocation (`gpu_executable_buffer_allocator`), run
  options (`gpu_executable_run_options`), or execution invariants preserve
  behavior when loading legacy AOT executables.
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
  - If both backward compatibility (Step 2) and forward compatibility (Step 3)
    checks pass (or if the change does not affect serialized AOT artifacts per
    Step 1.3), and no breaking changes to `GpuExecutableProto` or runtime
    execution are detected, provide a summary under the heading
    `### XLA GPU Runtime & AOT Compatibility Review`:
    - State which modified files, protos, serializers/deserializers, thunks, or
      runtime helpers were analyzed.
    - If serialization or thunk execution was touched, explain why 6-month
      backward compatibility (Step 2) and 2-week forward compatibility (Step 3)
      are preserved (e.g., older serialized protos missing new fields are
      handled gracefully with safe defaults, new fields are optional/additive,
      wire-compatible fallback/dual-writing is used, or new compiler emission is
      gated on a default-off flag).
    - If no serialization or AOT execution paths were affected (Step 1.3), state
      that explicitly.
- **Blocking AOT compatibility issues found (Changes Requested)**:
  - If any backward compatibility (6-month window) or forward compatibility
    (2-week window) violation is detected:
    - Report each compatibility hazard with its `file:line` location, citing the
      required compatibility window and pointing to acceptable mitigation
      mechanisms from Steps 2 or 3 (e.g., dual-writing, gating compiler emission
      on a default-off flag, or maintaining legacy `FromProto` fallback paths).
    - End the finding with a link to the contributor-facing guide so the author
      can self-serve the fix (change triage, safe vs. breaking changes, staged
      rollout): `See the GPU AOT compatibility guide:`
      `https://openxla.org/xla/gpu_aot_compatibility`
      `(source: docs/gpu_aot_compatibility.md).`
