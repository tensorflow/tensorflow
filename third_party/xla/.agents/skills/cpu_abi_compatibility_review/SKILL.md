---
name: cpu-abi-compatibility-review
description: >-
  Verifies XLA:CPU compiler and runtime ABI compatibility under
  kHostExecutionAbiVersion and produces a structured ABI compatibility review
  summary. Use when authoring or reviewing any change touching XLA:CPU runtime
  files (xla/backends/cpu/runtime/...), CpuExecutable AOT serialization
  (xla/service/cpu/cpu_executable* and xla/service/cpu/executable.proto), or
  transitive proto dependencies of CompilationResultProto, or when modifying
  kHostExecutionAbiVersion in xla/backends/cpu/runtime/host_execution_abi.h,
  even when a change does not directly alter serialized protos or numerical
  behavior. Don't use for XLA:GPU AOT changes (use gpu-aot-compatibility-review)
  or compiler passes outside the CPU runtime/executable scope that do not affect
  serialized CPU executables or runtime execution.
---

# XLA:CPU Runtime & CpuExecutable ABI Compatibility Review

Use this skill when reviewing a change or validating local modifications that
touch any file under `xla/backends/cpu/runtime/...`, `CpuExecutable` AOT
serialization (`xla/service/cpu/cpu_executable.{h,cc}`,
`xla/service/cpu/executable.proto`), or transitive proto dependencies of
`CompilationResultProto`.

## Context & ABI Version Model

Ahead-Of-Time (AOT) compiled XLA:CPU executables (serialized as
`xla::cpu::CompilationResultProto` in `xla/service/cpu/executable.proto`,
containing `ThunkSequenceProto` / `ThunkProto` from
`xla/backends/cpu/runtime/thunk.proto`) use an **explicit ABI version**
(`xla::cpu::kHostExecutionAbiVersion` in
`xla/backends/cpu/runtime/host_execution_abi.h`):

1. **Compatibility Within an ABI Version**:
   Every compiler build and every runtime build with the same
   `kHostExecutionAbiVersion` ($N$) must be compatible with each other,
   regardless of when either was built. Concretely, any runtime at ABI version
   $N$ must load and correctly execute any AOT artifact produced by any compiler
   at ABI version $N$. This holds in both directions (older runtime with newer
   compiler and newer runtime with older compiler) and there is no time window
   after which the requirement lapses. Consequently, there is **no** "land
   runtime support now and change the compiler output later without an ABI
   bump" option: the compiler output may only change in ways that every runtime
   at version $N$ already handles.
2. **Bitwise / Numerical Stability**:
   Compatibility includes **numerical outputs**. Any change expected to alter
   the numerical results produced for an existing serialized artifact at the
   same ABI version (such as changing accumulation types, reduction order,
   library kernel selection, or NaN/$\pm 0$ handling) breaks the ABI contract.
3. **Strong Preference Against Unnecessary ABI Bumps**:
   Bumping `kHostExecutionAbiVersion` invalidates cached AOT executables. Always
   prioritize:
   - **(1) Making the change compatible within the current ABI version**, or
   - **(2) Conditioning new compiler emission on the next ABI version bump
     ($N + 1$)** so the code lands now without an immediate bump, and
   - **(3) Bumping `kHostExecutionAbiVersion` only when unavoidable** (for
     example, an intentional numerical bugfix for existing serialized thunks).

---

## Step 1: Scope the Diff & Classify Affected ABI Surfaces

1. **Obtain only the target change's diff**:
   - Inspect only the delta introduced by the change or commit under review
     (e.g., `git diff` or `git diff main...HEAD`), avoiding changes from parent
     or unrelated commits in a stack.

2. **Classify each modified hunk across the XLA:CPU ABI surfaces**:

   **A. Serialized AOT Format** - every runtime at ABI version $N$ must
   deserialize and execute artifacts produced by every compiler at $N$.

   - `xla/service/cpu/executable.proto` (`CompilationResultProto`)
   - `xla/service/cpu/cpu_executable.{h,cc}`
   - `xla/backends/cpu/runtime/thunk.proto` (`ThunkProto`, `ThunkSequenceProto`)
   - Transitive protos: `xnn_fusion_options.proto`, `ynn_fusion_options.proto`,
     `buffer_assignment.proto`, `hlo.proto`, `shaped_slice.proto`
   - `thunk_proto_serdes.cc`, `thunk_proto_serdes_utils.cc`,
     `xla/backends/cpu/runtime/thunk_serdes/*`

   **B. Kernel & Comparator C ABI** - compiled machine code (OBJ files / LLVM
   kernels) embedded in the AOT artifact calls across this boundary directly.

   - `xla/backends/cpu/runtime/kernel_c_api.h` (`XLA_CPU_KernelCallFrame`,
     dim/arg structs)
   - `kernel.{h,cc}`, `kernel_thunk.*`
   - Sort comparator function signatures in `sort_thunk.*` / `sort_lib.*`
   - `SymbolProto` symbol resolution in `cpu_executable.*` /
     `function_library.*`

   **C. Thunk Execution Semantics** - behavioral invariants, buffer slice
   indexing, concurrency semantics, and runtime preconditions must hold for all
   artifacts at ABI version $N$.

   - `xla/backends/cpu/runtime/*_thunk.{h,cc}`, `thunk.h`, `thunk_executor.*`
   - `buffer_allocations.*`, `function_library.*`, `work_queue.h`
   - `xfeed_manager.*`, `custom_call_thunk.*`, `collective_thunk.*`

   **D. Numerical Runtime Libraries** - math libraries are linked into the
   runtime binary; changing their numerical output alters results for existing
   AOT artifacts.

   - `dot_lib.*`, `convolution_lib.*`, `sort_lib.*`, `topk_lib.h`,
     `rng_state_lib.*`
   - `fft_thunk.*` (DUCC), `ynnpack/*`, `onednn/*`, Eigen contractions

   **E. Non-ABI / Internal Only** - compatible within the current ABI version.

   - Logging, comments, error message strings, unit tests, build configs
   - Internal C++ API additions (e.g. optional parameters on
     `Thunk::ExecuteParams` defaulted to `nullptr` that do not alter existing
     thunk behavior)
   - Pure refactors / parallelization with identical outputs
   - When all modified hunks fall under Surface E, proceed to **Step 5** and
     include an ABI compatibility summary stating why the change only touches
     Surface E.

   Files outside the core runtime path set (e.g.
   `xla/service/cpu/thunk_emitter.*`, `xla/service/hlo.proto`,
   `xla/service/buffer_assignment.proto`) must also be evaluated when they
   change in the same commit as a runtime/executable file. Compiler-side changes
   such as emitting a different existing thunk are compatible on their own as
   long as every runtime at version $N$ already handles the emitted output.

---

## Step 2: Check Compatibility Within the Current ABI Version

For every hunk touching Surfaces A-D, check that any runtime at the current
`kHostExecutionAbiVersion` still executes any artifact produced at that version,
both before and after this change:

### 1. Protobuf Wire & `ToProto` / `FromProto` Compatibility (Surface A)
- **Wire-breaking schema edits**: Flag removed, renumbered, or retyped in-use
  proto fields (for example, changing a field's message type from
  `BufferAllocationSliceProto` to `ShapedSliceProto` in place), removed `oneof`
  variants, or deleted fields without `reserved` tag numbers and names.
- **New runtime, existing artifacts**: Verify `FromProto` and thunk
  constructors gracefully handle serialized protos produced before this change
  at the same ABI version, where newly added fields are unset or
  default-initialized, preserving prior execution behavior.
- **Existing runtimes, new artifacts**: Flag any change where `ToProto` /
  `ThunkEmitter` / `CpuExecutable` immediately stops populating a field that
  runtimes at ABI version $N$ built before this change require, or
  unconditionally emits a new `ThunkProto` `oneof` kind or field that those
  runtimes will reject or misinterpret.
- **Removing legacy `FromProto` or runtime fallback paths**:
  - Deleting a legacy deserialization or execution fallback without bumping
    `kHostExecutionAbiVersion` is safe **only if no compiler at the current
    `kHostExecutionAbiVersion` ever emitted the legacy representation**.
  - Verify this formally:
    1. Check `git log -p xla/backends/cpu/runtime/host_execution_abi.h` to find
       the commit that bumped `kHostExecutionAbiVersion` to its current value.
    2. Check `git log -p` on the serializer/compiler file (`ToProto` or
       `thunk_emitter.cc`) to check when emission of the legacy representation
       was removed or disabled.
    3. If any compiler at the current `kHostExecutionAbiVersion` could have
       emitted the legacy format, flag the removal as breaking compatibility
       within the current ABI version.

### 2. Compiled Kernel & Comparator C ABI (Surface B)
- Flag any change to struct field layout, sizes, alignment, or calling
  conventions in `XLA_CPU_KernelCallFrame` (`kernel_c_api.h`) or sort comparator
  callback signatures (`less_than` in `sort_thunk.*` / `sort_lib.*`). Embedded
  object code in serialized `CompilationResultProto` artifacts cannot adapt to
  C ABI layout changes without recompilation.
- Flag changes to kernel symbol naming or `SymbolProto` ordering/resolution in
  `CpuExecutable` unless fallback logic covering both old and new artifacts is
  preserved.

### 3. Execution Semantics & Numerical Stability (Surfaces C & D)
- **Numerical invariance**: Flag any change that alters numerical outputs for
  the same serialized thunk (e.g., changing accumulation types such as casting
  `s8`/`s16` operands before contraction in `dot_lib.h`, changing reduction
  tree order, changing floating-point rounding/fast-math flags, or altering
  NaN/signed-zero ordering in `sort_lib` / `topk_lib`).
- **Compatible performance/parallelization changes**: Parallelizing independent
  slices across `intra_op_threadpool` (such as parallelizing independent 1-D
  sorts over non-sorting dimensions in `SortThunk`) or refactoring internal
  data structures without changing serialized protos, C ABI, or per-element
  numerical results is **ABI compatible**.

### 4. Rollback & Roll-Forward Rule for `kHostExecutionAbiVersion`
- If a change rolls back the **most recent** commit that bumped
  `kHostExecutionAbiVersion` (and no subsequent ABI bump has occurred), it is
  acceptable for `kHostExecutionAbiVersion` to return to the previous version
  number.
- However, rolling back an **older** ABI bump out of order, or **rolling
  forward** a previously rolled-back ABI change, must **increment**
  `kHostExecutionAbiVersion` to a brand-new version number (never reuse a
  version number that may already be embedded in serialized executables with
  different semantics).

---

## Step 3: Recommend Compatible Implementation Patterns (Preferred Over Bumping)

When a change breaks compatibility within the current ABI version, **focus your
review on concrete, actionable code suggestions** to make the change compatible
without an immediate `kHostExecutionAbiVersion` bump:

### Pattern 1: Additive Proto Fields with Dual-Writing & Legacy Fallback
When evolving a proto message (e.g., replacing a field type or refactoring thunk
metadata):
- Keep the existing proto field number and type intact (do not change or remove
  it in place).
- Add the new field with a new tag number.
- At the current ABI version, continue populating the legacy field (or
  dual-write both the legacy and new fields if older runtimes safely ignore the
  unknown new field) and support both in `FromProto`.

### Pattern 2: Purely Informational / Optional Runtime Additions
- New proto fields used only for logging, profiling, or diagnostics that older
  runtimes safely ignore and that default to no-op in `FromProto` when reading
  older artifacts are compatible immediately.
- Internal C++ runtime struct additions (such as adding an optional pointer
  defaulted to `nullptr` on `Thunk::ExecuteParams` in `thunk.h`) that do not
  affect `ThunkProto` serialization, `kernel_c_api.h`, or existing thunk
  execution behavior are compatible immediately.

### Pattern 3: Conditioning a New Feature on the Next ABI Version Bump (`kHostExecutionAbiVersion`)
When a new feature, thunk variant, or incompatible serialization format cannot
be understood by older runtimes at the current ABI version (let $N$ be the
current value of `kHostExecutionAbiVersion` in
`xla/backends/cpu/runtime/host_execution_abi.h`, e.g., `2`), recommend landing
the code now and conditioning compiler emission on the **next** ABI version
($N + 1$):

1. **Define the ABI version threshold in XLA:CPU** (depend on
   `//xla/backends/cpu/runtime:host_execution_abi`):

   ```cpp
   #include "xla/backends/cpu/runtime/host_execution_abi.h"

   // Snapshot: enable <feature> once kHostExecutionAbiVersion exceeds 2.
   inline constexpr int kLastAbiBeforeFoo = 2;
   ```

2. **Land runtime (`FromProto` / `Execute`) support unconditionally**:
   Support deserializing and executing **both** the new representation and the
   legacy fallback unconditionally in `FromProto` and the thunk's `Execute`
   method. This ensures every runtime built from this commit onward (including
   all future ABI version $N+1$ runtimes) already supports the new format.
   *(Note: If a change is purely an in-place runtime numerical/algorithm change
   with no distinguishing proto field, either add a proto field/enum to select
   the behavior or gate the runtime branch on
   `xla::cpu::kHostExecutionAbiVersion > kLastAbiBeforeFoo`.)*
3. **Gate compiler / serializer (`ToProto` / `ThunkEmitter`) emission on the
   next ABI version, with a parameter or flag for unit tests**:
   Gate emitting the new representation on `abi_version > kLastAbiBeforeFoo`
   (where `abi_version` defaults to `xla::cpu::kHostExecutionAbiVersion`, or can
   be overridden via a `DebugOptions` flag / constructor parameter):

   ```cpp
   if (abi_version > kLastAbiBeforeFoo) {
     // Emit new proto format / thunk variant (enabled once ABI >= 3).
   } else {
     // Emit legacy format while kHostExecutionAbiVersion == 2.
   }
   ```

   In unit tests, pass `abi_version = kLastAbiBeforeFoo + 1` (or enable the test
   flag) so the new emission and deserialization path is tested end-to-end in CI
   immediately while `kHostExecutionAbiVersion` is still $N$.

---

## Step 4: Verify `host_execution_abi.h` Modifications

1. **When `host_execution_abi.h` is NOT modified**:
   - If the change is **not** compatible within the current ABI version, flag
     the compatibility break, recommend the appropriate compatible pattern from
     Step 3, and note that if an immediate breaking change is truly unavoidable,
     `kHostExecutionAbiVersion` in
     `xla/backends/cpu/runtime/host_execution_abi.h` must be bumped.

2. **When `host_execution_abi.h` IS modified (`kHostExecutionAbiVersion`
   bump)**:
   - **Check if the bump is avoidable**: First evaluate whether the change can
     be made compatible or gated on the next ABI version bump using Step 3. If
     it is an intentional numerical bugfix for existing thunks (such as fixing
     overflow in `DotThunk`), confirm that the ABI version bump is justified and
     required.
   - **Verify version increment**: Check that `kHostExecutionAbiVersion` is
     incremented by `+1` (or follows the rollback rule in Step 2.4).

---

## Step 5: Report ABI Compatibility Summary & Findings

- **ABI compatibility checks pass**:
  - Provide a summary under the heading
    `### XLA:CPU Runtime & ABI Compatibility Review`:
    - State which modified files and ABI surfaces (Surfaces A-E from Step 1)
      were analyzed.
    - Explain why compatibility under `kHostExecutionAbiVersion` is preserved
      (e.g., additive proto fields with fallback, identical per-element
      numerical outputs, feature conditioned on
      `abi_version > kLastAbiBefore...`, or Surface E internal-only changes),
      or - when `kHostExecutionAbiVersion` is bumped - explain why the bump is
      justified.
- **Blocking ABI compatibility issues found (Changes Requested)**:
  - Report each compatibility hazard with its `file:line` location and recommend
    concrete compatible implementation patterns from Step 3 (e.g., dual-writing
    proto fields or conditioning compiler emission on the next
    `kHostExecutionAbiVersion` bump).
