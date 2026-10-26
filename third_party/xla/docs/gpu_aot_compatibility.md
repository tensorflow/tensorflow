# XLA:GPU AOT compatibility guide

This guide defines the compatibility contract between the XLA:GPU compiler that
produces a serialized executable and the runtime that loads it, and tells you
how to land a change without breaking that contract. For what AOT compilation is
and how an artifact is structured and loaded, see
[XLA:GPU ahead-of-time compilation](gpu_aot.md).

The contract applies exclusively to the XLA:GPU backend
([`GpuExecutableProto`](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_executable.proto)).
CPU and TPU AOT executables use different serialization mechanisms and are out
of scope.

## Compatibility windows {#compatibility}

XLA:GPU provides two compatibility windows:

*   **Backward compatibility (6 months):** A runtime containing your change must
    load and execute, with identical semantics, every artifact emitted by
    compilers over the preceding 6 months. This allows loading models compiled
    some time ago on an updated runtime.
*   **Forward compatibility (2 weeks):** A compiler containing your change must
    not emit any serialization construct that a runtime up to 2 weeks older
    cannot deserialize or would misinterpret. This allows loading models
    compiled with the latest compiler on older, long-running inference servers
    during rolling deployments.

Every proto message and field transitively reachable from `GpuExecutableProto`
is part of this contract.

## Change triage: is my change safe? {#change-triage}

Follow this decision sequence to determine whether your change requires a staged
rollout:

1.  **Does your change touch any of the following?**

    *   a proto reachable from `GpuExecutableProto`
    *   `ToProto`/`FromProto`
    *   HLO opcodes or HLO proto that can end up in a GPU executable
    *   names or semantics of registered kernel symbols, custom calls, or FFI
        handlers
    *   load-time thunk passes or flags the runtime reads
    *   execution semantics of existing thunks

    **No:** The change is compiler-internal (e.g. optimizations, fusion logic,
    codegen rewrites) or a purely internal runtime refactor with unchanged
    behavior for existing thunks. It is safe to land directly.

    **Yes:** Proceed to step 2.

2.  **Does the change alter how the runtime interprets existing artifacts
    (execution of existing thunks, load-time passes, defaults of runtime-read
    flags)?**

    **Yes:** Allowed only if correct for every artifact emitted in the last 6
    months; if the semantic meaning changes, introduce a new field or flag
    instead.

    **No:** Proceed to step 3.

3.  **Does the change remove an existing thunk kind, proto field, or
    deserialization fallback branch?**

    **Yes:** Violates **backward compatibility** unless compiler emission ceased
    at least 6 months ago. Keep the deserialization fallback until the 6-month
    window has elapsed, and mark deleted proto tags and names as `reserved`.

    **No:** Proceed to step 4.

4.  **Does the change emit a new thunk kind, new proto field required for
    correctness, or new HLO opcode into the serialized artifact?**

    **Yes:** Violates **forward compatibility** if released immediately. Follow
    the [staged rollout](#staged-rollout).

    **No:** Proceed to step 5.

5.  **Does the change rename or delete a registered kernel symbol, custom call,
    or FFI handler name?**

    **Yes:** Violates **backward compatibility**. Retain backwards-compatible
    registrations under legacy names for at least 6 months.

    **No:** Proceed to step 6.

6.  **Does the change add an optional proto field that older runtimes can safely
    ignore without correctness regressions?**

    **Yes:** Safe to land, provided unset proto fields preserve legacy runtime
    semantics.

    **No:** Redesign the change to preserve backward and forward compatibility.

## Safe vs. breaking changes

<!-- mdformat off(reason: wide table) -->
| Change | Verdict | Compatibility impact | Safe procedure |
| :--- | :--- | :--- | :--- |
| Compiler pass, fusion, or codegen optimization (same schema) | Safe | None | Land directly. |
| In-memory runtime refactor with unchanged behavior for existing thunks (caches, buffer ownership) | Safe | None | Land directly with unit tests. |
| `ToProto`/`FromProto` refactor with identical serialized bytes | Safe | None | Land directly; include round-trip tests. |
| Add new proto field whose unset default preserves legacy behavior | Safe | None | Ensure `FromProto` handles unset state cleanly. |
| Add new proto field required for runtime correctness | Breaking | Forward compatibility | Staged rollout (runtime support first). |
| Emit new thunk kind, oneof variant, or enum value | Breaking | Forward compatibility | Staged rollout (add runtime thunk first). |
| Emit new HLO opcode into serialized module | Breaking | Forward compatibility | Staged rollout; gate emission until runtime window passes. |
| Remove thunk kind, proto field, or `FromProto` branch | Breaking | Backward compatibility | Wait 6 months after emission ceased; reserve tags. |
| Change existing field number or proto data type | Breaking | Forward & backward compatibility | Never modify existing field tags; allocate a new tag. |
| Change semantic meaning of an existing field | Breaking | Forward & backward compatibility | Introduce a new field representing the new behavior. |
| Rename or remove registered kernel symbol or FFI handler | Breaking | Backward compatibility | Retain alias registration under legacy name for 6 months. |
| Alter load-time thunk pass or toggle runtime-read flag default | Depends | Backward compatibility | Allowed only if correct for every artifact emitted in the last 6 months. |
<!-- mdformat on -->

## Staged rollout for breaking changes {#staged-rollout}

When introducing a new serialization construct, land it in three phases so that
both compatibility windows are satisfied:

1.  **Runtime support.** Add deserialization and execution logic to the runtime.
    If compiler emission is included in the same PR, keep emission disabled by
    default behind an experimental flag. Then wait at least **2 weeks** from the
    merge of this change so the updated runtime propagates across deployment
    environments.
2.  **Compiler emission.** Enable compiler emission by default (or flip the
    default value of the feature flag). Then keep the legacy deserialization
    fallback in the runtime for at least **6 months**, counted from this phase
    (when the compiler stopped emitting the legacy form), not from phase 1.
3.  **Cleanup.** Delete the legacy deserialization path and mark the deprecated
    proto tag numbers and field names as `reserved`.

Phases 1 and 2 can be two pull requests (the second sent once the first is 2
weeks old) or a single PR where emission is disabled by default behind a flag
and flipped in a follow-up.

### Removal comments and examples

Every compatibility fallback in the runtime should document its expiration date:

```cpp
// Backward-compatibility fallback for legacy AOT-compiled kernels without
// an explicit CollectiveKernelSpec.
// Can be removed in <month year> (6 months backward compatibility window).
```

Public OpenXLA rollout examples:

*   [PR #49046](https://github.com/openxla/xla/pull/49046): Introduced dedicated
    stream assignment in the runtime first, keeping emission disabled by default
    during the rollout window.
*   [PR #46865](https://github.com/openxla/xla/pull/46865): Added runtime thunk
    support for `CollectiveReduce`, with compiler emission guarded behind an
    experimental flag.

## Authoring runtime code for deserialized executables

When implementing runtime thunks and deserialization routines:

*   **No compiler-only types:** Runtime thunks must not depend on compiler data
    structures such as `HloInstruction*` or `BufferAssignment`.
*   **Validate deserialized bounds:** Always validate buffer allocation indices,
    slice offsets, and stream IDs against allocation bounds before indexing.
*   **Do not duplicate payloads:** Do not store the same large payload (for
    example constants) more than once in the artifact; it multiplies load-time
    memory.
*   **Guard initialization allocations:** Serving runtimes operate under strict
    memory limits. Avoid unconditional heap allocations during `FromProto` or
    thunk initialization.

## Testing

All serialization changes must be covered by automated tests:

*   **Round-trip unit tests:** Verify serialization and deserialization fidelity
    via `ToProto` and `FromProto` tests (see
    [`ThunkProtoDeserializationTest`](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/thunk_proto_deserialization_test.cc)).
*   **Golden AOT tests:** End-to-end compatibility tests live under
    [`xla/tests/aot_compatibility/gpu`](https://github.com/openxla/xla/blob/main/xla/tests/aot_compatibility/gpu)
    (such as `collective_ops_aot_test.cc`), running against serialized golden
    executables in `executables/<target>/v<N>/`.
*   **Testing all historical versions:** By default, golden tests validate
    against the boundary versions (oldest and second-newest). Set
    `XLA_AOT_TEST_ALL_VERSIONS=1` in your test environment to execute against
    all historical golden versions.
