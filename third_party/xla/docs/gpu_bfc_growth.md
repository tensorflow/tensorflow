# GPU BFC growth in a reserved virtual arena

Preallocated spatial GPU BFC pools can opt into growth when using device memory.
The default memory policy retains a fixed cap of 75% of device memory.
The memory fraction or absolute size determines the initial physical
allocation. If that much memory is not available, the initial region shrinks
like a fixed pool does, with a warning, and the collective (lower-end) limit is
whatever was obtained. The growth cap is a fraction of total device memory (all
of it for the `START+` shorthand), rounded down to the mapping granularity.
Reserving this capacity does not allocate physical memory or guarantee that
growth will succeed. Non-spatial pools, unified memory, and other allocator
kinds retain their existing behavior.

## Configuration

One setting describes both the initial allocation and the growth cap. It is
spelled as `START`, `START+`, or `START-CAP`, fractions of total device memory,
and parsed into a typed `MemFraction` (`FlexMemFraction` or `FixedMemFraction`)
that is passed through to allocator construction:

| String      | Parsed policy                 | Behavior                                       |
|-------------|-------------------------------|------------------------------------------------|
| `0.75`      | `FixedMemFraction{0.75}`      | Default: retain the existing fixed cap of 75%. |
| `0.75+`     | `FlexMemFraction{0.75, 1.0}`  | Preallocate 75%, grow to all device memory.    |
| `0.75-0.85` | `FlexMemFraction{0.75, 0.85}` | Preallocate 75%, grow to at most 85%.          |
| `0.75-0.75` | `FixedMemFraction{0.75}`      | Fixed cap of 75%, equivalent to `0.75`.        |
| `0.5-1.0`   | `FlexMemFraction{0.5, 1.0}`   | Preallocate 50%, grow to all device memory.    |

A bare fraction is always fixed; values above 1 only make sense with unified
memory. `START+` is shorthand for `START-1.0`, so `1+` is fixed and a start
above 1 is rejected. The cap must be at least the start and a growth cap cannot
exceed 1.

Where the string is accepted:

- The PJRT C API create option `memory_fraction_policy` (string). The older
  float option `memory_fraction` still works and retains its fixed cap. JAX
  support for the extended `XLA_PYTHON_CLIENT_MEM_FRACTION` grammar requires
  forwarding the string through `memory_fraction_policy`.
- `--xla_gpu_memory_fraction_policy` in `XLA_FLAGS`, which overrides the
  client's setting for the BFC allocator and works without any JAX change.
- `GpuAllocatorConfig::memory_fraction_policy` for C++ clients;
  `ParseMemFraction` converts the string form. The legacy numeric
  `memory_fraction` field supplies the fixed cap when no policy is set.

The initial allocation (the start fraction, or `gpu_system_memory_size` when
set) is rounded up to the allocation granularity, while the cap is rounded
down. The rounded initial size must fit under the rounded cap or client
creation fails with `InvalidArgument`, even when the unrounded fractions satisfy
`start < cap`. Choose a wider range if both fall within the same granule.
Equal endpoints select a fixed policy, including policies constructed in C++.
Once a pool reaches its cap, allocation reverts to the usual retry and OOM path.
`--xla_gpu_enable_nccl_user_buffers_in_default_space` pins a growable policy to
`FixedMemFraction{start}` with a warning, because automatic registration needs a
fixed, fully registered arena. Allocator kinds other than the BFC pool use only
the start fraction.

## Virtual reservation and physical backing

On CUDA, the existing `DeviceMemAllocator` reserves one VA range for the cap
before BFC's first allocation. `CudaMemoryReservation::Create` calls
`cuMemAddressReserve`; the entire range is initially unmapped. BFC first asks
for the configured preallocation size, rounded up to the mapping granularity.
`CudaRawMemoryAllocation::Create` obtains physical memory with `cuMemCreate`.
`MemoryReservation::MapTo` maps it at the reservation base with `cuMemMap`, then
grants access to the local device and P2P-capable peers with `cuMemSetAccess`.
The rest of the reservation stays unmapped and consumes no GPU physical memory.
The reserved VA range may include alignment padding beyond the cap. That padding
does not increase the physical-memory limit.

An S(0) allocation first tries its owned holes and the central gap. On a miss,
BFC calls its existing `Extend` path, using normal growth sizing and
backpedaling. `DeviceMemAllocator` allocates additional physical backing and
maps it immediately after the mapped prefix. It never unmaps or moves existing
backing. Only after mapping and access setup succeed does it publish the new
address and byte count to BFC. Failure releases only the new backing (and unmaps
the new slice if access setup failed); the existing prefix and BFC accounting
remain unchanged.

```text
                     fixed S(1) limit P                  VA cap C
Initial: | S(1) -> | gap | <- S(0) |....... unmapped ...........|
Growth:  | S(1) -> | gap | <- S(0) |  more S(0)  |.. unmapped ..|
         ^ base never moves                    ^ mapped end M
```

The three limits are distinct: reserved capacity C, physically mapped capacity
M, and the initial collective allocation limit P. Growth advances M, up to C;
it never changes P or the reservation base. A physical allocation may fail below
C if cuDNN, NCCL, another process, or other users consume available device
memory.

## BFC region extension and collective safety

In reserved mode, `DeviceMemAllocator::SupportsCoalescing()` is true and each
successful extension is adjacent to the preceding prefix. BFC's existing
`RegionManager::AddOrExtendAllocationRegion` extends the region's end and handle
table. The new chunk is upper-owned and coalesces with eligible adjacent free
chunks. S(0) buffers can span physical mapping boundaries. S(1), including
alignment and padding, remains confined to the original allocation even when a
free span crosses P. A lower-end request never triggers growth.

BFC asks for its normal growth size first, then backpedals in whole mapping
granules. The floor is the amount needed to complete the region's free tail,
not the full request, so a request can succeed when physical memory is nearly
exhausted but a free tail only needs a small adjacent extension. Generic
suballocators that do not reserve VA retain separate-region growth: nonadjacent
allocations stay upper-owned and cannot coalesce, and an extension that was
sized for a tail but lands elsewhere is returned. Only an explicitly unsupported
reservation falls back to that path; actual CUDA reservation failures are
reported.

Allocation policy remains attached to the memory space: S(1) uses ascending
equal-size holes and exact splitting; S(0) uses descending equal-size holes and
the BFC heuristic for both holes and gap carves.

Ranks may grow independently. When optional automatic registration of ordinary
backing allocations is requested, the pool is pinned to a fixed fraction for
that client so the registered arena never grows; collective-window registration
continues to use the fixed S(1) range. Existing buffers never move or change
backing. Each physical handle is retained until its mapping is unmapped at
teardown, after the device has been synchronized, then the VA reservation is
released.

## Headroom and host run-ahead

Without growth, a default-memory miss enters the allocator retry path and waits
up to about ten seconds for in-flight frees, which throttles a host thread that
has run ahead of the device. With growth, the miss maps more physical memory
immediately, so run-ahead can inflate the mapped prefix. Freed buffers return to
BFC, but their backing stays mapped until teardown (`garbage_collection=false`).
Memory consumed this way is not available to cuDNN or NCCL workspaces, CUDA
graph instantiation, or other processes sharing the GPU. Use a capped policy
such as `0.75-0.85` to leave headroom, or a fixed one such as `0.75-0.75` to
restore the fixed-pool back-pressure.

Growth adds no headroom or future collective-gap reservation. Live S(0)
buffers can still occupy initial capacity needed by a later S(1) request.
Freed buffers return to BFC; backing allocations remain until destruction.
The PjRt cross-host fabric descriptor carries one physical handle, so exporting
an S(0) buffer spanning mappings returns an explicit unsupported-transfer error.
Representing multiple handles in that protocol is a separate extension.
