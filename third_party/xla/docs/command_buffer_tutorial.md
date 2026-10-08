# Tuning GPU command buffers

Command buffers reduce the CPU work needed to launch GPU operations. XLA records
an eligible sequence of operations into a GPU graph, then replays that graph on
subsequent executions. On NVIDIA GPUs, this uses CUDA Graphs; supported AMD GPU
operations use HIP graphs.

This is especially useful when a training step or inference request repeatedly
launches many small kernels. A workload dominated by a few long-running kernels
may benefit less. Recording, updating, and retaining graphs also have costs, so
measure the complete workload before adopting a setting.

This guide describes upstream XLA `main`. Your framework's bundled XLA version
may expose different flags or defaults. The defaults below are XLA defaults; a
framework can override them.

## Set flags before starting your application

For applications that accept XLA environment flags, set `XLA_FLAGS` before
starting the process. In Python, set it before importing or initializing your
XLA-based framework. Restart the process for each configuration you compare.

The examples use Bash and `python train.py` as a placeholder for your
application:

```bash
# Keep the default command types and also allow library collectives.
XLA_FLAGS="--xla_gpu_enable_command_buffer=+COLLECTIVES" python train.py
```

Separate flags with spaces and list entries with commas. These examples replace
`XLA_FLAGS` for that invocation; include any other flags your application needs
in the same string. Use the uppercase enum names shown in this guide.

## Start with the defaults

Command buffers are already enabled for several operation types. You do not
need an extra flag to enable the default behavior.

| Setting | Default | What it controls |
| --- | --- | --- |
| `xla_gpu_enable_command_buffer` | See the command-type table below | Which operations may be captured. |
| `xla_gpu_enable_collectives_command_buffer_filter` | `ALLCOLLECTIVES` | Which supported library collective types may be captured when `COLLECTIVES` is enabled. |
| `xla_gpu_graph_min_graph_size` | `5` | Minimum eligible region size for capture. |
| `xla_gpu_command_buffer_scheduling_mode` | `LHS` | How operations inside a graph may overlap. |
| `xla_gpu_command_buffer_update_mode` | `ALWAYS_UPDATE` | How buffer addresses are managed to reduce graph updates. |
| `xla_cmd_buffer_trace_cache_size` | `16` | Number of cached captures per traced-command cache. |
| `xla_gpu_command_buffer_unroll_loops` | `false` | Whether eligible loops with known trip counts are unrolled during graph lowering. |
| `xla_enable_command_buffers_during_profiling` | `false` | Whether graph execution remains enabled during an XLA-detected profiling session. |

## Choose which operations to capture

`--xla_gpu_enable_command_buffer` accepts a list of operation categories, not a
boolean. It has three useful forms:

```bash
# Add to the defaults.
XLA_FLAGS="--xla_gpu_enable_command_buffer=+COLLECTIVES,+WHILE" python train.py

# Remove one category from the defaults.
XLA_FLAGS="--xla_gpu_enable_command_buffer=-CUSTOM_CALL" python train.py

# Replace the defaults: allow only these categories.
XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLASLT" python train.py

# Disable command buffer conversion for a baseline comparison.
XLA_FLAGS="--xla_gpu_enable_command_buffer=" python train.py
```

Use either a plain replacement list or a list of `+`/`-` modifiers. A plain list
does not preserve omitted defaults. Disabling capture does not disable the
operations themselves: they execute through the normal GPU runtime.

| Category | Enabled by default? | Operations it allows |
| --- | --- | --- |
| `FUSION` | Yes | GPU kernels, including fusion/custom kernels, and supported copies within one device. |
| `CUBLAS` | Yes | Operations using the GEMM thunk path. Retained as a category even when a build lowers most matrix multiplications through other paths. |
| `CUBLASLT` | Yes | Matrix multiplication through the cuBLASLt thunk path. |
| `CUDNN` | Yes | cuDNN thunk operations, such as supported library fusions. |
| `CUSTOM_CALL` | Yes | Compatible custom calls. An FFI handler must advertise command buffer compatibility. |
| `DYNAMIC_SLICE_FUSION` | Yes | Supported fusions combining dynamic slicing with other operations. |
| `CONDITIONAL` | Yes | Supported conditional branches. |
| `COLLECTIVES_KERNEL` | Yes | Collective-kernel thunks, such as eligible one-shot collective kernels. Separate from library collectives. |
| `COLLECTIVES` | No | Supported library collective operations, groups, and send/receive paths. |
| `WHILE` | No | Supported while loops whose condition and body can be captured. |
| `CONVOLUTION` | No | The convolution thunk path, including supported ROCm convolutions. |
| `HOST_EXECUTE` | No | Supported host-execution start/done operations; support is still initial. |

These categories follow the compiled execution path, not the name of your
framework operation. For example, a matrix multiplication implemented as a
generated kernel belongs to `FUSION`, while a cuBLASLt call belongs to
`CUBLASLT`.

Enabling a category makes it eligible for capture; it does not force every
operation into a graph. XLA still checks runtime support, dependencies, nested
operations, and the minimum graph size.

### Select library collectives

To capture only selected library collective types, enable `COLLECTIVES` and set
`--xla_gpu_enable_collectives_command_buffer_filter`:

```bash
XLA_FLAGS="--xla_gpu_enable_command_buffer=+COLLECTIVES \
--xla_gpu_enable_collectives_command_buffer_filter=ALLREDUCE,ALLGATHER" \
python train.py
```

The named operation types are `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`,
`COLLECTIVEBROADCAST`, `ALLTOALL`, `COLLECTIVEPERMUTE`, and `RAGGEDALLTOALL`.
`ALLCOLLECTIVES` allows all supported types and is the default. An empty filter
also means all types, so it is not a way to disable collective capture.

The filter does not enable `COLLECTIVES` by itself. It applies to mapped
collective operation types within that category; it is not a universal filter
for groups, send/receive, or `COLLECTIVES_KERNEL`. Use a plain list such as the
example above to narrow the filter: subtracting a type while retaining
`ALLCOLLECTIVES` does not exclude it.

To exclude both library collectives and collective kernels from capture:

```bash
XLA_FLAGS="--xla_gpu_enable_command_buffer=-COLLECTIVES,-COLLECTIVES_KERNEL" \
python train.py
```

## Adjust the minimum graph size

`--xla_gpu_graph_min_graph_size=5` controls how large an eligible region must be
before XLA captures it. The current implementation counts leaf runtime
operations, including those inside nested structures. This is not a count of
Python operators, tensor elements, or necessarily the final GPU graph nodes.

Lowering the threshold can capture smaller regions:

```bash
XLA_FLAGS="--xla_gpu_graph_min_graph_size=1" python train.py
```

Try this when launch overhead matters and eligible regions are small. Raising
the threshold avoids building graphs for small regions whose capture and update
costs outweigh replay savings. A threshold of `1` still does not make
unsupported operations capturable.

## Choose a scheduling mode

`--xla_gpu_command_buffer_scheduling_mode` controls concurrency inside command
buffers. Start with `LHS`, then compare one alternative at a time.

| Value | Meaning | When to consider it |
| --- | --- | --- |
| `LHS` | Uses the latency-hiding schedule's overlap decisions. | Default starting point, especially for workloads that overlap communication with computation. |
| `SERIALIZE` | Serializes commands within each command buffer. | Diagnose whether concurrency affects performance or execution behavior. It does not serialize the entire application. |
| `CONCURRENT_REGIONS` | Allows concurrency in regions of small, latency-bound kernels while serializing larger kernels. Also changes buffer assignment to limit reuse within concurrent regions. | Workloads with independent small kernels that individually underutilize the GPU. Peak memory can increase compared with `LHS` or `SERIALIZE`. |
| `CONCURRENT` | Allows independent operations to overlap and changes buffer assignment to support that overlap. | An aggressive experiment when the workload has available GPU capacity and memory headroom. Memory use can increase enough to cause an out-of-memory error. |

```bash
XLA_FLAGS="--xla_gpu_command_buffer_scheduling_mode=CONCURRENT_REGIONS" \
python train.py
```

More concurrency is not always faster. Kernels can compete for compute
resources, memory bandwidth, or instruction-cache capacity. Compare both elapsed
time and peak memory usage.

## Reduce graph update overhead

A graph records the device addresses used by its operations. When addresses
change, XLA may need to update graph nodes or capture library calls again.
`--xla_gpu_command_buffer_update_mode` controls the allocation strategy used to
reduce this work.

Virtual memory management (VMM) lets XLA present a fixed device virtual address
to the graph even when the underlying allocation changes. Remapping also has a
cost; it is most useful when it saves more graph-update work than it adds.

| Value | Behavior | When to consider it |
| --- | --- | --- |
| `ALWAYS_UPDATE` | Uses normal allocation addresses and updates commands when their addresses change. Despite the name, it does not unconditionally rebuild the graph on every execution. | Default baseline. |
| `SKIP_TEMP` | Gives command-buffer-referenced preallocated temporary buffers fixed VMM addresses. Other changing allocations can still require updates. | First experiment for reducing update overhead from temporary storage. |
| `SKIP_PROFILED` | Includes those temporary buffers and observes other eligible allocations during initial executions. Selected stable candidates then use fixed VMM addresses. | Repeated workloads where input/output address patterns are stable and graph-update overhead is significant. |

```bash
XLA_FLAGS="--xla_gpu_command_buffer_update_mode=SKIP_TEMP" python train.py

# Run separately to compare profile-guided selection.
XLA_FLAGS="--xla_gpu_command_buffer_update_mode=SKIP_PROFILED" python train.py
```

Both skip modes require a compatible VMM allocator. The upstream StreamExecutor
GPU PJRT client automatically selects its VMM allocator when either is
requested, overriding the requested allocator kind. Other integrations must
supply compatible allocator support; the flag alone does not guarantee that
remapping is active. Compare memory behavior as well as latency because the
allocator changes.

In the current implementation, `SKIP_PROFILED` observes the first three
successful executions per executable/device state and uses ordinary
operation-by-operation execution during that observation window. Subsequent
executions can use the selected remappings and command buffers. Warm up beyond
both the observation window and initial graph setup before timing steady-state
execution.

This internal address profiling is separate from an external performance
profiler. It does not require
`xla_enable_command_buffers_during_profiling=true`. A later address change is
handled through remapping rather than assuming the original address is still
valid. For distributed runs, use consistent settings and representative,
symmetric warmup behavior across ranks.

Neither skip mode promises to eliminate all graph updates: other dynamic buffers
and changing operation parameters can still require them.

## Tune the trace cache

`--xla_cmd_buffer_trace_cache_size=16` sets the capacity of each traced-command
cache. Library calls can be captured by observing the GPU work they enqueue;
the cache reuses those captures for previously seen buffer-address combinations.
This is not the framework's compilation cache or a global limit on graph count.

```bash
XLA_FLAGS="--xla_cmd_buffer_trace_cache_size=32" python train.py
```

A larger cache may help when a workload repeatedly cycles through more address
combinations than the cache holds. It retains more graph resources and increases
cache lookup work. It may not help if addresses rarely repeat. Use a positive
integer; `0` is not a supported way to disable this cache.

## Capture and unroll loops

`--xla_gpu_command_buffer_unroll_loops=true` allows eligible loops with a known
trip count to be expanded into repeated graph commands. The default is `false`.
Enable `WHILE` capture as well when testing whole-loop capture:

```bash
XLA_FLAGS="--xla_gpu_enable_command_buffer=+WHILE \
--xla_gpu_command_buffer_unroll_loops=true" python train.py
```

Unrolling can reduce loop-control overhead and enables some cases with
loop-dependent dynamic-slice offsets. It can also produce a much larger graph,
increasing setup time and memory use. Start with short, fixed-trip-count loops;
unknown trip counts or unsupported body operations prevent this transformation.

## Profile the execution path you intend to measure

`--xla_enable_command_buffers_during_profiling` defaults to `false`. During an
XLA-detected active profiling session, XLA normally falls back to individual
operations. A profiled run can therefore behave differently from an unprofiled
run even when capture is enabled.

To keep graph execution enabled during such a session:

```bash
XLA_FLAGS="--xla_enable_command_buffers_during_profiling=true" python train.py
```

This option is marked experimental. Use it when investigating graph replay, and
compare against unprofiled wall-clock measurements. It does not enable
additional command categories or bypass capture eligibility checks. External
profilers that do not activate XLA's profiling-session detection may behave
differently.

## Run a useful comparison

1. Measure the defaults and a separate run with command buffers disabled.
2. Warm up each compiled workload before measuring. Exclude compilation, initial
   capture, and `SKIP_PROFILED` observation/setup from steady-state timings, but
   measure startup separately if it matters to your application.
3. Wait for GPU work to finish before stopping the timer. For example, with JAX,
   call `jax.block_until_ready(result)`; timing asynchronous dispatch alone does
   not measure execution time.
4. Change one setting at a time. Keep shapes, inputs, device count, and profiler
   configuration comparable. Apply the same configuration on every worker.
5. Compare output correctness, steady-state latency or throughput, startup time,
   and peak memory. Keep a setting only when it improves your target workload.

## Troubleshooting and version compatibility

| Observation | What to check |
| --- | --- |
| No speedup | The workload may already be dominated by long kernels, or capture/update costs may offset launch savings. Compare warmed-up, synchronized measurements. |
| Eligible operations remain outside graphs | Check command categories, graph-size threshold, dependencies, FFI compatibility, and GPU runtime support. An unsupported operation can split a region. |
| Behavior changes under profiling | Check `xla_enable_command_buffers_during_profiling` and whether the profiler activates XLA's fallback. |
| Higher memory use | Compare against `LHS`, the default trace-cache capacity, loop unrolling disabled, and `ALWAYS_UPDATE`, one change at a time. |
| Unknown flag or enum value | Check the XLA version bundled with your framework. A newer upstream flag may not yet be in a release. |

Backend support differs. CUDA command buffer capture is disabled if either the
known runtime or driver version is below 12.3. DynamicSliceFusionV2 capture
requires runtime, driver, and build-time CUDA toolkit versions of at least 12.9.
ROCm disables conditional/while graph capture, and oneAPI command buffer
conversion is disabled. Enabling a category does not override these checks.

For implementation details, see the
[flag definitions](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc),
[option enums](https://github.com/openxla/xla/blob/main/xla/xla.proto),
[capture eligibility checks](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc),
and [VMM allocation policy](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_executable_va_remap_allocator.cc).
