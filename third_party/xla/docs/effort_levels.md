# Effort Levels

XLA provides options to control the amount of effort the compiler will expend to

*   optimize for runtime performance, and
*   make the program "fit in memory" (which has a platform-dependent meaning)

## Optimization Level

Similar to the -O flags in gcc or clang, this field allows the user to influence
how much work the compiler does in optimizing for execution time. It can be set
via the
[optimization_level](https://github.com/openxla/xla/blob/f4d624b6811c28925c3006f5b779f1149b3b39ac/xla/pjrt/proto/compile_options.proto#L71)
field of the ExecutableBuildOptionsProto message, or the
[optimization_level](https://github.com/openxla/xla/blob/f4d624b6811c28925c3006f5b779f1149b3b39ac/xla/xla.proto#L1580)
field of the ExecutionOptions message.

Lower optimization levels will cause various HLO passes to behave differently,
typically doing less work, or may disable certain HLO passes entirely. The
optimization level may also influence the compiler backend, such that the exact
effect of this field has a dependence on the target platform. However, as a
general guideline, the following table describes the expected overall effect of
each value:

| Level     | Use Case                                                                |
| :-------- | :---------------------------------------------------------------------- |
| EFFORT_O0 | Fastest compilation, slowest runtime                                    |
| EFFORT_O1 | Faster compilation with reasonable runtime                              |
| EFFORT_O2 | Strongly prioritize runtime (suitable default for production workloads) |
| EFFORT_O3 | Expensive or experimental optimizations                                 |

### Use in XLA:GPU

In XLA:GPU, there are several passes that we disable by default because they
significantly increase compilation time by increasing the HLO size. For
convenience, we consolidate them under the optimization level option, such that
setting optimization_level to O1 or above will lead to the following behavior:

*   Collectives commonly used for data-parallel communication will be pipelined.
    This behavior can also be steered more granularly by enabling individual
    flags.
    *   `xla_gpu_enable_pipelined_all_gather`
    *   `xla_gpu_enable_pipelined_all_reduce`
    *   `xla_gpu_enable_pipelined_reduce_scatter`
*   Unrolling while loops by a factor of two. Breaks down the loop-barrier
    potentially leading to a better compute-communication overlap and less
    copies.
    *   `xla_gpu_enable_while_loop_double_buffering`
*   Latency Hiding Scheduler will do most of the work to hide the communication
    latency.
    *   `xla_gpu_enable_latency_hiding_scheduler`
*   To maximize networking bandwidth, combiner passes will combine pipelined
    collectives to the maximum available memory. The optimization does not kick
    in if the loop is already unrolled in the input HLO.
    *   [all_gather_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/all_gather_combiner.cc#L78)
    *   [all_reduce_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/all_reduce_combiner.cc#L76)
    *   [reduce_scatter_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/reduce_scatter_combiner.cc#L76)

## Memory Fitting Level

Another effort level option controls the degree to which the compiler will
attempt to make the resulting program "fit in memory", where "fit" and "memory"
have backend-dependent meanings (for example, in XLA:TPU, this option controls
the degree to which the compiler works to keep the TPU's high-bandwidth memory
(HBM) usage below the HBM capacity). It can be set via the
[memory_fitting_level](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/pjrt/proto/compile_options.proto#L79)
field of the ExecutableBuildOptionsProto message, or the
[memory_fitting_level](https://github.com/openxla/xla/blob/f4d624b6811c28925c3006f5b779f1149b3b39ac/xla/xla.proto#L1588)
field of the ExecutionOptions message.

As with optimization level, the exact meaning of each effort level value is
backend-dependent, but the following table describes the expected effect as a
general guideline:

| Level     | Use Case                                                                |
| :-------- | :---------------------------------------------------------------------- |
| EFFORT_O0 | Minimal effort to fit (fail compilation as quickly as possible instead) |
| EFFORT_O1 | Reduced effort to fit                                                   |
| EFFORT_O2 | Significant effort to fit (suitable default for production workloads)   |
| EFFORT_O3 | Expensive or experimental algorithms to reduce memory usage             |
