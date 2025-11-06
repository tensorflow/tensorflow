# XLA:GPU optimization Level (-On).

There are some passes which we find beneficial to run (especially at scale), but
they increase the HLO size, and thus compilation time. That's why they are not
enabled by default. For convenience, we consolidate them under a single option.

Setting [optimization_level](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/pjrt/proto/compile_options.proto#L66-L71)
to [O1 or above](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/xla.proto#L1481) will lead to the following behaviour:

* Collectives commonly used for data-parallel communication will be pipelined.
This behavior can also be steered more granularly by enabling individual flags.
  * `xla_gpu_enable_pipelined_all_gather`
  * `xla_gpu_enable_pipelined_all_reduce`
  * `xla_gpu_enable_pipelined_reduce_scatter`
* Unrolling while loops by a factor of two. Breaks down the loop-barrier potentially leading to a better compute-communication overlap and less copies.
  * `xla_gpu_enable_while_loop_double_buffering`
* Latency Hiding Scheduler will do most the work to hide the communication latency.
  * `xla_gpu_enable_latency_hiding_scheduler`
* To maximize networking bandwidth, combiner passes will combine pipelined
collectives to the maximum available memory. The optimization does not kick in
if the loop is already unrolled in the input HLO.
  * [all_gather_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/all_gather_combiner.cc#L78)
  * [all_reduce_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/all_reduce_combiner.cc#L76)
  * [reduce_scatter_combiner](https://github.com/openxla/xla/blob/5b54d0e9cf34f4e5ab05b3752ecb390145ca5716/xla/service/gpu/transforms/collectives/reduce_scatter_combiner.cc#L76)
