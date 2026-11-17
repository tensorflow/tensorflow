# Debugging non-determinism with XSC

Non-determinism in programs can occur due to a bad chip or a bug in XLA (both
HLOs and kernels). The [XLA SDC Checker (XSC)](./xsc.md) can detect
non-determinism in all of these cases. It reports them as SDC events that need
to be analyzed to localize the cause of non-determinism.

## Basic configuration

Enable XSC Checker with the following flag. By default, it instruments the most
non-trivial HLOs to run the HLO twice and check if the output checksum is the
same for both runs.

```
--xla_tpu_enable_sdc_checker=true
```

You can see the list of instrumented HLOs in the compilation logs, including the
reason of non-instrumenting, for example:

![With XSC enabled, logs include messages like "HLO is instrumented (HLO rerun,
double-checksum, alternating-cores) :: pad_add_fusion" for each of the
instrumented HLOs](../images/xsc_basic_config.png)

## SDC Events analysis: Bad chips or XLA bugs

All detected non-determinism is reported as SDC events, and you need to
distinguish between the events indicating bad chips from those indicating
XLA/kernel bugs. The best way to do so is to collect statistics over many events
and reason about event locality:

*   If all collected events are identified at the same chip (rarely few chips)
    and happen in various HLOs/kernels, it most probably indicates hardware
    issues.
*   If the events are distributed evenly across the chips in the slice, but
    mostly happen in the same/similar HLOs (e.g. the similar `AllReduce` HLOs,
    or similar kernels), then it may indicate some bug in those HLOs.

## More configurations

The following configurations can be helpful for detecting more non-determinism
events and localizing them:

*   [**Silent reporting**](./xsc.md#silent-reporting-of-sdc-events): It is
    recommended to use "silent reporting mode", where XSC does not crash the
    program but allows it to continue execution after reporting SDC event
    information to the log. Set the flag:
    `--xla_tpu_sdc_check_halt_on_detection=false`
*   [**Increase number of HLO repeats**](./xsc.md#hlo-rerun): It may be
    beneficial in some cases to increase the number of HLO reruns by setting
    N>1: `--xla_tpu_sdc_check_repeat_count=N`
*   [**Filtering instrumented HLOs**](./xsc.md#fine-level-hlo-filtering): Make
    debugging more focused on particular HLO kinds or particular set of HLOs.
    Set the flag: `--xla_tpu_sdc_checker_extra_filter=...`

If the basic configuration does not find *any* events: [Compile your program
without **deduplicated HLOs**](./xsc.md#working-with-deduplicated-hlos).

Depending on your program, also see:

*   [Disable HLO input/output **aliasing**](./xsc.md#checking-hlo-with-aliasing)
*   [Change the **checksum algorithm**](./xsc.md#change-checksum-algorithm)

## SparseCore HLO rerun

Some of SparseCore programs, namely offloaded collectives, also support **HLO
rerun directly on SparseCore**. TensorCore instrumentation does NOT support
rerun of *-start or *-done of the offloaded collective, so rerunning the
operation on SparseCore is the only way to reproduce non-determinism.

SparseCore instrumentation is enabled with the following flag.

```
--xla_tpu_enable_sc_sdc_checker=true
```

**NOTE:** The XSC should still be enabled.

The number of reruns may be changed with the following flag. Not specifying this
flag results in XSC performing the same number of operations on SparseCore, as
it does on TensorCore.

```
--xla_tpu_enable_sc_sdc_checker_loop_iteration_count=N
```

## Instrumenting custom kernels

Sometimes kernels implementation does not comply with requirements of
determinism XSC assumes. For example, in case of ragged kernels, some part of
the output tensor is intentionally left uninitialized.

To address this problem XSC forces initialization of all kernel outputs with 0
value, which causes extra overhead and may hide some of the issues. Even with
zeroing output buffers, custom kernels' may still produce non deterministic
results for any reason the author of the kernel may have. Sometimes it is
beneficial to disable all kernel implementation:

```
--xla_tpu_sdc_checker_enable_tpu_custom_calls=false
```

If there are just a few kernels which should be disabled, more [flexible HLO
name filtering](./xsc.md#fine-level-hlo-filtering) can be used.

## Double-checksum

XSC has an option to compute each of the checksum twice in a row and see if it
has changed across the computations. This checksum can be enabled with:

```
--xla_tpu_sdc_checker_double_check_checksums='true'
```

An event detected here would mean that the buffers are corrupted at the moment
of checksumming. Such corruption may be a result of the HLO being executed or
any other HLO executed concurrently to it.

Usually this feature is combined with enabling checksumming of the HLO inputs:

```
--xla_tpu_sdc_check_inputs=true
```

And, disabling HLO rerun to allow the XSC to instrument more HLOs (i.e.
asynchronous operations which cannot be instrumented when N>0):

```sh
--xla_tpu_sdc_check_repeat_count=0
```

## Performance impact of the instrumentation

Rerunning HLOs and checksumming inputs and outputs comes at cost. Depending on
how many HLOs are instrumented and how big checksummed buffers are the
performance impact of the instrumentation may be significant.

With checksumming only outputs and only 1 HLO rerun, on an average model you can
expect performance overhead of 1.8-2.5x. For debugging, it is usually not a
problem unless extra execution time impacts reproducibility of the issue being
debugged.

## Flags reference

*   `--xla_tpu_enable_sdc_checker=true`: enables instrumentation on TensorCore
*   `--xla_tpu_sdc_check_repeat_count`: the number of HLO reruns, default: 1
    (reruns HLO once after the primary HLO execution)
*   `--xla_tpu_sdc_check_halt_on_detection`: if false, enables silent reporting
    mode, default: true
*   `--xla_tpu_enable_sc_sdc_checker='true'`: if xla_tpu_enable_sdc_checker
    is set also enables checks on SparseCore
*   `--xla_tpu_enable_sc_sdc_checker_loop_iteration_count`: specifies number of
    reruns on SparseCore, default: value xla_tpu_sdc_check_repeat_count
*   `--xla_tpu_sdc_check_inputs`: if the inputs of the HLO need to be
    checksummed, default: false
*   `--xla_tpu_sdc_checker_enable_tpu_custom_calls`: if the kernels need to be
    instrumented, default: true
*   `--xla_tpu_sdc_checker_checksum_algo`: specifies checksum algorithm to be
    used, values: {'default', 'xor', 'siphash13'}, default: 'xor'
*   `--xla_tpu_sdc_checker_double_check_checksums`: enables double-checksumming,
    default: false
*   `--xla_tpu_sdc_checker_extra_filter=...`
*   `--xla_tpu_disable_aliasing_extra_filter=...`
*   `--xla_jf_hlo_deduplication_all_unique=true`
*   `--xla_jf_hlo_deduplication_all_unique_filter=...`
