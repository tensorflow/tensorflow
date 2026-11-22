# Debugging NaNs with XSC

Tracing of NaNs and Infs[^1] is a feature supported by [XLA SDC
Checker (XSC),](./xsc.md) which optionally
checks HLO outputs (and optionally inputs) for NaNs/Infs and reports these as a
special kind of Silent Data Corruption (SDC) event.

The debugging workflow involves re-running the program (or a reproducible
snippet) with extra XSC flags set, and then manually analyzing reported SDC
events to identify the original HLO producing the NaNs/Inf. You will likely
encounter many false positives that need to be detected and removed to identify
the problematic HLO.

## Basic configuration

Run your program with the following flags to detect and report all HLOs with
NaNs or Infs in the output.

```sh
--xla_tpu_enable_sdc_checker=true
--xla_tpu_sdc_check_repeat_count=0
--xla_tpu_sdc_check_halt_on_detection=false
--xla_tpu_sdc_checker_enable_nan_checking=true
```

Example output:

![With XSC NaN checking enabled, the logs include messages like "[XLA_SDC_CHECKER] chip[0x0x0_TC0] host[rscoz6:pe1:2] ::> OUTPUT data has Nan :: input.1, HLO :: <n/a> "](./images/xsc_basic_config_nan.png)

**NOTE:** SDC Events allow differentiation between NaNs and Infs. Although,
usually these two kinds of special values are both relevant to NaN debugging
because Infs often have tendency to become NaNs. Hence, understanding the source
of NaNs needs to account for the cases when NaNs are produced from Infs, and
when those Infs are in-turn created.

## SDC events analysis

After all events are collected, you need to map the events to the program
control flow graph and analyze them to determine the primary sources of NaNs
which made their way to the place where the original symptom was observed.

### Detect and remove false positives

The primary goal of the analysis is to detect and remove false positives, which
are very common for NaN debugging. False positives are the cases where NaN/Infs
are indeed produced by the program but are expected to be eliminated later, for
example:

1.  A tensor gets bitcast to a wider shape (i.e. now includes padding),
    processed by a subgraph, and later gets bitcast back to the original shape,
    in which case intermediate HLOs may observe NaN/Infs which are expected to
    be eliminated eventually.
2.  A tensor with Inf padding is generated and processed by a subgraph, and
    later only a subshape is extracted with dynamic slice with the result not
    expecting to have NaN/Infs.
3.  Ragged pallas kernel intentionally carries implicit padding in its input
    and/or output tensors.

This is often manual analysis of the reported SDC events. Try to reason about
where the NaN/Infs are reported in the HLO graph, and if there is any
programmatic reason for the NaNs. For example, only the first event could be
worth investigating and the following ones could be propagating from the
original.

**TIP:** [Fine-level filtering](./xsc.md#fine-level-hlo-filtering)
can help eliminate false positives by ignoring certain HLOs with regex matching.

## Configurations to track NaN/Infs

### Track NaN/Infs in input

Set the following flag to track NaN/Infs in HLO input (as well) and use this
information during analysis.

```
--xla_tpu_sdc_check_inputs=true
```

### Track only new NaN/Infs

Pass the following flags to ONLY report the SDC Events for HLOs which have
NaN/Infs in the output but NOT in inputs.

```sh
--xla_tpu_sdc_check_inputs=true
--xla_tpu_sdc_checker_new_nans_only=true
```

This option helps dramatically reduce the number of hits at the cost of losing
some events which are potentially real culprits. Good for the first round of
NaN/Inf debugging.

**For example:** HLO produces an output with expected Infs (will be reported as
"new" hit), but those NaN/Infs won't be properly removed later (see the cases of
false positives explained above). In this case, the real culprit would be an HLO
which failed to properly remove padding, but it will NOT be reported since it
has NaN/Infs in both input and output.

**Important notes:**
*   Collective ops require extra attention since they tend to propagate
    NaNs/Infs from a single core to all cores in the relevant replica group.
*   All async HLOs are always reported if they have NaN/Infs since currently we
    don't do checks across several HLOs.

## Configurations to improve NaN/Inf detection

### Padding output buffers

The following flags make XSC also pad the output buffers with NaNs or a
specially provided value. This may help increase the chances for NaNs coming
from uninitialized memory.

```sh
--xla_tpu_sdc_checker_poison_padding=true
--xla_tpu_sdc_checker_poison_padding_value=... // optional, NANs by default
```

### Fine-level filtering

In some cases it may be important to filter out false positives by just not
instrumenting some of HLOs, e.g. bitcasts, this can be done by passing an extra
XSC flag:

```
--xla_tpu_sdc_checker_extra_filter=...
```

Learn more about [filtering in the XSC
documentation](./xsc.md#fine-level-hlo-filtering).

## Flags reference

*   `--xla_tpu_enable_sdc_checker`: if true enables SDC Checker, default: false
*   `--xla_tpu_sdc_check_repeat_count`: N, enables HLO rerun with N additional
    HLO executions, disables HLO rerun at value 0, default: 1
*   `--xla_tpu_sdc_check_halt_on_detection`: if false, does NOT halt TPU program
    when an SDC is detected, but reports the SDC event details to the host and
    continues TPU program execution, default: true
*   `--xla_tpu_sdc_checker_enable_nan_checking`: if true enables reporting SDC
    events for HLOs that have NaN/Inf
*   `--xla_tpu_sdc_check_inputs`: if true tracks NaN/Infs in HLO inputs
*   `--xla_tpu_sdc_checker_new_nans_only`: if true reports the SDC events ONLY
    for HLOs which have NaN/Infs in the output but NOT in inputs
*   `--xla_tpu_sdc_checker_poison_padding=true`: if true, pads the output
    buffers with NaNs
*   `-xla_tpu_sdc_checker_poison_padding_value`: pad the output buffers with the
    provided values
*   `---xla_tpu_sdc_checker_extra_filter`: excludes HLOs that match the regex
    from instrumentation

[^1]: In the IEEE754 standard, inf (infinity values) are numeric data type
    values that exceed the maximum (or minimum) representable value, and NaN
    (not a number) is a numeric data type value representing an undefined or
    non-representable value.
