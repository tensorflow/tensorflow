# Debugging Workflow

This document describe a general workflow for debugging MXLA issues.

## Prerequisite

1.  Use JAX 0.6 or up, and enable JAX distributed service. This version of JAX
    contains additional logging that can help identify which workers are
    experiencing issues.
2.  Generate an HLO dump using the --xla_dump_to flag when initializing your
    workload. This is discussed in the [XLA
    documentation](https://openxla.org/xla/hlo_dumps).
3.  Set --vmodule=real_program_continuator=1 to enable verbose logging for the
    TPU program execution status.

## Flow chart

The flowchart below illustrates the debugging process. To access detailed
playbooks for each step, click on the corresponding item in the chart.

<iframe src="flow_chart.svg" width="1000" height="2000" style="border: none;"></iframe>