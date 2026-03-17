<!-- linter style off -->

# Hangs

## Locate the Megascale Hang Detected Error

If you see the following error message in your TPU worker logs, this means that
MXLA timed out after detecting no progress:

```
Megascale hang detected: Timed out waiting for 4 graphs to complete at launch_id 13650. Already completed: 100. StepGloballyInProgress: true. Timeout: 1m
```

1.  Workers will report errors to a coordinator for processing.
    *   For **Pathways** jobs: the digest can be found in the logs of
        `resource_manager` job.
    *   For **McJAX** jobs: the logs can be found on MXLA Coordinator. This is
        typically task 0 of slice 0.
2.  Check logs around time of error detection, and look for `Megascale detects a
    hang`.
3.  Follow steps below to diagnose the issue based on the identified cause.

## Diagnosis

### Bad TPU Chip (tensor core or sparse core)

```
Megascale detects a hang that is likely caused by bad TPU chips on the following hosts. Please remove the hosts from the fleet and restart the workload. If problem persists please contact Megascale XLA team.
  The host that have bad TPUs are: <host_name>
  Full error digest:
    Potential cause: Bad TPU chips
    Potential culprit workers: <job_name>/<task_id>:<host_name>
```

This error means that the issue is potentially caused by a faulty TPU chip. The
error message should include the job information and host name of the faulty
chip. In the example above, the faulty chip is on host `<host_name>`, affecting
task `<task_id>` of the job `<job_name>`. You can configure your job to avoid
that host.

**Note:** There are some cases that the hang was caused by a XLA or custom
pallas kernel bug, but if you see the same host appearing multiple times (for
example more than 3 times) in multiple hang events, the TPU on that host is very
likely faulty.

### Networking issue

```
Megascale detects a hang that is likely caused by a networking issue. Please examine the underlying networking stack for the following hosts.
  The hosts are: <host_name>
  Full error digest:
    Potential cause: Networking issue
    Potential culprit workers: <host_name_1>, <host_name_2>
```

This error indicates that your job has encountered a failed network link. The
error message should include a single or pair of job name, task id, host name of
the faulty network link. In the example above, the faulty network link is
between host `host_name_1` and `host_name_2`. Sometimes RapidEye can further
localize the faulty host if a single host appears in multiple broken network
links. You can configure your job to avoid those hosts.

### Different modules

```
Megascale detects a hang that is likely caused by running different modules on different devices. Please confirm that all workers is running the exact same program. It can also be caused by a hang in a subset of devices and the unaffected devices have moved on to the next program. Please inspect the digest below to further root cause the hang.
Example hosts that have different HLO modules: <host_name>
Full error digest:
  Potential cause: Different module
  Potential culprit workers: <host_name>
  TPU stats:
    <host_name>: <pc>
  TPU states:
    Module: jit_loss_and_grad
    Fingerprint: <fingerprint>
    Launch ID: 193
      <tag>:<pc>(<hlo>): <host_name>
    Module: jit_optimizer_apply
    Fingerprint: <fingerprint>
    Launch ID: 0
      <tag>:<pc>(<hlo>): <host_name>
```

This error may indicate that a hang has occurred in a subset of workers, causing
those workers to be stuck at the current module while unaffected workers advance
to the next module. To identify the root cause, inspect the digest printed by
RapidEye in the logs.

The `TPU states` section of the logs shows which modules are running on which
workers. In the example above, workers are running different modules:
`jit_loss_and_grad` and `jit_optimizer_apply`.

### Fingerprint mismatch for HLO module

```
Megascale detects a hang that is likely caused by inconsistent HLO module compilation across workers. This is likely a bug in JAX tracing or XLA compiler. Please inspect the HLO dumps to confirm the root cause.
  Example hosts that have different HLO fingerprints: <host_name>
  Full error digest:
    Potential cause: Fingerprint mismatch
  Potential culprit workers: <host_name>
  TPU stats:
    Module: reduce.31
    Fingerprint: <fingerprint_1>
    Launch ID: 37
      <tag>:<pc>(<hlo>): <host_name>
    Module: reduce.31
    Fingerprint: <fingerprint_2>
    Launch ID: 40
      <tag>:<pc>(<hlo>): <host_name>
```

This log message indicates the hang was likely caused by inconsistent HLO module
compilation across workers, possibly due to an issue with JAX tracing or the XLA
compiler. If you see this log, follow [these
steps](https://openxla.org/xla/hlo_dumps) to collect HLO dumps from the culprit
workers for further debugging.

### Data input stall

**Note:** This error is not yet implemented.

```
Megascale detects a hang that is likely caused by data input stall on the
following hosts. Please check the workers to make sure the data input pipeline
is working properly.
  The host that have data input stalls are: <host_name>
```

This error means that all devices launched the same program, but that input data
was not provided to the program before the system timed out. To fix this issue,
confirm that:

1.  The identified hosts can access the input datasource.
2.  The identified hosts are properly loading/parsing the input datasource.
3.  Confirm identified hosts are not throttled on reads to the input datasource.

### Unrecoverable error

```
Some workers have halted with an unrecoverable error:
  <worker> : {some error}
  Please inspect the error log of these workers:
  <worker>
```

This error means that there was an issue that prevented the program from
properly executing and could not be recovered automatically. This error was
unable to be specifically categorized. Further information can be obtained from
checking the logs of the worker(s) mentioned in the error report.

If the error appears to be specific to the given machine (ex. failure to copy
data from TPU to host), then you can configure your job to avoid those hosts.

### Unknown Error

```
Megascale detects a hang but cannot determine the root cause. Please inspect the
full digest below.
```

This error means that there was an issue that prevented the program from
properly executing and could not be recovered automatically. This error was
unable to be specifically categorized and there is no further error information
available.

<!-- linter style on -->
