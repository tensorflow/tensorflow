# OSS HLO Benchmarks

This directory contains OSS HLO benchmarks, which should be put into either of
 3 groups below, based on next criteria (for each file to run):

| Group name                | Max RAM used  | Timeout |
|:--------------------------|:-------------:|--------:|
|REGULAR_HLO_FILES          | 12 GB         | 5 mins  |
|SLOW_HLO_FILES             | 12 GB         | 15 mins |
|MEMORY_INTENSIVE_HLO_FILES | 24 GB         | 5 min   |

> [!NOTE]
> Timeouts above are estimated for a single-core CPU.
> To estimate the time threshold  for a specific HLO, divide a timeout above
> by your system's number of cores.