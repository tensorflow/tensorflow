# OSS huge HLO Benchmarks

This directory contains OSS HLO benchmarks that are exceptionally large or
memory-intensive. The main criterion for putting them is the RAM required for
 running a HLO benchmark: it should go here if it requires > 24GB of RAM.

They are separated from the main `hlo/` directory to avoid being accidentally
run by generic wildcard commands or globs (e.g.,
`xla/backends/cpu/benchmarks/hlo/*`).

To run these benchmarks, you should reference them explicitly from this package.
