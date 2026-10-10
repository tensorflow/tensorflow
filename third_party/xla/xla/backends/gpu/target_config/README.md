The specs in this folder are obtained by calling
`Compiler::GpuTargetConfig::ToString()`, which turns the config into a
`GpuTargetConfigProto`, and then to a `std::string`. Most of the spec is the
device description as a proto `GpuDeviceInfoProto`.

The specs are useful when compiling with the flag
`--xla_gpu_target_config_filename`. Since a hardware generation may have several
SKUs, a spec may not be identical to what we would get on a particular machine,
but it will be "close enough".

Multi-GPU topology specs (e.g. `h100_1x8.txtpb`, `gb200_2x4.txtpb`) are text
serializations of `GpuTopologyProto` and can be passed to
`--xla_gpu_topology_filename` (which also accepts an inline shorthand
`[platform:]num_partitionsxnum_hosts_per_partitionxnum_devices_per_host`).
