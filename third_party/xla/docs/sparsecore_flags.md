# XLA Flags: SparseCore

The `--xla_sc_assert_level` flag sets assert level verbosity, controlling what
is checked (e.g. memory and bounds safety, hardware and control flow integrity,
communication and offloading) and what is logged (how much diagnostic detail is
output.

Possible values for the `--xla_sc_assert_level` flag are either `[no|san|prod]`
(more below under [Aliases](#aliases)), or comma-separated combinations of
options including the following:

| Value | Description |
| :---- | :---- |
| `alltoall` | asserts related to all-to-all op |
| `always` | various critical asserts |
| `bounds` | various bounds checks |
| `checksums` | validate buffer checksums for corruption, related to flag `--xla_tpu_buffer_contents_sanitizer_config` |
| `continuations` | asserts related to launching async sparsecore tasks from TensorCore |
| `csrs` | related to compressed sparse row, for embedding |
| `dma` | related to dma such as alignment |
| `masks` | related to some checks on vector masks, such as when `[1, 1, 1, 0, 0, 0, 0, 0]` is fine, but not `[1, 0, 1, 0,…]` |
| `overlays` | related to overlays which is how we manage and execute program that’s larger than instruction memory on SparseCore sequencer |
| `radixsort` | related to radix sort op/emitter |
| `runids` | related to unique execution id of sparsecore programs, e.g. in order check |
| `scalar-loads` | related to scalar loads, such as bounds check |
| `scalar-stores` | related to scalar stores, such as bounds check |
| `syncflags` | related to sync flags, such as within-bounds/ expected-0 check |
| `streams` | related to stream |
| `vector-loads` | related to vector loads |
| `vector-stores` | related to vector stores |

### Aliases

This flag can also be set to one of the aliases below:

| Alias | Description |
| :---- | :---- |
| `prod` | set to `always`, i.e. critical asserts only. *(default)* |
| `san` | every assert |
| `san-lite` | `always`, `streams`, `dma`, `alltoall`, `runids` |
| `vector-loads-stores` | `vector-loads, vector-stores` |
| `all-loads-stores` | `vector-loads`, `vector-stores`, `scalar-loads`, `scalar-stores` |
| `no` | nothing asserted |
