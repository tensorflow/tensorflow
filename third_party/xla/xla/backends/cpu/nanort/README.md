# Nano Client For XLA:CPU for ULTRA Low Latency Inference

Warning: **All** users must prefer the official PJRT APIs over NanoRt's.
NanoRt is only meant for a handful of users who cannot accept _any_ overhead.

Warning: **USE AT YOUR OWN RISK**. This API might be deleted at any time and XLA
CPU team does not intend to provide any backward compatibility guarantees.

This is an XLA:CPU API that resembles PjRt Client and Executable, but with a
laser focus on absolute minimal overheads at run time.

Key differences from PjRt:

1. It is focused on ultra low latency inference where each nanosecond matters.
2. It is single replica and partition and does not support any collective
   operations.
3. Memory for parameters, results and temp allocation managed by the user: there
   is no type that corresponds to `PjRtBuffer`, and executable uses destination
   passing style to return results into user-provided memory buffers.
4. NanoRt API is unstable and does not provide any backward compatibility
   guarantees.
