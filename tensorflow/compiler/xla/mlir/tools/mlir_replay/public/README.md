# Public API of mlir_replay

This contains protocol buffers and utilities that can be reused for other
debugging tools:

1.  **The compiler trace proto**: A record of the state of the IR after each
    compilation pass
1.  A compiler instrumentation to create the above proto.
1.  **The execution trace proto**: A record of SSA values as the IR is executed
1.  Utilities for working with the above protos.
