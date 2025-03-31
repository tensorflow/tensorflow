#ifndef TT_PATTERNS
#define TT_PATTERNS

include "mlir/IR/PatternBase.td"
include "triton/Dialect/Triton/IR/TritonOps.td"

// broadcast(splat(x)) -> splat(x)
def BroadcastSplatPattern :
    Pat<(TT_BroadcastOp (TT_SplatOp $x)),
        (TT_SplatOp $x)>;

// broadcast(broadcast(x)) -> broadcast(x)
def BroadcastBroadcastPattern :
    Pat<(TT_BroadcastOp (TT_BroadcastOp $x)),
        (TT_BroadcastOp $x)>;

#endif
