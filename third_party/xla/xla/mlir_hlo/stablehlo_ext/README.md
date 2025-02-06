# XLA StableHLO Extension

This folder contains StableHLO passes required for XLA compilation.

This includes some handling for some XLA specific custom calls, as well as
XLA-specific passes which include patterns from the StableHLO repo with some
minor modifications needed for compilation, such as directly lowering CHLO ops
to MHLO ops which have compiler support (ex: TopK, Erf, Tan), as opposed to
decomposing them to other primitive StableHLO ops.
