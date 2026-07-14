This directory contains optimized implementations for int8 fully integer kernels.

Weight filters of convs are expected to be symmetric per-channel quantized in
the range [-127, 127].
Inputs/activations are expected to be asymmetric per-layer quantized in the
range [-128, 127].

THESE ARE EXPERIMENTAL AND PRONE TO CHANGE.
