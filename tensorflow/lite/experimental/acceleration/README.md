# Accelerator allowlisting

Experimental library and tools for determining whether an accelerator engine
works well on a given device, and for a given model.

## Platform-agnostic, Android-first

Android-focused, since the much smaller set of configurations on iOS means there
is much less need for allowlisting on iOS.

## Not just for TfLite

This code lives in the TfLite codebase, since TfLite is the first open-source
customer. It is however meant to support other users (direct use of NNAPI,
mediapipe).
