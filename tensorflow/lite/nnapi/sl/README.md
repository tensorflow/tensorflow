# NNAPI Support Library

Files in this directory are a copy of NNAPI Support Library
[files](https://cs.android.com/android/platform/superproject/+/master:packages/modules/NeuralNetworks/shim_and_sl/;drc=629cea610b447266b1e6b01e4cb6a952dcb56e7e)
in AOSP.

The files had to be modified to make them work in TF Lite context. Here is the
list of differences from the AOSP version:

*   `#include` directives use fully-qualified paths.
*   `#pragma once` directives are changed to header guards. Android paths in
    header guards are changed to TF Lite paths.
*   `tensorflow/lite/nnapi/NeuralNetworksTypes.h` is used for definitions of
    NNAPI types instead of
    [`NeuralNetworksTypes.h` from AOSP](https://cs.android.com/android/_/android/platform/packages/modules/NeuralNetworks/+/6f0a05b9abdfe0d17afe0269c5329340809175b5:runtime/include/NeuralNetworksTypes.h;drc=a62e56b26b7382a62c5aa0e5964266eba55853d8).
*   `loadNnApiSupportLibrary(...)` is using `tensorflow/lite/minimal_logging.h`
    for logging on errors.
*   `SupportLibrary.h` declarations are wrapped into `tflite::nnapi` namespace.
*   `__BEGIN_DECLS` and `__END_DECLS` are changed to explicit `extern "C"`
    blocks.
*   Copyright notice is changed to the one used in Tensorflow project.
