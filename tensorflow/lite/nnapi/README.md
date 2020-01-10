# Android Neural Network API

The Android Neural Networks API (NNAPI) is an Android C API designed for running
computationally intensive operators for machine learning on mobile devices.
Tensorflow Lite is designed to use the NNAPI to perform hardware-accelerated
inference operators on supported devices.
Based on the app’s requirements and the hardware capabilities on a device, the
NNAPI can distribute the computation workload across available on-device
processors, including dedicated neural network hardware, graphics processing
units (GPUs), and digital signal processors (DSPs).
For devices that lack a specialized vendor driver, the NNAPI runtime relies on
optimized code to execute requests on the CPU. For more information about the
NNAPI, please refer to the [NNAPI documentation](https://developer.android.com/ndk/guides/neuralnetworks/index.html)


