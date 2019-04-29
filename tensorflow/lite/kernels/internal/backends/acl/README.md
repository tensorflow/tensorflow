<div align="center">
 <img src="https://raw.githubusercontent.com/ARM-software/ComputeLibrary/gh-pages/ACL_logo.png"><br><br>
</div>

**Arm Compute Library** is an open source software library of optimized machine learning and computer vision primitives that target Arm IPs.

Release repository: https://github.com/arm-software/ComputeLibrary

Development repository: https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary

This backend enables **TensorFlow Lite** to dispatch selected kernels to **Arm Compute Library**.

The following kernels are currently off-loaded:
- Conv2d (FP32)
- Depthwise Conv2d (FP32)

Further steps:
- [ ] Add bazel build dependency
- [ ] Use TensorFlow Lite memory manager in Compute Library
- [ ] Add PreferredBackend(char *) interface to interpreter
- [ ] Enable native half-precision floating point support
- [ ] Enable quantized support
- [ ] Profile and offload additional kernels

## Installation

Follow tutorial here https://www.tensorflow.org/lite/guide/build_arm64

Add Compute library dependency:
```
cd tensorflow/lite/tools/make/downloads

# Latest stable release
git clone https://github.com/ARM-software/ComputeLibrary.git

# or

# Latest development branch
git clone "https://review.mlplatform.org/ml/ComputeLibrary"
```

Compile Compute Library
```
# Linux (Cross-compile)
scons Werror=1 -j8 debug=0 asserts=0 neon=1 opencl=0 os=linux arch=arm64-v8a

# Linux (Native)
scons Werror=1 -j8 debug=0 asserts=0 neon=1 opencl=0 os=linux arch=arm64-v8a build=native

# Android
CXX=clang++ CC=clang scons Werror=1 -j8 debug=0 asserts=0 neon=1 opencl=0 os=android arch=arm64-v8a
```
More details can be found in https://arm-software.github.io/ComputeLibrary/latest/index.xhtml

Compile TensorFlow Lite with Compute Library support:
```
./tensorflow/lite/tools/make/build_aarch64_lib.sh USE_NNAPI=false USE_ACL=true
```

Strip symbols and reduce:
```
strip -R <binary>
```