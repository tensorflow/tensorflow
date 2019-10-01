# Arm NN

Arm NN is a key component of the [machine learning platform](https://mlplatform.org/), which is part of the [Linaro Machine Intelligence Initiative](https://www.linaro.org/news/linaro-announces-launch-of-machine-intelligence-initiative/).
For more information on the machine learning platform and Arm NN, see: <https://mlplatform.org/>, also there is further Arm NN information available from <https://developer.arm.com/products/processors/machine-learning/arm-nn>

The following kernels are currently off-loaded:
- Pool2d
    - AvgPool
    - MaxPool
    - L2Pool
- Conv2d
- Depthwise Conv2d
- Softmax
- Squeeze

Further steps:
- Add bazel build dependency
- Add further operator support :
    - Activation
        - Logistic
        - TanH
        - Relu
        - Relu6
    - Abs
    - Add
    - ArgMin/ArgMax
    - BatchToSpaceNd
    - Concat
    - DepthToSpace
    - Dequantize
    - DilatedConv2d
    - DilatedDepthwiseConv2d
    - Div
    - Equal, Greater, GreaterEqual, LessEqual, Less
    - Floor
    - FullyConnected
    - Gather
    - LSTM / QuantizedLSTM
    - Maximum/Minimum
    - Normalization
    - Mul
    - Pad
    - Permute
    - Prelu
    - Dequantize
    - Reshape
    - ResizeBilinear / ResizeNearestNeighbour
    - Rsqrt
    - Slice
    - SpaceToBatch
    - SpaceToDepth
    - Stack
    - StridedSlice
    - Sub
    - Switch
    - TransposedConv2d
- Add support for new TensorFlow quantization scheme
- Enable native half-precision floating point support

## Installation
Follow tutorial here https://www.tensorflow.org/lite/guide/build_arm64
Add ArmNN dependency:
``` bash
cd tensorflow/lite/tools/make/downloads
# Latest stable release
git clone https://github.com/ARM-software/armnn.git
# or
# Latest development branch
git clone "https://review.mlplatform.org/ml/armnn"
```
Compile ArmNN
``` bash
# Build Boost
tar -zxvf boost_1_64_0.tar.gz
cd boost_1_64_0
echo "using gcc : arm : aarch64-linux-gnu-g++ ;" > user_config.jam
./bootstrap.sh --prefix=$HOME/armnn-devenv/boost_arm64_install
./b2 install toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options -j32 --user-config=user_config.jam

# Build ComputeLibrary
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary/
scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j8 internal_only=0

# Build Bare ArmNN
cd armnn
mkdir build
cd build

CXX=aarch64-linux-gnu-g++ \
CC=aarch64-linux-gnu-gcc \
cmake .. \
-DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary \
-DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build/ \
-DBOOST_ROOT=$HOME/armnn-devenv/boost_arm64_install/ \
-DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1
```
More details can be found in https://github.com/ARM-software/armnn/blob/branches/armnn_19_08/BuildGuideCrossCompilation.md

Compile TensorFlow Lite with ArmNN delegate support:
``` bash
make -f tensorflow/lite/tools/make/Makefile -j32 BUILD_WITH_ARMNN=true ARMNN_DIR={ARMNN_ROOT} ARMNN_BUILD_PATH={ARMNN_LIB_PATH}

# Note might need to comment #define TFLITE_REDUCE_INSTANTIATIONS_OPEN_SOURCE
```

Enable ArmNN delegate:
``` cpp
  Interpreter::TfLiteDelegatePtr CreateArmNNDelegate(
      ArmNNDelegate::Options options) {
    return Interpreter::TfLiteDelegatePtr(
        new ArmNNDelegate(options), [](TfLiteDelegate* delegate) {
          delete reinterpret_cast<ArmNNDelegate*>(delegate);
        });
  }

  ArmNNDelegate::Options opts;
  opts.backend_name = "CpuRef";
  auto armnn_delegate = CreateArmNNDelegate(opts));
  interpreter->ModifyGraphWithDelegate(armnn_delegate);
```