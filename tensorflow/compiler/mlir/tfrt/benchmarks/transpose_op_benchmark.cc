/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

static const char* mlir_2d_input = R"(
func.func @compute(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {{
    %0 = "tf.Const"()
         {{value = dense<[1, 0]> : tensor<2xi64>,
          device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : () -> tensor<2xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    func.return %1 : tensor<?x?xf32>
  }
)";

static const char* mlir_3d_input = R"(
func.func @compute(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {{
    %0 = "tf.Const"()
         {{value = dense<[{0}, {1}, {2}]> : tensor<3xi64>,
          device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : () -> tensor<3xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    func.return %1 : tensor<?x?x?xf32>
  }
)";

static std::string Transpose2D() { return llvm::formatv(mlir_2d_input); }

static std::string Transpose3D(std::array<int32_t, 3> perm) {
  return llvm::formatv(mlir_3d_input, perm[0], perm[1], perm[2]);
}

template <int32_t size>
static auto Shuffle(std::array<int32_t, size> perm) {
  return [perm](llvm::ArrayRef<Tensor> inputs,
                llvm::Optional<Eigen::ThreadPoolDevice> device) {
    std::array<int64_t, size> shuffled;
    for (unsigned d = 0; d < size; d++)
      shuffled[d] = inputs[0].dim_size(perm[d]);

    Tensor output(DT_FLOAT, TensorShape(shuffled));

    auto in0 = inputs[0].tensor<float, size>();
    auto out0 = output.tensor<float, size>();

    if (device.hasValue()) {
      out0.device(*device) = in0.shuffle(perm);
    } else {
      out0 = in0.shuffle(perm);
    }
  };
}

static llvm::SmallVector<InputTensorSpec> Inputs(llvm::ArrayRef<ssize_t> dims) {
  return {InputTensorSpec(DT_FLOAT, dims)};
}

#define BM(FN) BM_##FN->Arg(0)->Arg(4)->Arg(8);

// Small 2D Transpose: [1, 0]
BM(Jitrt(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(JitrtV(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(Tfrt(Transpose_small_1x0, Transpose2D(), "compute", Inputs({128, 128})));
BM(Eigen(Transpose_small_1x0, Shuffle<2>({1, 0}), Inputs({128, 128})));

// Small 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_0x2x1, Shuffle<3>({0, 2, 1}), Inputs({32, 32, 16})));

// Small 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_2x0x1, Shuffle<3>({2, 0, 1}), Inputs({32, 32, 16})));

// Small 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_2x1x0, Shuffle<3>({2, 1, 0}), Inputs({32, 32, 16})));

// Small 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_1x2x0, Shuffle<3>({1, 2, 0}), Inputs({32, 32, 16})));

// Small 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({32, 32, 16})));
BM(JitrtV(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({32, 32, 16})));
BM(Tfrt(Transpose_small_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({32, 32, 16})));
BM(Eigen(Transpose_small_1x0x2, Shuffle<3>({1, 0, 2}), Inputs({32, 32, 16})));

// Medium 2D Transpose: [1, 0]
BM(Jitrt(Transpose_medium_1x0, Transpose2D(), "compute", Inputs({4096, 4096})));
BM(JitrtV(Transpose_medium_1x0, Transpose2D(), "compute",
          Inputs({4096, 4096})));
BM(Tfrt(Transpose_medium_1x0, Transpose2D(), "compute", Inputs({4096, 4096})));
BM(Eigen(Transpose_medium_1x0, Shuffle<2>({1, 0}), Inputs({4096, 4096})));

// Medium 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_0x2x1, Shuffle<3>({0, 2, 1}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_2x0x1, Shuffle<3>({2, 0, 1}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_2x1x0, Shuffle<3>({2, 1, 0}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_1x2x0, Shuffle<3>({1, 2, 0}),
         Inputs({256, 256, 256})));

// Medium 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({256, 256, 256})));
BM(JitrtV(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({256, 256, 256})));
BM(Tfrt(Transpose_medium_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({256, 256, 256})));
BM(Eigen(Transpose_medium_1x0x2, Shuffle<3>({1, 0, 2}),
         Inputs({256, 256, 256})));

// Large 2D Transpose: [1, 0]
BM(Jitrt(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(JitrtV(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(Tfrt(Transpose_large_1x0, Transpose2D(), "compute", Inputs({8192, 8192})));
BM(Eigen(Transpose_large_1x0, Shuffle<2>({1, 0}), Inputs({8192, 8192})));

// Large 3D Transpose: [0, 2, 1]
BM(Jitrt(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_0x2x1, Transpose3D({0, 2, 1}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_0x2x1, Shuffle<3>({0, 2, 1}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [2, 0, 1]
BM(Jitrt(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_2x0x1, Transpose3D({2, 0, 1}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_2x0x1, Shuffle<3>({2, 0, 1}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [2, 1, 0]
BM(Jitrt(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_2x1x0, Transpose3D({2, 1, 0}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_2x1x0, Shuffle<3>({2, 1, 0}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [1, 2, 0]
BM(Jitrt(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_1x2x0, Transpose3D({1, 2, 0}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_1x2x0, Shuffle<3>({1, 2, 0}),
         Inputs({448, 448, 448})));

// Large 3D Transpose: [1, 0, 2]
BM(Jitrt(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
         Inputs({448, 448, 448})));
BM(JitrtV(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
          Inputs({448, 448, 448})));
BM(Tfrt(Transpose_large_1x0x2, Transpose3D({1, 0, 2}), "compute",
        Inputs({448, 448, 448})));
BM(Eigen(Transpose_large_1x0x2, Shuffle<3>({1, 0, 2}),
         Inputs({448, 448, 448})));

}  // namespace tensorflow
