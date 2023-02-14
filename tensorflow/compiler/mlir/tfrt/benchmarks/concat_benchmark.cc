/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <initializer_list>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

llvm::SmallVector<int64_t> GetOutputTypeShape(llvm::ArrayRef<int64_t> arg_shape,
                                              llvm::ArrayRef<bool> dynamic_dims,
                                              int64_t concat_dim,
                                              int64_t num_concats) {
  llvm::SmallVector<int64_t> out_ty =
      GetTensorTypeShape(arg_shape, dynamic_dims);
  if (dynamic_dims[concat_dim] == kStaticDim) out_ty[concat_dim] *= num_concats;
  return out_ty;
}

static const char* kBinaryConcatIR = R"(
func.func @main(%lhs: {0}, %rhs: {0}) -> {1} {
  %0 = "tf.Log"(%lhs): ({0}) -> {0}
  %1 = "tf.Log"(%rhs): ({0}) -> {0}
  %2 = "tf.Const"() {{ value = dense<{2}> : tensor<i64> } : () -> tensor<i64>
  %3 = "tf.ConcatV2"(%0, %1, %2) : ({0}, {0}, tensor<i64>) -> {1}
  %4 = "tf.Log"(%3): ({1}) -> {1}
  func.return %4 : {1}
}
)";

std::string GetBinaryConcatIR(llvm::ArrayRef<int64_t> arg_shape,
                              llvm::ArrayRef<bool> dynamic_dims,
                              int64_t concat_dim) {
  llvm::SmallVector<int64_t> in_ty =
      GetTensorTypeShape(arg_shape, dynamic_dims);
  llvm::SmallVector<int64_t> out_ty = GetOutputTypeShape(
      arg_shape, dynamic_dims, concat_dim, /*num_concats=*/2);
  return llvm::formatv(kBinaryConcatIR, PrintTensorType(in_ty, "f32"),
                       PrintTensorType(out_ty, "f32"), concat_dim);
}

static const char* kTeraryConcatIR = R"(
func.func @main(%arg0: {0}, %arg1: {0}, %arg2: {0}) -> {1} {
  %0 = "tf.Log"(%arg0): ({0}) -> {0}
  %1 = "tf.Log"(%arg1): ({0}) -> {0}
  %2 = "tf.Log"(%arg2): ({0}) -> {0}
  %3 = "tf.Const"() {{ value = dense<{2}> : tensor<i64> } : () -> tensor<i64>
  %4 = "tf.ConcatV2"(%0, %1, %2, %3) : ({0}, {0}, {0}, tensor<i64>) -> {1}
  %5 = "tf.Log"(%4): ({1}) -> {1}
  func.return %5 : {1}
}
)";

std::string GetTernaryConcatIR(llvm::ArrayRef<int64_t> arg_shape,
                               llvm::ArrayRef<bool> dynamic_dims,
                               int64_t concat_dim) {
  llvm::SmallVector<int64_t> in_ty =
      GetTensorTypeShape(arg_shape, dynamic_dims);
  llvm::SmallVector<int64_t> out_ty = GetOutputTypeShape(
      arg_shape, dynamic_dims, concat_dim, /*num_concats=*/3);
  return llvm::formatv(kTeraryConcatIR, PrintTensorType(in_ty, "f32"),
                       PrintTensorType(out_ty, "f32"), concat_dim);
}

static const char* kOctonaryConcatIR = R"(
func.func @main(%arg0: {0}, %arg1: {0}, %arg2: {0}, %arg3: {0}, %arg4: {0},
    %arg5: {0}, %arg6: {0}, %arg7: {0}) -> {1} {
  %0 = "tf.Log"(%arg0): ({0}) -> {0}
  %1 = "tf.Log"(%arg1): ({0}) -> {0}
  %2 = "tf.Log"(%arg2): ({0}) -> {0}
  %3 = "tf.Log"(%arg3): ({0}) -> {0}
  %4 = "tf.Log"(%arg4): ({0}) -> {0}
  %5 = "tf.Log"(%arg5): ({0}) -> {0}
  %6 = "tf.Log"(%arg6): ({0}) -> {0}
  %7 = "tf.Log"(%arg7): ({0}) -> {0}
  %8 = "tf.Const"() {{ value = dense<{2}> : tensor<i64> } : () -> tensor<i64>
  %9 = "tf.ConcatV2"(%0, %1, %2, %3, %4, %5, %6, %7, %8)
      : ({0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, tensor<i64>) -> {1}
  %10 = "tf.Log"(%9): ({1}) -> {1}
  func.return %10 : {1}
}
)";

std::string GetOctonaryConcatIR(llvm::ArrayRef<int64_t> arg_shape,
                                llvm::ArrayRef<bool> dynamic_dims,
                                int64_t concat_dim) {
  llvm::SmallVector<int64_t> in_ty =
      GetTensorTypeShape(arg_shape, dynamic_dims);
  llvm::SmallVector<int64_t> out_ty = GetOutputTypeShape(
      arg_shape, dynamic_dims, concat_dim, /*num_concats=*/8);
  return llvm::formatv(kOctonaryConcatIR, PrintTensorType(in_ty, "f32"),
                       PrintTensorType(out_ty, "f32"), concat_dim);
}

template <int64_t D>
TensorShape GetOutShape(llvm::ArrayRef<Tensor> inputs) {
  auto lhsShape = inputs[0].shape();
  auto rhsShape = inputs[1].shape();
  std::vector<int64_t> shape;
  shape.reserve(inputs[0].dims());
  for (int64_t i = 0; i < inputs[0].dims(); ++i)
    shape.push_back(inputs[0].dim_size(i));
  for (int64_t i = 1; i < inputs.size(); ++i) shape[D] += inputs[i].dim_size(D);
  return TensorShape(shape);
}

template <int64_t R, int64_t D>
auto GetEigenBinaryConcatFn() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    auto lhs = inputs[0].tensor<float, R>();
    auto rhs = inputs[1].tensor<float, R>();
    Tensor output(DT_FLOAT, GetOutShape<D>(inputs));
    auto out = output.tensor<float, R>();
    if (device.has_value()) {
      out.device(*device) = lhs.log().concatenate(rhs.log(), D).log();
    } else {
      out = lhs.log().concatenate(rhs.log(), D).log();
    }
  };
}

template <int64_t R, int64_t D>
auto GetEigenTernaryConcatFn() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    auto arg0 = inputs[0].tensor<float, R>();
    auto arg1 = inputs[1].tensor<float, R>();
    auto arg2 = inputs[2].tensor<float, R>();
    Tensor output(DT_FLOAT, GetOutShape<D>(inputs));
    auto out = output.tensor<float, R>();
    if (device.has_value()) {
      out.device(*device) = arg0.log()
                                .concatenate(arg1.log(), D)
                                .concatenate(arg2.log(), D)
                                .log();
    } else {
      out = arg0.log()
                .concatenate(arg1.log(), D)
                .concatenate(arg2.log(), D)
                .log();
    }
  };
}

template <int64_t R, int64_t D>
auto GetEigenOctonaryConcatFn() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    auto arg0 = inputs[0].tensor<float, R>();
    auto arg1 = inputs[1].tensor<float, R>();
    auto arg2 = inputs[2].tensor<float, R>();
    auto arg3 = inputs[3].tensor<float, R>();
    auto arg4 = inputs[4].tensor<float, R>();
    auto arg5 = inputs[5].tensor<float, R>();
    auto arg6 = inputs[6].tensor<float, R>();
    auto arg7 = inputs[7].tensor<float, R>();
    Tensor output(DT_FLOAT, GetOutShape<D>(inputs));
    auto out = output.tensor<float, R>();
    if (device.has_value()) {
      out.device(*device) = arg0.log()
                                .concatenate(arg1.log(), D)
                                .concatenate(arg2.log(), D)
                                .concatenate(arg3.log(), D)
                                .concatenate(arg4.log(), D)
                                .concatenate(arg5.log(), D)
                                .concatenate(arg6.log(), D)
                                .concatenate(arg7.log(), D)
                                .log();
    } else {
      out = arg0.log()
                .concatenate(arg1.log(), D)
                .concatenate(arg2.log(), D)
                .concatenate(arg3.log(), D)
                .concatenate(arg4.log(), D)
                .concatenate(arg5.log(), D)
                .concatenate(arg6.log(), D)
                .concatenate(arg7.log(), D)
                .log();
    }
  };
}

#define WRAP(...) __VA_ARGS__

#define BM_BINARY_CONCAT(NAME, RANK, ARG_SHAPE, DYNAMIC_DIMS, CONCAT_DIM)      \
  BM_Jitrt(BinaryConcat_##NAME,                                                \
           GetBinaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM), "main", \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),             \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))           \
      ->Arg(0);                                                                \
  BM_JitrtV(BinaryConcat_##NAME,                                               \
            GetBinaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM),        \
            "main",                                                            \
            llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),            \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))          \
      ->Arg(0);                                                                \
  BM_Eigen(BinaryConcat_##NAME, (GetEigenBinaryConcatFn<RANK, CONCAT_DIM>()),  \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),             \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))           \
      ->Arg(0)

#define BM_TERNARY_CONCAT(NAME, RANK, ARG_SHAPE, DYNAMIC_DIMS, CONCAT_DIM) \
  BM_Jitrt(TernaryConcat_##NAME,                                           \
           GetTernaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM),    \
           "main",                                                         \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))       \
      ->Arg(0);                                                            \
  BM_JitrtV(TernaryConcat_##NAME,                                          \
            GetTernaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM),   \
            "main",                                                        \
            llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),        \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),        \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))      \
      ->Arg(0);                                                            \
  BM_Eigen(TernaryConcat_##NAME,                                           \
           (GetEigenTernaryConcatFn<RANK, CONCAT_DIM>()),                  \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))       \
      ->Arg(0)

#define BM_OCTONARY_CONCAT(NAME, RANK, ARG_SHAPE, DYNAMIC_DIMS, CONCAT_DIM) \
  BM_Jitrt(OcternaryConcat_##NAME,                                          \
           GetOctonaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM),    \
           "main",                                                          \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))        \
      ->Arg(0);                                                             \
  BM_JitrtV(OcternaryConcat_##NAME,                                         \
            GetOctonaryConcatIR({ARG_SHAPE}, {DYNAMIC_DIMS}, CONCAT_DIM),   \
            "main",                                                         \
            llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),         \
                            InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))       \
      ->Arg(0);                                                             \
  BM_Eigen(OcternaryConcat_##NAME,                                          \
           (GetEigenOctonaryConcatFn<RANK, CONCAT_DIM>()),                  \
           llvm::ArrayRef({InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE}),          \
                           InputTensorSpec(DT_FLOAT, {ARG_SHAPE})}))        \
      ->Arg(0)

#define BM_NARY_CONCAT(NAME, RANK, ARG_SHAPE, DYNAMIC_DIMS, CONCAT_DIM) \
  BM_BINARY_CONCAT(NAME, RANK, WRAP(ARG_SHAPE), WRAP(DYNAMIC_DIMS),     \
                   CONCAT_DIM);                                         \
  BM_TERNARY_CONCAT(NAME, RANK, WRAP(ARG_SHAPE), WRAP(DYNAMIC_DIMS),    \
                    CONCAT_DIM);                                        \
  BM_OCTONARY_CONCAT(NAME, RANK, WRAP(ARG_SHAPE), WRAP(DYNAMIC_DIMS),   \
                     CONCAT_DIM)

// Static Concat 1D
#define BM_NARY_CONCAT_STATIC_1D(N) \
  BM_NARY_CONCAT(Static1D_##N, 1, WRAP(N), WRAP(kStaticDim), 0)
BM_NARY_CONCAT_STATIC_1D(1);
BM_NARY_CONCAT_STATIC_1D(8);
BM_NARY_CONCAT_STATIC_1D(1024);
BM_NARY_CONCAT_STATIC_1D(1026);
BM_NARY_CONCAT_STATIC_1D(1048576);
BM_NARY_CONCAT_STATIC_1D(1048578);

// Dynamic Concat 1D
#define BM_NARY_CONCAT_DYNAMIC_1D(N) \
  BM_NARY_CONCAT(Dynamic1D_##N, 1, WRAP(N), WRAP(kDynamicDim), 0)
BM_NARY_CONCAT_DYNAMIC_1D(1);
BM_NARY_CONCAT_DYNAMIC_1D(8);
BM_NARY_CONCAT_DYNAMIC_1D(1024);
BM_NARY_CONCAT_DYNAMIC_1D(1026);
BM_NARY_CONCAT_DYNAMIC_1D(1048576);
BM_NARY_CONCAT_DYNAMIC_1D(1048578);

// Static Concat 2D
#define BM_NARY_CONCAT_STATIC_2D(M, N, CONCAT_DIM)                    \
  BM_NARY_CONCAT(Static2D_##M##x##N##_dim##CONCAT_DIM, 2, WRAP(M, N), \
                 WRAP(kStaticDim, kStaticDim), CONCAT_DIM)
// Sqaure operands
BM_NARY_CONCAT_STATIC_2D(512, 512, 0);
BM_NARY_CONCAT_STATIC_2D(512, 512, 1);
BM_NARY_CONCAT_STATIC_2D(514, 514, 0);
BM_NARY_CONCAT_STATIC_2D(514, 514, 1);
BM_NARY_CONCAT_STATIC_2D(1024, 1024, 0);
BM_NARY_CONCAT_STATIC_2D(1024, 1024, 1);
BM_NARY_CONCAT_STATIC_2D(1026, 1026, 0);
BM_NARY_CONCAT_STATIC_2D(1026, 1026, 1);
// Slice operands
BM_NARY_CONCAT_STATIC_2D(1, 1024, 0);
BM_NARY_CONCAT_STATIC_2D(1024, 1, 1);
BM_NARY_CONCAT_STATIC_2D(1, 1026, 0);
BM_NARY_CONCAT_STATIC_2D(1026, 1, 1);
BM_NARY_CONCAT_STATIC_2D(1, 1048576, 0);
BM_NARY_CONCAT_STATIC_2D(1048576, 1, 1);
BM_NARY_CONCAT_STATIC_2D(1, 1048578, 0);
BM_NARY_CONCAT_STATIC_2D(1048578, 1, 1);

// Concat 2D with static concatenation dimension
#define BM_NARY_CONCAT_W_STATIC_CONCAT_DIM0_2D(M, N)                \
  BM_NARY_CONCAT(StaticConcatDim2D_##M##x##N##_dim0, 2, WRAP(M, N), \
                 WRAP(kStaticDim, kDynamicDim), 0)
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM0_2D(1, 1024);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM0_2D(1, 1026);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM0_2D(1, 1048576);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM0_2D(1, 1048578);
#define BM_NARY_CONCAT_W_STATIC_CONCAT_DIM1_2D(M, N)                \
  BM_NARY_CONCAT(StaticConcatDim2D_##M##x##N##_dim1, 2, WRAP(M, N), \
                 WRAP(kDynamicDim, kStaticDim), 1)
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM1_2D(1024, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM1_2D(1026, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM1_2D(1048576, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM1_2D(1048578, 1);

// Dynamic Concat 2D
#define BM_NARY_CONCAT_DYNAMIC_2D(M, N, CONCAT_DIM)                    \
  BM_NARY_CONCAT(Dynamic2D_##M##x##N##_dim##CONCAT_DIM, 2, WRAP(M, N), \
                 WRAP(kDynamicDim, kDynamicDim), CONCAT_DIM)
// Sqaure operands
BM_NARY_CONCAT_DYNAMIC_2D(512, 512, 0);
BM_NARY_CONCAT_DYNAMIC_2D(512, 512, 1);
BM_NARY_CONCAT_DYNAMIC_2D(514, 514, 0);
BM_NARY_CONCAT_DYNAMIC_2D(514, 514, 1);
BM_NARY_CONCAT_DYNAMIC_2D(1024, 1024, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1024, 1024, 1);
BM_NARY_CONCAT_DYNAMIC_2D(1026, 1026, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1026, 1026, 1);
// Slice operands
BM_NARY_CONCAT_DYNAMIC_2D(1, 1024, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1024, 1, 1);
BM_NARY_CONCAT_DYNAMIC_2D(1, 1026, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1026, 1, 1);
BM_NARY_CONCAT_DYNAMIC_2D(1, 1048576, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1048576, 1, 1);
BM_NARY_CONCAT_DYNAMIC_2D(1, 1048578, 0);
BM_NARY_CONCAT_DYNAMIC_2D(1048578, 1, 1);

// Static Concat 4D
#define BM_NARY_CONCAT_STATIC_4D(M, N, O, P, CONCAT_DIM)                     \
  BM_NARY_CONCAT(                                                            \
      Static4D_##M##x##N##x##O##x##P##_dim##CONCAT_DIM, 4, WRAP(M, N, O, P), \
      WRAP(kStaticDim, kStaticDim, kStaticDim, kStaticDim), CONCAT_DIM)
// Sqaure operands
BM_NARY_CONCAT_STATIC_4D(32, 32, 32, 32, 0);
BM_NARY_CONCAT_STATIC_4D(32, 32, 32, 32, 1);
BM_NARY_CONCAT_STATIC_4D(34, 34, 34, 34, 0);
BM_NARY_CONCAT_STATIC_4D(34, 34, 34, 34, 1);
BM_NARY_CONCAT_STATIC_4D(1024, 1024, 4, 4, 0);
BM_NARY_CONCAT_STATIC_4D(1024, 1024, 4, 4, 1);
BM_NARY_CONCAT_STATIC_4D(1026, 1026, 2, 6, 0);
BM_NARY_CONCAT_STATIC_4D(1026, 1026, 2, 6, 1);
// Slice operands
BM_NARY_CONCAT_STATIC_4D(32, 32, 1, 1024, 2);
BM_NARY_CONCAT_STATIC_4D(32, 32, 1024, 1, 3);
BM_NARY_CONCAT_STATIC_4D(34, 34, 1, 1026, 2);
BM_NARY_CONCAT_STATIC_4D(34, 34, 1026, 1, 3);
BM_NARY_CONCAT_STATIC_4D(4, 4, 1, 1048576, 2);
BM_NARY_CONCAT_STATIC_4D(4, 4, 1048576, 1, 3);
BM_NARY_CONCAT_STATIC_4D(2, 6, 1, 1048578, 2);
BM_NARY_CONCAT_STATIC_4D(2, 6, 1048578, 1, 3);

// Concat 4D with static concatenation dimension
#define BM_NARY_CONCAT_W_STATIC_CONCAT_DIM2_4D(M, N, O, P)          \
  BM_NARY_CONCAT(StaticConcatDim4D_##M##x##N##x##O##x##P##_dim0, 4, \
                 WRAP(M, N, O, P),                                  \
                 WRAP(kDynamicDim, kDynamicDim, kStaticDim, kDynamicDim), 2)
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM2_4D(32, 32, 1, 1024);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM2_4D(34, 34, 1, 1026);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM2_4D(4, 4, 1, 1048576);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM2_4D(2, 6, 1, 1048578);
#define BM_NARY_CONCAT_W_STATIC_CONCAT_DIM3_4D(M, N, O, P)          \
  BM_NARY_CONCAT(StaticConcatDim4D_##M##x##N##x##O##x##P##_dim1, 4, \
                 WRAP(M, N, O, P),                                  \
                 WRAP(kDynamicDim, kDynamicDim, kDynamicDim, kStaticDim), 3)
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM3_4D(32, 32, 1024, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM3_4D(34, 34, 1026, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM3_4D(4, 4, 1048576, 1);
BM_NARY_CONCAT_W_STATIC_CONCAT_DIM3_4D(2, 6, 1048578, 1);

// Dynamic Concat 4D
#define BM_NARY_CONCAT_DYNAMIC_4D(M, N, O, P, CONCAT_DIM)                     \
  BM_NARY_CONCAT(                                                             \
      Dynamic4D_##M##x##N##x##O##x##P##_dim##CONCAT_DIM, 4, WRAP(M, N, O, P), \
      WRAP(kDynamicDim, kDynamicDim, kDynamicDim, kDynamicDim), CONCAT_DIM)
// Sqaure operands
BM_NARY_CONCAT_DYNAMIC_4D(32, 32, 32, 32, 0);
BM_NARY_CONCAT_DYNAMIC_4D(32, 32, 32, 32, 1);
BM_NARY_CONCAT_DYNAMIC_4D(34, 34, 34, 34, 0);
BM_NARY_CONCAT_DYNAMIC_4D(34, 34, 34, 34, 1);
BM_NARY_CONCAT_DYNAMIC_4D(1024, 1024, 4, 4, 0);
BM_NARY_CONCAT_DYNAMIC_4D(1024, 1024, 4, 4, 1);
BM_NARY_CONCAT_DYNAMIC_4D(1026, 1026, 2, 6, 0);
BM_NARY_CONCAT_DYNAMIC_4D(1026, 1026, 2, 6, 1);
// Slice operands
BM_NARY_CONCAT_DYNAMIC_4D(32, 32, 1, 1024, 2);
BM_NARY_CONCAT_DYNAMIC_4D(32, 32, 1024, 1, 3);
BM_NARY_CONCAT_DYNAMIC_4D(34, 34, 1, 1026, 2);
BM_NARY_CONCAT_DYNAMIC_4D(34, 34, 1026, 1, 3);
BM_NARY_CONCAT_DYNAMIC_4D(4, 4, 1, 1048576, 2);
BM_NARY_CONCAT_DYNAMIC_4D(4, 4, 1048576, 1, 3);
BM_NARY_CONCAT_DYNAMIC_4D(2, 6, 1, 1048578, 2);
BM_NARY_CONCAT_DYNAMIC_4D(2, 6, 1048578, 1, 3);

}  // namespace
}  // namespace tensorflow
