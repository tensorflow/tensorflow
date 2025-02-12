/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Output kernels for fusing computation into Eigen Tensor contractions:
//   (1) FusedConv2DOp
//   (2) FusedMatMulOp
//
// Supported fused computations:
//   (1) {Conv2D/MatMul} + BiasAdd + <Activation>
//   (2) {Conv2D/MatMul} + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...

#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_

#include <type_traits>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

enum class FusedComputationType {
  kUndefined,
  kBiasAdd,
  kBiasAddWithRelu,
  kBiasAddWithRelu6,
  kBiasAddWithTanh,
  kBiasAddWithSigmoid,
  kBiasAddWithElu,
  kBiasAddWithLeakyRelu,
  kBiasAddWithGeluApproximate,
  kBiasAddWithGeluExact,
  kFusedBatchNorm,
  kFusedBatchNormWithRelu,
  kFusedBatchNormWithRelu6,
  kFusedBatchNormWithElu,
  kFusedBatchNormWithLeakyRelu
};

// We have to pass around additional arguments for all possible fusion types.
struct FusedComputationArgs {
  float epsilon = 0.0;          // Used by `FusedBatchNorm` fusion only
  float leakyrelu_alpha = 0.0;  // Used by `LeakyRelu` fusion only
};

struct FusedComputationPattern {
  FusedComputationType fused_computation;
  std::vector<string> fused_ops;
};

// Parse attributes from the kernel construction context, and verifies that they
// specify valid fused computation pattern.
absl::Status InitializeFusedComputation(
    OpKernelConstruction* context, const string& kernel_name,
    const std::vector<FusedComputationPattern>& patterns,
    FusedComputationType* fused_computation,
    FusedComputationArgs* fused_computation_args);

// Type alias for the tensor contraction output mapper.
template <typename Scalar, typename StorageIndex>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<Scalar, StorageIndex, Eigen::ColMajor>;

// Returns input expression without any transformations.
struct Identity {
  template <typename XprType>
  static auto apply(XprType expr) -> XprType {
    return expr;
  };
};

// Applies `Relu` to the passed input expression.
struct Relu {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0));
  };
};

// Applies `Relu6` to the passed input expression.
struct Relu6 {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())
                      .cwiseMin(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0))
        .cwiseMin(static_cast<typename XprType::Scalar>(6));
  };
};

// Applies `Tanh` to the passed input expression.
struct Tanh {
  template <typename XprType>
  static auto apply(XprType expr) -> decltype(expr.tanh()) {
    return expr.tanh();
  };
};

// Applies `Sigmoid` to the passed input expression.
struct Sigmoid {
  template <typename XprType>
  static auto apply(XprType expr) -> decltype(expr.sigmoid()) {
    return expr.sigmoid();
  };
};

// Applies `Elu` to the passed input expression.
struct Elu {
  template <typename XprType>
  static auto apply(XprType expr) -> decltype(
      (expr < std::declval<typename XprType::Scalar>())
          .select(expr.exp() -
                      expr.constant(std::declval<typename XprType::Scalar>()),
                  expr)) {
    return (expr < static_cast<typename XprType::Scalar>(0))
        .select(expr.exp() -
                    expr.constant(static_cast<typename XprType::Scalar>(1)),
                expr);
  };
};

// Applies `LeakyRelu` to the passed input expression.
struct LeakyRelu {
  template <typename XprType>
  static auto apply(XprType expr, const float leakyrelu_alpha) -> decltype(
      (expr < std::declval<typename XprType::Scalar>())
          .select(expr *
                      expr.constant(std::declval<typename XprType::Scalar>()),
                  expr)) {
    return (expr < static_cast<typename XprType::Scalar>(0))
        .select(expr * expr.constant(static_cast<typename XprType::Scalar>(
                           leakyrelu_alpha)),
                expr);
  };
};

template <typename T>
struct BiasAddArgs {
  const T* bias_add_data = nullptr;
  float leakyrelu_alpha;

  static bool IsSupported(FusedComputationType fusion) {
    return fusion == FusedComputationType::kBiasAdd ||
           fusion == FusedComputationType::kBiasAddWithRelu ||
           fusion == FusedComputationType::kBiasAddWithRelu6 ||
           fusion == FusedComputationType::kBiasAddWithTanh ||
           fusion == FusedComputationType::kBiasAddWithSigmoid ||
           fusion == FusedComputationType::kBiasAddWithElu ||
           fusion == FusedComputationType::kBiasAddWithLeakyRelu;
  }
};

template <typename T>
struct FusedBatchNormArgs {
  const T* scale_data = nullptr;
  const T* offset_data = nullptr;
  const T* estimated_mean_data = nullptr;
  const T* estimated_variance_data = nullptr;

  // Precomputed expression:
  //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
  Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor;

  float leakyrelu_alpha;

  static bool IsSupported(FusedComputationType fusion) {
    return fusion == FusedComputationType::kFusedBatchNorm ||
           fusion == FusedComputationType::kFusedBatchNormWithRelu ||
           fusion == FusedComputationType::kFusedBatchNormWithRelu6 ||
           fusion == FusedComputationType::kFusedBatchNormWithElu ||
           fusion == FusedComputationType::kFusedBatchNormWithLeakyRelu;
  }
};

// TensorContraction swaps lhs with rhs, and changes layout from RowMajor
// (default in Tensorflow) to ColMajor (preferred in Eigen), and computes matmul
// using these tensors.
//
// (1) Spatial Convolution (see eigen_spatial_convolutions.h):
//
//   TensorContraction output matrix (before reshape) has a ColMajor layout, and
//   has dimensions:
//   - rows: output_channels
//   - cols: all other dimensions
//
//   First element in every column is:
//     [batch ??, height ??, width ??, out_channel = i]
//
//   We do not know what are the values of the 'batch', 'height', and 'width'
//   here (if we know original dimensions, they can be computed from 'j').
//
//   Each column of an output block is a continuous slice along the output
//   channel dimension, so we can use it to efficiently compute any
//   transformation that depends only on a channel value (e.g. add channel
//   bias).
//
// (2) Matrix Multiplication (see matmul_op.cc):
//
//   For the `MxK * KxN` matrix multiplication, output matrix has a `MxN`
//   dimensions. Each column in output block is a slice of the innermost
//   dimension of the output matrix starting at offset 'i'.
//
//   Example: In Tensorflow MatMul [8x32] * [32x64], each output block column
//   will correspond to MatMul output row of size 64 (because Tensorflow uses
//   row major storage order).

// Output kernel that fuses BiasAdd operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct BiasAddOutputKernel {
  explicit BiasAddOutputKernel(const BiasAddArgs<T>& args)
      : bias_data(args.bias_add_data) {}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename TTypes<T>::UnalignedConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      Scalar* output_base = &output_mapper(0, col);
      typename TTypes<Scalar>::UnalignedTensor output(output_base, num_rows);
      if constexpr (std::is_same_v<Scalar, T>) {
        const auto expr = output + bias;
        output = Activation::template apply<decltype(expr)>(expr);
      } else {
        const auto bias_expr = bias.template cast<Scalar>();
        const auto expr = output + bias_expr;
        output = Activation::template apply<decltype(expr)>(expr);
      }
    }
  }

 private:
  const T* bias_data;
};

template <typename T>
struct BiasAddOutputKernel<T, LeakyRelu> {
  explicit BiasAddOutputKernel(const BiasAddArgs<T>& args)
      : bias_data(args.bias_add_data), leakyrelu_alpha(args.leakyrelu_alpha) {}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename TTypes<T>::UnalignedConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      Scalar* output_base = &output_mapper(0, col);
      typename TTypes<Scalar>::UnalignedTensor output(output_base, num_rows);
      if constexpr (std::is_same_v<Scalar, T>) {
        const auto expr = output + bias;
        output =
            LeakyRelu::template apply<decltype(expr)>(expr, leakyrelu_alpha);
      } else {
        const auto bias_expr = bias.template cast<Scalar>();
        const auto expr = output + bias_expr;
        output =
            LeakyRelu::template apply<decltype(expr)>(expr, leakyrelu_alpha);
      }
    }
  }

 private:
  const T* bias_data;
  float leakyrelu_alpha;
};

// Output kernel that fuses FusedBatchNorm operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct FusedBatchNormOutputKernel {
  FusedBatchNormOutputKernel(T epsilon, const FusedBatchNormArgs<T>& args)
      : epsilon(epsilon),
        scaling_factor_data(args.scaling_factor.data()),
        offset_data(args.offset_data),
        estimated_mean_data(args.estimated_mean_data) {}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename TTypes<T>::UnalignedConstTensor scaling_factor(scaling_factor_base,
                                                            num_rows);
    typename TTypes<T>::UnalignedConstTensor offset(offset_base, num_rows);
    typename TTypes<T>::UnalignedConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = Activation::template apply<decltype(shifted)>(shifted);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
};

template <typename T>
struct FusedBatchNormOutputKernel<T, LeakyRelu> {
  FusedBatchNormOutputKernel(T epsilon, const FusedBatchNormArgs<T>& args)
      : epsilon(epsilon),
        scaling_factor_data(args.scaling_factor.data()),
        offset_data(args.offset_data),
        estimated_mean_data(args.estimated_mean_data),
        leakyrelu_alpha(args.leakyrelu_alpha) {}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename TTypes<T>::UnalignedConstTensor scaling_factor(scaling_factor_base,
                                                            num_rows);
    typename TTypes<T>::UnalignedConstTensor offset(offset_base, num_rows);
    typename TTypes<T>::UnalignedConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = LeakyRelu::template apply<decltype(shifted)>(shifted,
                                                            leakyrelu_alpha);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
  float leakyrelu_alpha;
};

// Type aliases for the output kernels, purely for the sake of better launch
// dispatching code readability.
template <typename T>
using WithBiasAdd = BiasAddOutputKernel<T>;
template <typename T>
using WithBiasAddAndRelu = BiasAddOutputKernel<T, Relu>;
template <typename T>
using WithBiasAddAndRelu6 = BiasAddOutputKernel<T, Relu6>;
template <typename T>
using WithBiasAddAndTanh = BiasAddOutputKernel<T, Tanh>;
template <typename T>
using WithBiasAddAndSigmoid = BiasAddOutputKernel<T, Sigmoid>;
template <typename T>
using WithBiasAddAndElu = BiasAddOutputKernel<T, Elu>;
template <typename T>
using WithBiasAddAndLeakyRelu = BiasAddOutputKernel<T, LeakyRelu>;
template <typename T>
using WithFusedBatchNorm = FusedBatchNormOutputKernel<T>;
template <typename T>
using WithFusedBatchNormAndRelu = FusedBatchNormOutputKernel<T, Relu>;
template <typename T>
using WithFusedBatchNormAndRelu6 = FusedBatchNormOutputKernel<T, Relu6>;
template <typename T>
using WithFusedBatchNormAndElu = FusedBatchNormOutputKernel<T, Elu>;
template <typename T>
using WithFusedBatchNormAndLeakyRelu = FusedBatchNormOutputKernel<T, LeakyRelu>;

template <typename T>
absl::Status InitBiasAddArgs(OpKernelContext* context, BiasAddArgs<T>* args,
                             const float* leakyrelu_alpha = nullptr) {
  // Bias of the following dimensions: [ output_depth ]
  const Tensor& bias = context->input(2);

  if (bias.dims() != 1)
    return errors::InvalidArgument("bias must be 1-dimensional",
                                   bias.shape().DebugString());

  const auto data_ptr = [](const Tensor& tensor) -> const T* {
    return reinterpret_cast<const T*>(tensor.tensor_data().data());
  };

  args->bias_add_data = data_ptr(bias);

  if (leakyrelu_alpha) {
    args->leakyrelu_alpha = *leakyrelu_alpha;
  }

  return absl::OkStatus();
}

template <typename T>
absl::Status InitFusedBatchNormArgs(OpKernelContext* context, float epsilon,
                                    FusedBatchNormArgs<T>* args,
                                    const float* leakyrelu_alpha = nullptr) {
  const Tensor& scale = context->input(2);
  const Tensor& offset = context->input(3);
  const Tensor& estimated_mean = context->input(4);
  const Tensor& estimated_variance = context->input(5);

  if (scale.dims() != 1)
    return errors::InvalidArgument("scale must be 1-dimensional",
                                   scale.shape().DebugString());
  if (offset.dims() != 1)
    return errors::InvalidArgument("offset must be 1-dimensional",
                                   offset.shape().DebugString());
  if (estimated_mean.dims() != 1)
    return errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                   estimated_mean.shape().DebugString());
  if (estimated_variance.dims() != 1)
    return errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                   estimated_variance.shape().DebugString());

  const auto data_ptr = [](const Tensor& tensor) -> const T* {
    return reinterpret_cast<const T*>(tensor.tensor_data().data());
  };

  args->scale_data = data_ptr(scale);
  args->offset_data = data_ptr(offset);
  args->estimated_mean_data = data_ptr(estimated_mean);
  args->estimated_variance_data = data_ptr(estimated_variance);

  // Precompute scaling factor once for all output blocks (kernels).
  args->scaling_factor =
      (estimated_variance.flat<T>() + static_cast<T>(epsilon)).rsqrt() *
      scale.flat<T>();

  if (leakyrelu_alpha) {
    args->leakyrelu_alpha = *leakyrelu_alpha;
  }

  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_
