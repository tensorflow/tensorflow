/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implements convolution operations with other kernels baked into the
// processing, to optimize latency and memory usage:
//  - Conv2D + BiasAdd + <Activation>
//  - Conv2D + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...
//
// Kernels for convolutions fused with image transformations (resize and mirror
// padding) defined in `conv_ops_fused_image_transform.cc`.

#define EIGEN_USE_THREADS

#include <string>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Type aliases for the unaligned tensors (tensor maps) used in output kernels.
template <typename T>
struct OutputTypes {
  // There is no guarantee that the output block passed to the output kernel
  // will be aligned.

  using Tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
                       Eigen::Unaligned>;

  using ConstTensor = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Unaligned>;
};

// Type alias for the tensor contraction output mapper.
template <typename Scalar, typename Index>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<Scalar, Index, Eigen::ColMajor>;

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

// TensorContraction swaps lhs with rhs, and changes layout from RowMajor
// (default in Tensorflow) to ColMajor (preferred in Eigen), and computes matmul
// using these tensors.
//
// TensorContraction output matrix (before reshape) has a ColMajor layout, and
// has dimensions:
//  - rows: output_channels
//  - cols: all other dimensions
//
// First element in every column is:
//   [batch ??, height ??, width ??, out_channel = i]
//
// We do not know what are the values of the 'batch', 'height', and 'width' here
// (if we know original dimensions, they can be computed from 'j').
//
// Each column of an output block is a continuous slice along the output channel
// dimension, so we can use it to efficiently compute any transformation that
// depends only on a channel value (e.g. add channel bias).

// Output kernel that fuses BiasAdd operation into the output of tensor
// contraction + any other transformation defined by Transform.
template <typename T, typename Transform = Identity>
struct BiasAddOutputKernel {
  explicit BiasAddOutputKernel(const T* bias_data) : bias_data(bias_data) {}

  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, Index>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename OutputTypes<T>::ConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename OutputTypes<T>::Tensor output(output_base, num_rows);
      const auto expr = output + bias;
      output = Transform::template apply<decltype(expr)>(expr);
    }
  }

 private:
  const T* bias_data;
};

// Output kernel that fuses FusedBatchNorm operation into the output of tensor
// contraction + any other transformation defined by Transform.
template <typename T, typename Transform = Identity>
struct FusedBatchNormOutputKernel {
  FusedBatchNormOutputKernel(T epsilon, const T* scaling_factor_data,
                             const T* offset_data, const T* estimated_mean_data)
      : epsilon(epsilon),
        scaling_factor_data(scaling_factor_data),
        offset_data(offset_data),
        estimated_mean_data(estimated_mean_data) {}

  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, Index>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename OutputTypes<T>::ConstTensor scaling_factor(scaling_factor_base,
                                                        num_rows);
    typename OutputTypes<T>::ConstTensor offset(offset_base, num_rows);
    typename OutputTypes<T>::ConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename OutputTypes<T>::Tensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = Transform::template apply<decltype(shifted)>(shifted);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
};

// Type aliases for the output kernels, purely for the sake of better launch
// dispatching code readability.
template <typename T>
using WithBiasAdd = BiasAddOutputKernel<T>;
template <typename T>
using WithBiasAddAndRelu = BiasAddOutputKernel<T, Relu>;
template <typename T>
using WithFusedBatchNorm = FusedBatchNormOutputKernel<T>;
template <typename T>
using WithFusedBatchNormAndRelu = FusedBatchNormOutputKernel<T, Relu>;

// Dispatch 2D convolution to the appropriate primitive operation:
//   (1) MatMul for the case of 1x1 convolution.
//   (2) MatMul for the case when filter size equals to the input size.
//   (3) General spatial 2D convolution for all other cases.
template <typename T>
class LaunchConv2DWithOutputKernel {
 public:
  LaunchConv2DWithOutputKernel(int row_stride, int col_stride,      //
                               int row_dilation, int col_dilation,  //
                               Padding padding)
      : row_stride_(row_stride),
        col_stride_(col_stride),
        row_dilation_(row_dilation),
        col_dilation_(col_dilation),
        padding_(padding) {}

  template <typename OutputKernel>
  void operator()(const OutputKernel& output_kernel, OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter, Tensor* output) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        row_stride_ == 1 && col_stride_ == 1) {
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair, output_kernel);

    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation_ == 1 &&
               col_dilation_ == 1 && padding_ == VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair,
          output_kernel);

    } else {
      functor::SpatialConvolution<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
          input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_, col_stride_,
          row_dilation_, col_dilation_, BrainPadding2EigenPadding(padding_),
          output_kernel);
    }
  }

 private:
  int row_stride_;
  int col_stride_;
  int row_dilation_;
  int col_dilation_;
  const Padding padding_;
};

}  // namespace

// Conv2D op with fused output kernels. Supports only CPUDevice.
template <typename T>
class FusedConv2DOp : public OpKernel {
 public:
  explicit FusedConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    // 'fused_ops' and 'num_args' attributes are specified by the Grappler
    // Remapper optimizer.

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused Conv2D must have at least one fused op."));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));

    // TODO(ezhulenev): Add support for fusion element-wise op chains defined
    // at runtime, e.g. Relu+Sqrt+Tanh+etc...

    // Match combination of fused ops to one of the supported fusions.
    if (FusedOpsMatches(fused_ops, {"BiasAdd"})) {
      fused_computation_ = FusedComputationType::kBiasAdd;
    } else if (FusedOpsMatches(fused_ops, {"BiasAdd", "Relu"})) {
      fused_computation_ = FusedComputationType::kBiasAddWithRelu;
    } else if (FusedOpsMatches(fused_ops, {"FusedBatchNorm"})) {
      fused_computation_ = FusedComputationType::kFusedBatchNorm;
    } else if (FusedOpsMatches(fused_ops, {"FusedBatchNorm", "Relu"})) {
      fused_computation_ = FusedComputationType::kFusedBatchNormWithRelu;
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        str_util::Join(fused_ops, ","), "]"));
    }

    // Depending on a picked fusion type validate fusion-specific arguments.

    if (fused_computation_ == FusedComputationType::kBiasAdd ||
        fused_computation_ == FusedComputationType::kBiasAddWithRelu) {
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    }

    if (fused_computation_ == FusedComputationType::kFusedBatchNorm ||
        fused_computation_ == FusedComputationType::kFusedBatchNormWithRelu) {
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument("Fused FusedBatchNorm must have four extra "
                                  "arguments: scale, offset, mean, variance."));
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    }
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "FusedConv2DWithBias: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    OP_REQUIRES(context, params_.data_format == FORMAT_NHWC,
                errors::Unimplemented("Fused conv implementation only supports "
                                      "NHWC tensor format for now."));
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));

    BiasAddArgs bias_add;
    FusedBatchNormArgs fused_batch_norm;

    LaunchConv2DWithOutputKernel<T> conv2d(
        dimensions.stride_rows, dimensions.stride_cols,
        dimensions.dilation_rows, dimensions.dilation_cols, params_.padding);

    switch (fused_computation_) {
      case FusedComputationType::kBiasAdd:
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add));
        conv2d(WithBiasAdd<T>(bias_add.bias_add_data), context, input, filter,
               output);
        break;

      case FusedComputationType::kBiasAddWithRelu:
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add));
        conv2d(WithBiasAddAndRelu<T>(bias_add.bias_add_data), context, input,
               filter, output);
        break;

      case FusedComputationType::kFusedBatchNorm:
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, &fused_batch_norm));
        conv2d(WithFusedBatchNorm<T>(epsilon_,
                                     fused_batch_norm.scaling_factor.data(),
                                     fused_batch_norm.offset_data,
                                     fused_batch_norm.estimated_mean_data),
               context, input, filter, output);
        break;

      case FusedComputationType::kFusedBatchNormWithRelu:
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, &fused_batch_norm));
        conv2d(WithFusedBatchNormAndRelu<T>(
                   epsilon_, fused_batch_norm.scaling_factor.data(),
                   fused_batch_norm.offset_data,
                   fused_batch_norm.estimated_mean_data),
               context, input, filter, output);
        break;
    }
  }

 private:
  bool FusedOpsMatches(const std::vector<string>& fused_ops,
                       const std::vector<string>& expected) const {
    return fused_ops == expected;
  }

  struct BiasAddArgs {
    const T* bias_add_data = nullptr;
  };

  struct FusedBatchNormArgs {
    const T* scale_data = nullptr;
    const T* offset_data = nullptr;
    const T* estimated_mean_data = nullptr;
    const T* estimated_variance_data = nullptr;

    // Precomputed expression:
    //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
    Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor;
  };

#define TF_REQUIRES(EXP, STATUS) \
  if (!TF_PREDICT_TRUE(EXP)) return (STATUS)

  void InitDataPtr(const Tensor& tensor, const T** ptr) const {
    *ptr = reinterpret_cast<const T*>(tensor.tensor_data().data());
  }

  Status InitBiasAddArgs(OpKernelContext* context, BiasAddArgs* args) const {
    // Bias of the following dimensions: [ output_depth ]
    const Tensor& bias = context->input(2);

    TF_REQUIRES(bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional",
                                        bias.shape().DebugString()));

    InitDataPtr(bias, &args->bias_add_data);

    return Status::OK();
  }

  Status InitFusedBatchNormArgs(OpKernelContext* context,
                                FusedBatchNormArgs* args) const {
    const Tensor& scale = context->input(2);
    const Tensor& offset = context->input(3);
    const Tensor& estimated_mean = context->input(4);
    const Tensor& estimated_variance = context->input(5);

    TF_REQUIRES(scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    TF_REQUIRES(offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    TF_REQUIRES(estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    TF_REQUIRES(
        estimated_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_variance.shape().DebugString()));

    InitDataPtr(scale, &args->scale_data);
    InitDataPtr(offset, &args->offset_data);
    InitDataPtr(estimated_mean, &args->estimated_mean_data);
    InitDataPtr(estimated_variance, &args->estimated_variance_data);

    // Precompute scaling factor once for all output blocks (kernels).
    args->scaling_factor =
        (estimated_variance.flat<T>() + static_cast<T>(epsilon_)).rsqrt() *
        scale.flat<T>();

    return Status::OK();
  }

#undef TF_REQUIRES

  // Element-wise ops applied to the result of Conv2D.
  // TODO(ezhulenev): Add support for runtime-defined op chains.
  enum class FusedComputationType {
    kBiasAdd,
    kBiasAddWithRelu,
    kFusedBatchNorm,
    kFusedBatchNormWithRelu
  };

  Conv2DParameters params_;
  FusedComputationType fused_computation_;

  // FusedBatchNorm attributes.
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedConv2DOp);
};

#define REGISTER_FUSED_CONV2D(T)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<T>);

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
// TODO(b/119765980): Upgrade upstream Eigen to set `m_can_use_xsmm=false` for
// contractions with non-default contraction output kernels.
#if !defined(USE_GEMM_FOR_CONV) && !defined(EIGEN_USE_LIBXSMM)
TF_CALL_float(REGISTER_FUSED_CONV2D);
TF_CALL_double(REGISTER_FUSED_CONV2D);
#endif  // !USE_GEMM_FOR_CONV

}  // namespace tensorflow
