/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/pooling_ops_common.h"

#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#if TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#endif
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace {

template <typename T>
struct RawType {
  using type = T;
};

template <>
struct RawType<qint8> {
  using type = int8;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
struct PadInputWithNegativeInf {
  Status operator()(const GPUDevice& d,
                    typename TTypes<T, 4, int>::ConstTensor in,
                    int input_pad_top, int input_pad_bottom, int input_pad_left,
                    int input_pad_right, typename TTypes<T, 4, int>::Tensor out,
                    TensorFormat format) {
    T padding_value = -std::numeric_limits<T>::infinity();
    functor::PadInput<GPUDevice, T, int, 4>()(
        d, in, {{input_pad_top, input_pad_left}},
        {{input_pad_bottom, input_pad_right}}, out, format, padding_value);
    return OkStatus();
  }
};

template <>
struct PadInputWithNegativeInf<qint8> {
  Status operator()(const GPUDevice& d,
                    typename TTypes<qint8, 4, int>::ConstTensor in,
                    int input_pad_top, int input_pad_bottom, int input_pad_left,
                    int input_pad_right,
                    typename TTypes<qint8, 4, int>::Tensor out,
                    TensorFormat format) {
    return errors::InvalidArgument(
        "Explicit padding not yet supported with qint8");
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace

Status CheckPaddingSize(int64_t window_rows, int64_t window_cols,
                        int64_t pad_top, int64_t pad_bottom, int64_t pad_left,
                        int64_t pad_right) {
  if (!FastBoundsCheck(pad_top, window_rows)) {
    return errors::InvalidArgument("Top padding ", pad_top,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_rows);
  }
  if (!FastBoundsCheck(pad_bottom, window_rows)) {
    return errors::InvalidArgument("Bottom padding ", pad_bottom,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_rows);
  }
  if (!FastBoundsCheck(pad_left, window_cols)) {
    return errors::InvalidArgument("Left padding ", pad_left,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_cols);
  }
  if (!FastBoundsCheck(pad_right, window_cols)) {
    return errors::InvalidArgument("Right padding ", pad_right,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_cols);
  }
  return OkStatus();
}

PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding,
                               std::vector<int64_t> explicit_paddings,
                               TensorFormat data_format,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 2 spatial dimensions.
  // Note: the total number of dimensions could be 4 for NHWC, NCHW,
  // or 5 for NCHW_VECT_C.
  OP_REQUIRES(context,
              GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
              errors::InvalidArgument(
                  "tensor_in_shape must have 2 spatial dimensions. ",
                  tensor_in_shape.dims(), " ", data_format));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
          (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));
  if (padding == Padding::EXPLICIT) {
    OP_REQUIRES_OK(context, CheckValidPadding(padding, explicit_paddings,
                                              /*num_dims=*/4, data_format));
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &pad_top,
                             &pad_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &pad_left,
                             &pad_right);
    OP_REQUIRES_OK(context, CheckPaddingSize(window_rows, window_cols, pad_top,
                                             pad_bottom, pad_left, pad_right));
  }

  if (depth_window == 1) {
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));
    pad_depth = 0;
    out_depth = depth;
  } else {
    OP_REQUIRES(context, depth_window > 0,
                errors::InvalidArgument("depth_window must not be 0"));
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the
    // depth_stride (no overlapping).
    OP_REQUIRES(
        context, depth % depth_window == 0,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to evenly divide the input depth"));
    OP_REQUIRES(
        context, depth_stride == depth_window,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to equal the depth stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    pad_depth = 0;
    out_depth = depth / depth_window;
  }
}

TensorShape PoolParameters::forward_output_shape() {
  if (depth_window == 1) {
    // Spatial pooling
    return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                           depth);
  } else {
    // Depthwise pooling
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void DnnPoolingOp<T>::Compute(OpKernelContext* context,
                              se::dnn::PoolingMode pooling_mode,
                              const std::vector<int32>& size,
                              const std::vector<int32>& stride, Padding padding,
                              std::vector<int64_t> explicit_paddings,
                              TensorFormat data_format, const Tensor& tensor_in,
                              const TensorShape& tensor_out_shape,
                              bool propagate_nans) {
  Tensor* tensor_out = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_out_shape, &tensor_out));
  if (tensor_in.shape().num_elements() == 0) {
    return;
  }

  PoolParameters params{
      context,           size,        stride,           padding,
      explicit_paddings, data_format, tensor_in.shape()};
  if (!context->status().ok()) {
    return;
  }

  int batch_size = params.tensor_in_batch;
  int depth = params.depth;
  int tensor_in_cols = params.tensor_in_cols;
  int tensor_in_rows = params.tensor_in_rows;

#if CUDNN_VERSION < 7300
  /// Earlier versions do not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, tensor_in.shape(),
                                                data_format),
                                &transformed_input));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                           tensor_in.tensor<T, 4>(),
                                           transformed_input.tensor<T, 4>());
  } else {
    transformed_input = tensor_in;
  }
  Tensor transformed_output;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, tensor_out_shape,
                                                data_format),
                                &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  se::dnn::DataLayout data_layout = se::dnn::DataLayout::kBatchDepthYX;
#else
  Tensor transformed_input = tensor_in;
  auto& transformed_output = *tensor_out;
  se::dnn::DataLayout data_layout;
  switch (data_format) {
    case FORMAT_NHWC:
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      break;
    case FORMAT_NCHW:
      data_layout = se::dnn::DataLayout::kBatchDepthYX;
      break;
    case FORMAT_NCHW_VECT_C:
      // NCHW_VECT_C is not supported by cudnnPoolingForward(), but can be
      // emulated via NHWC.
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      batch_size *= depth / 4;
      depth = 4;
      break;
    default:
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unsupported format: ",
                                          ToString(data_format)));
  }
#endif

  int64_t vertical_padding = params.pad_top;
  int64_t horizontal_padding = params.pad_left;

  if (padding == EXPLICIT && (params.pad_top != params.pad_bottom ||
                              params.pad_left != params.pad_right)) {
    // cuDNN only supports padding the same amount on the left and right sides,
    // and on the top and bottom sides. So we manually create a new padded
    // input tensor such that we can pass it to cuDNN.
    const int64_t common_padding_rows =
        std::min(params.pad_top, params.pad_bottom);
    const int64_t common_padding_cols =
        std::min(params.pad_left, params.pad_right);

    Tensor padded_input;
    const int64_t padding_rows_diff =
        std::abs(params.pad_top - params.pad_bottom);
    const int64_t padding_cols_diff =
        std::abs(params.pad_left - params.pad_right);

    const int64_t new_in_rows = tensor_in_rows + padding_rows_diff;
    const int64_t new_in_cols = tensor_in_cols + padding_cols_diff;

    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value,
                               ShapeFromFormat(data_format, batch_size,
                                               new_in_rows, new_in_cols, depth),
                               &padded_input));
    const int64_t input_pad_top = params.pad_top - common_padding_rows;
    const int64_t input_pad_bottom = params.pad_bottom - common_padding_rows;
    const int64_t input_pad_left = params.pad_left - common_padding_cols;
    const int64_t input_pad_right = params.pad_right - common_padding_cols;

    bool in_bounds =
        FastBoundsCheck(input_pad_top, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_bottom, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_left, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_right, std::numeric_limits<int>::max());
    if (!in_bounds) {
      context->SetStatus(errors::InvalidArgument("Padding is too large."));
      return;
    }

    // We need to call the const version of transformed_input.tensor()
    const Tensor& const_transformed_input = transformed_input;
    OP_REQUIRES_OK(
        context,
        PadInputWithNegativeInf<T>()(
            context->eigen_device<GPUDevice>(),
            To32Bit(const_transformed_input.tensor<T, 4>()),
            static_cast<int>(input_pad_top), static_cast<int>(input_pad_bottom),
            static_cast<int>(input_pad_left), static_cast<int>(input_pad_right),
            To32Bit(padded_input.tensor<T, 4>()), data_format));
    transformed_input = padded_input;
    vertical_padding = common_padding_rows;
    horizontal_padding = common_padding_cols;
    tensor_in_rows = new_in_rows;
    tensor_in_cols = new_in_cols;
  }

  se::dnn::PoolingDescriptor pooling_desc;
  pooling_desc.set_pooling_mode(pooling_mode)
      .set_window_height(params.window_rows)
      .set_window_width(params.window_cols)
      .set_vertical_stride(params.row_stride)
      .set_horizontal_stride(params.col_stride)
      .set_vertical_padding(vertical_padding)
      .set_horizontal_padding(horizontal_padding)
      .set_propagate_nans(propagate_nans);

  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(batch_size)
      .set_height(tensor_in_rows)
      .set_width(tensor_in_cols)
      .set_feature_map_count(depth)
      .set_layout(data_layout);

  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(batch_size)
      .set_height(params.out_height)
      .set_width(params.out_width)
      .set_feature_map_count(depth)
      .set_layout(data_layout);

  auto input_data =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         transformed_input.template flat<T>().data()),
                     transformed_input.template flat<T>().size());

  auto output_data =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         transformed_output.template flat<T>().data()),
                     transformed_output.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  OP_REQUIRES_OK(context, stream->ThenPoolForward(
                              pooling_desc, input_desc, input_data, output_desc,
                              &output_data, &scratch_allocator));
#else
  OP_REQUIRES_OK(context,
                 stream->ThenPoolForward(pooling_desc, input_desc, input_data,
                                         output_desc, &output_data));
#endif

#if CUDNN_VERSION < 7300
  if (data_format == FORMAT_NHWC) {
    /// Transform the output data from NCHW back to NHWC
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    using RT = typename RawType<T>::type;
    functor::NCHWToNHWC<GPUDevice, RT, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(transformed_output).template tensor<RT, 4>(),
        tensor_out->tensor<RT, 4>());
  }
#endif
}

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 4>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,   \
      const std::array<int, 2>& padding_left,                           \
      const std::array<int, 2>& padding_right,                          \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format, \
      const T& padding_value);                                          \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(int32);
}  // namespace functor

template <typename T>
void DnnPoolingGradOp<T>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::vector<int32>& size, const std::vector<int32>& stride,
    Padding padding, std::vector<int64_t> explicit_paddings,
    TensorFormat data_format, const Tensor* tensor_in, const Tensor* tensor_out,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape,
    bool propagate_nans) {
  CHECK((pooling_mode != se::dnn::PoolingMode::kMaximum) ||
        (tensor_in && tensor_out))
      << "For MaxPoolGrad, both tensor_in and tensor_out needs to be "
         "specified";

  Tensor* input_backprop = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_in_shape, &input_backprop));
  if (tensor_in_shape.num_elements() == 0) {
    return;
  }

  PoolParameters params{context,           size,        stride,         padding,
                        explicit_paddings, data_format, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }
  if (tensor_out) {
    OP_REQUIRES(context, tensor_out->shape() == params.forward_output_shape(),
                errors::InvalidArgument("Expected orig_output shape to be ",
                                        params.forward_output_shape(),
                                        ", but got ", tensor_out->shape()));
  }
  OP_REQUIRES(context, out_backprop.shape() == params.forward_output_shape(),
              errors::InvalidArgument("Expected grad shape to be ",
                                      params.forward_output_shape(),
                                      ", but got ", out_backprop.shape()));

  TensorFormat transformed_input_data_format = data_format;

#if CUDNN_VERSION < 7300
  /// For now, cudnn does not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  TensorShape transformed_input_shape;
  if (data_format == FORMAT_NHWC || !tensor_in) {
    transformed_input_shape =
        ShapeFromFormat(FORMAT_NCHW, tensor_in_shape, data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  TensorShape transformed_output_shape;
  if (data_format == FORMAT_NHWC || !tensor_out) {
    transformed_output_shape =
        ShapeFromFormat(FORMAT_NCHW, out_backprop.shape(), data_format);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_output_shape,
                                                   &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  Tensor transformed_input_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_input_shape,
                                          &transformed_input_backprop));
  } else {
    transformed_input_backprop = *input_backprop;
  }
  Tensor transformed_output_backprop;
  if (data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          transformed_output_shape,
                                          &transformed_output_backprop));
  } else {
    transformed_output_backprop = out_backprop;
  }

  if (data_format == FORMAT_NHWC) {
    /// Convert the data from NHWC to NCHW if necessary.
    if (tensor_in) {
      // For AvgPoolGrad, the original input tensor is not necessary. However,
      // cudnn still requires them to run, although they do not affect the
      // results.
      functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                             tensor_in->tensor<T, 4>(),
                                             transformed_input.tensor<T, 4>());
      transformed_input_data_format = FORMAT_NCHW;
    }
    if (tensor_out) {
      // For AvgPoolGrad, the original output tensor is not necessary. However,
      // cudnn still requires them to run, although they do not affect the
      // results.
      functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<Device>(),
                                             tensor_out->tensor<T, 4>(),
                                             transformed_output.tensor<T, 4>());
    }
    functor::NHWCToNCHW<GPUDevice, T, 4>()(
        context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
        transformed_output_backprop.tensor<T, 4>());
  }
  se::dnn::DataLayout data_layout = se::dnn::DataLayout::kBatchDepthYX;
#else
  Tensor transformed_input;
  if (!tensor_in) {
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          tensor_in_shape, &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  if (!tensor_out) {
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   out_backprop.shape(),
                                                   &transformed_output));
  } else {
    transformed_output = *tensor_out;
  }
  Tensor transformed_input_backprop = *input_backprop;
  Tensor transformed_output_backprop = out_backprop;
  se::dnn::DataLayout data_layout;
  switch (data_format) {
    case FORMAT_NHWC:
      data_layout = se::dnn::DataLayout::kBatchYXDepth;
      break;
    case FORMAT_NCHW:
      data_layout = se::dnn::DataLayout::kBatchDepthYX;
      break;
    default:
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unsupported format: ",
                                          ToString(data_format)));
  }
#endif  // CUDNN_VERSION < 7300

  int64_t vertical_padding = params.pad_top;
  int64_t horizontal_padding = params.pad_left;

  int batch_size = params.tensor_in_batch;
  int depth = params.depth;
  int tensor_in_cols = params.tensor_in_cols;
  int tensor_in_rows = params.tensor_in_rows;

  int64_t input_pad_top = 0;
  int64_t input_pad_bottom = 0;
  int64_t input_pad_left = 0;
  int64_t input_pad_right = 0;

  Tensor transformed_and_padded_input_backprop;

  if (padding == EXPLICIT && (params.pad_top != params.pad_bottom ||
                              params.pad_left != params.pad_right)) {
    // Pad the input in the same way we did during the forward pass, so that
    // cuDNN or MIOpen receives the same input during the backward pass function
    // as it did during the forward pass function.
    const int64_t common_padding_rows =
        std::min(params.pad_top, params.pad_bottom);
    const int64_t common_padding_cols =
        std::min(params.pad_left, params.pad_right);

    Tensor padded_input;
    const int64_t padding_rows_diff =
        std::abs(params.pad_top - params.pad_bottom);
    const int64_t padding_cols_diff =
        std::abs(params.pad_left - params.pad_right);

    const int64_t new_in_rows = tensor_in_rows + padding_rows_diff;
    const int64_t new_in_cols = tensor_in_cols + padding_cols_diff;

    VLOG(2) << "Create new tensor: "
            << " original rows=" << tensor_in_rows
            << " original cols=" << tensor_in_cols
            << " padding_rows=" << new_in_rows
            << " padding_cols=" << new_in_cols << " depth= " << depth
            << " batch_size=" << batch_size << " kernel_rows"
            << params.window_rows << " kernel_col" << params.window_cols
            << " stride_rows" << params.row_stride;

    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     ShapeFromFormat(transformed_input_data_format, batch_size,
                                     new_in_rows, new_in_cols, depth),
                     &padded_input));

    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     ShapeFromFormat(transformed_input_data_format, batch_size,
                                     new_in_rows, new_in_cols, depth),
                     &transformed_and_padded_input_backprop));

    input_pad_top = params.pad_top - common_padding_rows;
    input_pad_bottom = params.pad_bottom - common_padding_rows;
    input_pad_left = params.pad_left - common_padding_cols;
    input_pad_right = params.pad_right - common_padding_cols;

    bool in_bounds =
        FastBoundsCheck(input_pad_top, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_bottom, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_left, std::numeric_limits<int>::max()) &&
        FastBoundsCheck(input_pad_right, std::numeric_limits<int>::max());
    if (!in_bounds) {
      context->SetStatus(errors::InvalidArgument("Padding is too large."));
      return;
    }

    // PadInputWithNegativeInf functor requires input to be a const.
    const Tensor& const_transformed_input = transformed_input;
    OP_REQUIRES_OK(
        context,
        PadInputWithNegativeInf<T>()(
            context->eigen_device<GPUDevice>(),
            To32Bit(const_transformed_input.tensor<T, 4>()),
            static_cast<int>(input_pad_top), static_cast<int>(input_pad_bottom),
            static_cast<int>(input_pad_left), static_cast<int>(input_pad_right),
            To32Bit(padded_input.tensor<T, 4>()),
            transformed_input_data_format));

    transformed_input = padded_input;

    vertical_padding = common_padding_rows;
    horizontal_padding = common_padding_cols;
    VLOG(2) << "vertical padding set to: " << vertical_padding
            << " horizontal padding set to: " << horizontal_padding;
    tensor_in_rows = new_in_rows;
    tensor_in_cols = new_in_cols;
  } else {
    transformed_and_padded_input_backprop = transformed_input_backprop;
  }

  /// Get ready to call cudnn
  se::dnn::PoolingDescriptor pooling_desc;
  pooling_desc.set_pooling_mode(pooling_mode)
      .set_window_height(params.window_rows)
      .set_window_width(params.window_cols)
      .set_vertical_stride(params.row_stride)
      .set_horizontal_stride(params.col_stride)
      .set_vertical_padding(vertical_padding)
      .set_horizontal_padding(horizontal_padding)
      .set_propagate_nans(propagate_nans);

  se::dnn::BatchDescriptor orig_output_desc;
  orig_output_desc.set_count(params.tensor_in_batch)
      .set_height(params.out_height)
      .set_width(params.out_width)
      .set_feature_map_count(params.depth)
      .set_layout(data_layout);

  se::dnn::BatchDescriptor orig_input_desc;
  orig_input_desc.set_count(params.tensor_in_batch)
      .set_height(tensor_in_rows)
      .set_width(tensor_in_cols)
      .set_feature_map_count(params.depth)
      .set_layout(data_layout);

  auto orig_output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());
  auto orig_input_data =
      AsDeviceMemory(transformed_input.template flat<T>().data(),
                     transformed_input.template flat<T>().size());
  auto output_backprop_data =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop_data = AsDeviceMemory(
      transformed_and_padded_input_backprop.template flat<T>().data(),
      transformed_and_padded_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  OP_REQUIRES_OK(context,
                 stream->ThenPoolBackward(
                     pooling_desc, orig_input_desc, orig_input_data,
                     orig_output_desc, orig_output_data, output_backprop_data,
                     &input_backprop_data, &scratch_allocator));
#else
  OP_REQUIRES_OK(context, stream->ThenPoolBackward(
                              pooling_desc, orig_input_desc, orig_input_data,
                              orig_output_desc, orig_output_data,
                              output_backprop_data, &input_backprop_data));
#endif

  if (padding == EXPLICIT && (params.pad_top != params.pad_bottom ||
                              params.pad_left != params.pad_right)) {
    // Remove the padding that was added to the input shape above.
    functor::PadInput<GPUDevice, T, int, 4>()(
        context->eigen_device<GPUDevice>(),
        To32Bit(const_cast<const Tensor&>(transformed_and_padded_input_backprop)
                    .tensor<T, 4>()),
        {{static_cast<int>(-input_pad_top), static_cast<int>(-input_pad_left)}},
        {{static_cast<int>(-input_pad_bottom),
          static_cast<int>(-input_pad_right)}},
        To32Bit(transformed_input_backprop.template tensor<T, 4>()),
        transformed_input_data_format, T{});
  }

#if CUDNN_VERSION < 7300
  if (data_format == FORMAT_NHWC) {
    /// Transform the output data from NCHW back to NHWC.
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(transformed_input_backprop).template tensor<T, 4>(),
        input_backprop->tensor<T, 4>());
  }
#endif  // CUDNN_VERSION < 7300
}

#define DEFINE_DNN_OPS(T)         \
  template class DnnPoolingOp<T>; \
  template class DnnPoolingGradOp<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_DNN_OPS)

#if CUDNN_VERSION >= 7300
template class DnnPoolingOp<qint8>;
#endif

#undef DEFINE_DNN_OPS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
