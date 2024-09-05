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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cudnn_pooling_gpu.h"

#include <array>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void DnnPooling3dImpl(OpKernelContext* context,
                      se::dnn::PoolingMode pooling_mode,
                      const std::array<int64_t, 3>& window,
                      const std::array<int64_t, 3>& stride,
                      const std::array<int64_t, 3>& padding,
                      TensorFormat data_format, const Tensor& tensor_in,
                      Tensor* output) {
  const auto in_shape = tensor_in.shape();
  const auto out_shape = output->shape();

  const int64_t in_batch = GetTensorDim(tensor_in, data_format, 'N');
  const int64_t in_features = GetTensorDim(tensor_in, data_format, 'C');

  Tensor transformed_input;
  if (data_format == FORMAT_NHWC) {
    TensorShape transformed_input_shape;
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                FORMAT_NCHW, tensor_in.shape(), data_format,
                                &transformed_input_shape));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));
    functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                           tensor_in.tensor<T, 5>(),
                                           transformed_input.tensor<T, 5>());
  } else {
    transformed_input = tensor_in;
  }
  Tensor transformed_output;
  if (data_format == FORMAT_NHWC) {
    TensorShape transformed_output_shape;
    OP_REQUIRES_OK(
        context, ShapeFromFormatWithStatus(FORMAT_NCHW, out_shape, data_format,
                                           &transformed_output_shape));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_output_shape,
                                                   &transformed_output));
  } else {
    transformed_output = *output;
  }

  se::dnn::PoolingDescriptor pooling_desc(3);
  pooling_desc.set_pooling_mode(pooling_mode);
  se::dnn::BatchDescriptor input_desc(3);
  input_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::BatchDescriptor output_desc(3);
  output_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  for (size_t i = 0; i < window.size(); ++i) {
    const auto dim_i = static_cast<se::dnn::DimIndex>(i);
    pooling_desc.set_window(dim_i, window[i]);
    pooling_desc.set_stride(dim_i, stride[i]);
    pooling_desc.set_padding(dim_i, padding[i]);
    input_desc.set_spatial_dim(dim_i,
                               GetTensorDim(tensor_in, data_format, '2' - i));
    output_desc.set_spatial_dim(dim_i,
                                GetTensorDim(out_shape, data_format, '2' - i));
  }

  auto input_data = AsDeviceMemory(transformed_input.template flat<T>().data(),
                                   transformed_input.template flat<T>().size());
  auto output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
  auto* dnn = stream->parent()->AsDnn();
  OP_REQUIRES(context, dnn != nullptr,
              errors::Internal("No DNN support for stream."));
#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  OP_REQUIRES_OK(context, dnn->PoolForward(stream, pooling_desc,
                                           GetNumericOptionsForCuDnn(),
                                           input_desc, input_data, output_desc,
                                           &output_data, &scratch_allocator));
#else
  OP_REQUIRES_OK(
      context,
      dnn->PoolForward(stream, pooling_desc, GetNumericOptionsForCuDnn(),
                       input_desc, input_data, output_desc, &output_data));
#endif

  if (data_format == FORMAT_NHWC) {
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_output).template tensor<T, 5>(),
        output->tensor<T, 5>());
  }
}

template <typename T>
void DnnPooling3dOp<T>::Compute(OpKernelContext* context,
                                se::dnn::PoolingMode pooling_mode,
                                const std::array<int64_t, 3>& window,
                                const std::array<int64_t, 3>& stride,
                                const std::array<int64_t, 3>& padding,
                                TensorFormat data_format,
                                const Tensor& tensor_in, Tensor* output) {
  DnnPooling3dImpl<T>(context, pooling_mode, window, stride, padding,
                      data_format, tensor_in, output);
}

template <>
void DnnPooling3dOp<Eigen::bfloat16>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::array<int64_t, 3>& window, const std::array<int64_t, 3>& stride,
    const std::array<int64_t, 3>& padding, TensorFormat data_format,
    const Tensor& tensor_in, Tensor* output) {
  auto* stream = context->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);

  if (cast_to_float) {
    Tensor casted_in;
    Tensor casted_output;
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, tensor_in.shape(),
                                                   &casted_in));
    cast(device, casted_in.template flat<float>(),
         tensor_in.template flat<Eigen::bfloat16>());
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output->shape(),
                                                   &casted_output));

    DnnPooling3dImpl<float>(context, pooling_mode, window, stride, padding,
                            data_format, casted_in, &casted_output);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_output_const = casted_output;
    cast_back(device, output->template flat<Eigen::bfloat16>(),
              casted_output_const.template flat<float>());
    return;
  }
  DnnPooling3dImpl<Eigen::bfloat16>(context, pooling_mode, window, stride,
                                    padding, data_format, tensor_in, output);
}

template <typename T>
void DnnPooling3dGradImpl(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::array<int64_t, 3>& window, const std::array<int64_t, 3>& stride,
    const std::array<int64_t, 3>& padding,
    const std::array<int64_t, 3>& output_size, TensorFormat data_format,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape,
    const Tensor* tensor_in, const Tensor* tensor_out, Tensor* input_backprop) {
  CHECK((pooling_mode != se::dnn::PoolingMode::kMaximum) ||
        (tensor_in && tensor_out))
      << "For MaxPoolGrad, both tensor_in and tensor_out needs to be "
         "specified";

  // If input is empty, we are done.
  if (tensor_in_shape.num_elements() == 0) {
    return;
  }

  const int64_t in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  const int64_t in_features = GetTensorDim(tensor_in_shape, data_format, 'C');

  Tensor transformed_input;
  TensorShape transformed_input_shape;
  if (data_format == FORMAT_NHWC || tensor_in == nullptr) {
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                FORMAT_NCHW, tensor_in_shape, data_format,
                                &transformed_input_shape));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   transformed_input_shape,
                                                   &transformed_input));
  } else {
    transformed_input = *tensor_in;
  }
  Tensor transformed_output;
  TensorShape transformed_output_shape;
  if (data_format == FORMAT_NHWC || tensor_out == nullptr) {
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                FORMAT_NCHW, out_backprop.shape(), data_format,
                                &transformed_output_shape));
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
    if (tensor_in != nullptr) {
      functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                             tensor_in->tensor<T, 5>(),
                                             transformed_input.tensor<T, 5>());
    }
    if (tensor_out != nullptr) {
      functor::NHWCToNCHW<GPUDevice, T, 5>()(context->eigen_device<GPUDevice>(),
                                             tensor_out->tensor<T, 5>(),
                                             transformed_output.tensor<T, 5>());
    }
    functor::NHWCToNCHW<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
        transformed_output_backprop.tensor<T, 5>());
  }

  se::dnn::PoolingDescriptor pooling_desc(3);
  pooling_desc.set_pooling_mode(pooling_mode);

  se::dnn::BatchDescriptor orig_output_desc(3);
  orig_output_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);

  se::dnn::BatchDescriptor orig_input_desc(3);
  orig_input_desc.set_count(in_batch)
      .set_feature_map_count(in_features)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);

  for (size_t i = 0; i < window.size(); ++i) {
    const auto dim_i = static_cast<se::dnn::DimIndex>(i);
    pooling_desc.set_window(dim_i, window[i]);
    pooling_desc.set_stride(dim_i, stride[i]);
    pooling_desc.set_padding(dim_i, padding[i]);
    orig_input_desc.set_spatial_dim(
        dim_i, GetTensorDim(tensor_in_shape, data_format, '2' - i));
    orig_output_desc.set_spatial_dim(dim_i, output_size[i]);
  }

  auto orig_output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());
  auto orig_input_data =
      AsDeviceMemory(transformed_input.template flat<T>().data(),
                     transformed_input.template flat<T>().size());
  auto output_backprop_data =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop_data =
      AsDeviceMemory(transformed_input_backprop.template flat<T>().data(),
                     transformed_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
  auto* dnn = stream->parent()->AsDnn();
  OP_REQUIRES(context, dnn != nullptr,
              errors::Internal("No DNN support for stream."));

#if TENSORFLOW_USE_ROCM
  static int64 PoolingScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  DnnScratchAllocator scratch_allocator(PoolingScratchSize, context);
  OP_REQUIRES_OK(
      context,
      dnn->PoolBackward(stream, pooling_desc, GetNumericOptionsForCuDnn(),
                        orig_input_desc, orig_input_data, orig_output_desc,
                        orig_output_data, output_backprop_data,
                        &input_backprop_data, &scratch_allocator));
#else
  OP_REQUIRES_OK(context, dnn->PoolBackward(
                              stream, pooling_desc, GetNumericOptionsForCuDnn(),
                              orig_input_desc, orig_input_data,
                              orig_output_desc, orig_output_data,
                              output_backprop_data, &input_backprop_data));
#endif

  if (data_format == FORMAT_NHWC) {
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_input_backprop).template tensor<T, 5>(),
        input_backprop->tensor<T, 5>());
  }
}

template <typename T>
void DnnPooling3dGradOp<T>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::array<int64_t, 3>& window, const std::array<int64_t, 3>& stride,
    const std::array<int64_t, 3>& padding,
    const std::array<int64_t, 3>& output_size, TensorFormat data_format,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape,
    const Tensor* tensor_in, const Tensor* tensor_out, Tensor* input_backprop) {
  DnnPooling3dGradImpl<T>(
      context, pooling_mode, window, stride, padding, output_size, data_format,
      out_backprop, tensor_in_shape, tensor_in, tensor_out, input_backprop);
}

template <>
void DnnPooling3dGradOp<Eigen::bfloat16>::Compute(
    OpKernelContext* context, se::dnn::PoolingMode pooling_mode,
    const std::array<int64_t, 3>& window, const std::array<int64_t, 3>& stride,
    const std::array<int64_t, 3>& padding,
    const std::array<int64_t, 3>& output_size, TensorFormat data_format,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape,
    const Tensor* tensor_in, const Tensor* tensor_out, Tensor* input_backprop) {
  auto* stream = context->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);
  if (cast_to_float) {
    Tensor casted_out_backprop;
    Tensor casted_tensor_in;
    Tensor casted_tensor_out;
    Tensor casted_input_backprop;
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_FLOAT, out_backprop.shape(),
                                          &casted_out_backprop));
    cast(device, casted_out_backprop.template flat<float>(),
         out_backprop.template flat<Eigen::bfloat16>());
    if (tensor_in != nullptr) {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_FLOAT, tensor_in->shape(),
                                            &casted_tensor_in));
      cast(device, casted_tensor_in.template flat<float>(),
           tensor_in->template flat<Eigen::bfloat16>());
    }
    if (tensor_out != nullptr) {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_FLOAT, tensor_out->shape(),
                                            &casted_tensor_out));
      cast(device, casted_tensor_out.template flat<float>(),
           tensor_out->template flat<Eigen::bfloat16>());
    }
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_FLOAT, input_backprop->shape(),
                                          &casted_input_backprop));
    DnnPooling3dGradImpl<float>(
        context, pooling_mode, window, stride, padding, output_size,
        data_format, casted_out_backprop, tensor_in_shape,
        tensor_in != nullptr ? &casted_tensor_in : nullptr,
        tensor_out != nullptr ? &casted_tensor_out : nullptr,
        &casted_input_backprop);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_input_backprop_const = casted_input_backprop;
    cast_back(device, input_backprop->template flat<Eigen::bfloat16>(),
              casted_input_backprop_const.template flat<float>());
    return;
  }
  DnnPooling3dGradImpl<Eigen::bfloat16>(
      context, pooling_mode, window, stride, padding, output_size, data_format,
      out_backprop, tensor_in_shape, tensor_in, tensor_out, input_backprop);
}

#define DEFINE_DNN_OPS(T)           \
  template class DnnPooling3dOp<T>; \
  template class DnnPooling3dGradOp<T>;

TF_CALL_float(DEFINE_DNN_OPS) TF_CALL_half(DEFINE_DNN_OPS)
    TF_CALL_bfloat16(DEFINE_DNN_OPS)

#undef DEFINE_DNN_OPS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
