#include "tensorflow/core/kernels/pooling_ops_common.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/public/tensor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 4,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  depth = tensor_in_shape.dim_size(3);
  tensor_in_cols = tensor_in_shape.dim_size(2);
  tensor_in_rows = tensor_in_shape.dim_size(1);
  tensor_in_batch = tensor_in_shape.dim_size(0);
  window_rows = ksize[1];
  window_cols = ksize[2];
  depth_window = ksize[3];
  row_stride = stride[1];
  col_stride = stride[2];
  depth_stride = stride[3];

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));

  if (depth_window == 1) {
    OP_REQUIRES_OK(context, Get2dOutputSize(
                                tensor_in_rows, tensor_in_cols, window_rows,
                                window_cols, row_stride, col_stride, padding,
                                &out_height, &out_width, &pad_rows, &pad_cols));
  } else {
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
    return TensorShape({tensor_in_batch, out_height, out_width, depth});
  } else {
    // Depthwise pooling
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}

#ifdef GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                    uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void TransformDepth<GPUDevice, T>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor in, \
      const Eigen::DSizes<Eigen::DenseIndex, 4>& shuffle,        \
      typename TTypes<T, 4>::Tensor out);                        \
  extern template struct TransformDepth<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

template <typename T>
void DnnPoolingGradOp<T>::Compute(
    OpKernelContext* context,
    perftools::gputools::dnn::PoolingMode pooling_mode,
    const std::vector<int32>& size, const std::vector<int32>& stride,
    Padding padding, const Tensor* tensor_in, const Tensor* tensor_out,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  CHECK((pooling_mode == perftools::gputools::dnn::PoolingMode::kMaximum) ||
        (tensor_in && tensor_out))
      << "For MaxPoolGrad, both tensor_in and tensor_out needs to be "
         "specified";

  Tensor* output = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_in_shape, &output));

  PoolParameters params{context, size, stride, padding, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }

  /// For now, cudnn does not support NHWC format, so we need to convert it
  /// to NCHW before calling cudnn. We need to get rid of this once it is done
  Tensor transformed_input;
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DataTypeToEnum<T>::value,
                              TensorShape({tensor_in_shape.dim_size(0),
                                           tensor_in_shape.dim_size(3),
                                           tensor_in_shape.dim_size(1),
                                           tensor_in_shape.dim_size(2)}),
                              &transformed_input));
  Tensor transformed_input_backprop;
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DataTypeToEnum<T>::value,
                              TensorShape({tensor_in_shape.dim_size(0),
                                           tensor_in_shape.dim_size(3),
                                           tensor_in_shape.dim_size(1),
                                           tensor_in_shape.dim_size(2)}),
                              &transformed_input_backprop));
  Tensor transformed_output;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({out_backprop.dim_size(0), out_backprop.dim_size(3),
                       out_backprop.dim_size(1), out_backprop.dim_size(2)}),
          &transformed_output));
  Tensor transformed_output_backprop;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({out_backprop.dim_size(0), out_backprop.dim_size(3),
                       out_backprop.dim_size(1), out_backprop.dim_size(2)}),
          &transformed_output_backprop));

  auto nhwc_to_nchw = Eigen::DSizes<Eigen::DenseIndex, 4>(0, 3, 1, 2);
  if (tensor_in) {
    // For AvgPoolGrad, the original input tensor is not necessary. However,
    // cudnn still requires them to run, although they do not affect the
    // results.
    functor::TransformDepth<GPUDevice, T>()(
        context->eigen_device<Device>(), tensor_in->tensor<T, 4>(),
        nhwc_to_nchw, transformed_input.tensor<T, 4>());
  }
  if (tensor_out) {
    // For AvgPoolGrad, the original output tensor is not necessary. However,
    // cudnn still requires them to run, although they do not affect the
    // results.
    functor::TransformDepth<GPUDevice, T>()(
        context->eigen_device<Device>(), tensor_out->tensor<T, 4>(),
        nhwc_to_nchw, transformed_output.tensor<T, 4>());
  }
  functor::TransformDepth<GPUDevice, T>()(
      context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
      nhwc_to_nchw, transformed_output_backprop.tensor<T, 4>());

  /// Get ready to call cudnn
  perftools::gputools::dnn::PoolingDescriptor pooling_desc;
  pooling_desc.set_pooling_mode(pooling_mode)
      .set_window_height(params.window_rows)
      .set_window_width(params.window_cols)
      .set_vertical_stride(params.row_stride)
      .set_horizontal_stride(params.col_stride)
      .set_vertical_padding(params.pad_rows)
      .set_horizontal_padding(params.pad_cols);

  perftools::gputools::dnn::BatchDescriptor orig_output_desc;
  orig_output_desc.set_count(params.tensor_in_batch)
      .set_height(params.out_height)
      .set_width(params.out_width)
      .set_feature_map_count(params.depth)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

  perftools::gputools::dnn::BatchDescriptor orig_input_desc;
  orig_input_desc.set_count(params.tensor_in_batch)
      .set_height(params.tensor_in_rows)
      .set_width(params.tensor_in_cols)
      .set_feature_map_count(params.depth)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

  auto orig_output_data =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());
  auto orig_input_data =
      AsDeviceMemory(transformed_input.template flat<T>().data(),
                     transformed_input.template flat<T>().size());
  auto output_backprop =
      AsDeviceMemory(transformed_output_backprop.template flat<T>().data(),
                     transformed_output_backprop.template flat<T>().size());
  auto input_backprop =
      AsDeviceMemory(transformed_input_backprop.template flat<T>().data(),
                     transformed_input_backprop.template flat<T>().size());

  auto* stream = context->op_device_context<GPUDeviceContext>()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  bool status =
      stream->ThenPoolBackward(pooling_desc, orig_input_desc, orig_input_data,
                               orig_output_desc, orig_output_data,
                               output_backprop, &input_backprop)
          .ok();
  OP_REQUIRES(context, status,
              errors::Internal("cudnn PoolBackward launch failed"));

  /// Transform the output data from NCHW back to NHWC
  auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
  auto nchw_to_nhwc = Eigen::DSizes<Eigen::DenseIndex, 4>(0, 2, 3, 1);
  functor::TransformDepth<GPUDevice, T>()(
      context->eigen_device<Device>(),
      toConstTensor(transformed_input_backprop).template tensor<T, 4>(),
      nchw_to_nhwc, output->tensor<T, 4>());
}

template class DnnPoolingGradOp<float>;

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
