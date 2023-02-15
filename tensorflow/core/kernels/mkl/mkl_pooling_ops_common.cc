/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/kernels/mkl/mkl_pooling_ops_common.h"

#include <limits>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace tensorflow {
#ifndef ENABLE_ONEDNN_V3
#define GET_MEMORY_DESC(md) md.data
#else
#define GET_MEMORY_DESC(md) md
#endif  // !ENABLE_ONEDNN_V3
using dnnl::prop_kind;

template <typename T>
void MklPoolingFwdPrimitive<T>::Setup(const MklPoolingParams& fwdParams) {
  DCHECK(fwdParams.alg_kind == dnnl::algorithm::pooling_max ||
#ifndef ENABLE_ONEDNN_V3
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg ||
#endif  // !ENABLE_ONEDNN_V3
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg_include_padding ||
         fwdParams.alg_kind == dnnl::algorithm::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";

  context_.alg_kind = fwdParams.alg_kind;
  context_.prop_kind = fwdParams.prop_kind;

  // Create memory descriptor
  // TODO(intel-tf): Pooling doesn't expose to get the src_primitive_desc,
  //                 so src format is currently hard-coded.
  //                 A utility function is used to do this,
  //                 which may be broken with future CPU architectures
  context_.src_md.reset(new memory::desc(fwdParams.GET_MEMORY_DESC(src_md)));
  context_.dst_md.reset(new memory::desc({fwdParams.dst_dims}, MklDnnType<T>(),
                                         fwdParams.native_format
                                             ? fwdParams.src_format
                                             : memory::format_tag::any));

  // Create a pooling descriptor.
#ifndef ENABLE_ONEDNN_V3
  context_.fwd_desc.reset(new pooling_forward::desc(
      fwdParams.prop_kind, fwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, fwdParams.strides, fwdParams.filter_dims,
      fwdParams.padding_left, fwdParams.padding_right));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
#else
  context_.fwd_pd.reset(new pooling_forward::primitive_desc(
      cpu_engine_, fwdParams.prop_kind, fwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, fwdParams.strides, fwdParams.filter_dims,
      fwdParams.dilations, fwdParams.padding_left, fwdParams.padding_right));
#endif  // !ENABLE_ONEDNN_V3
  context_.dst_fmt = static_cast<memory::format_tag>(memory::format_tag::any);

  // Create oneDNN internal memory object with dummy data.
  context_.src_mem.reset(
      new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
  context_.dst_mem.reset(
      new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));

  // For max pooling, need to return workspace (ws) for backward computing.
  if (fwdParams.alg_kind == dnnl::algorithm::pooling_max &&
      fwdParams.prop_kind == prop_kind::forward_training) {
    context_.ws_mem.reset(new memory(context_.fwd_pd.get()->workspace_desc(),
                                     cpu_engine_, DummyData));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.src_mem},
                                 {DNNL_ARG_DST, *context_.dst_mem},
                                 {DNNL_ARG_WORKSPACE, *context_.ws_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
  } else {
    context_.net_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
  }

  context_.fwd_primitives.push_back(*context_.fwd);
}

template <typename T>
void MklPoolingFwdPrimitive<T>::Execute(const T* src_data, T* dst_data,
                                        void* ws_data,
                                        std::shared_ptr<stream> fwd_stream) {
#ifdef DNNL_AARCH64_USE_ACL
  mutex_lock lock(primitive_execution_mu_);
#endif
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)), *fwd_stream);
  context_.dst_mem->set_data_handle(static_cast<void*>(dst_data), *fwd_stream);
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(ws_data, *fwd_stream);
  }
#else
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)));
  context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(ws_data);
  }
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V3
  execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

  // Set back data handle.
  context_.src_mem->set_data_handle(DummyData);
  context_.dst_mem->set_data_handle(DummyData);
  if (context_.alg_kind == dnnl::algorithm::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingFwdPrimitive<float>;
template class MklPoolingFwdPrimitive<bfloat16>;

#ifndef ENABLE_ONEDNN_V3
template class MklPoolingFwdPrimitive<quint8>;
template class MklPoolingFwdPrimitive<qint8>;
#endif  // !ENABLE_ONEDNN_V3

template <typename T>
void MklPoolingBwdPrimitive<T>::Setup(const MklPoolingParams& bwdParams) {
  DCHECK(bwdParams.alg_kind == dnnl::algorithm::pooling_max ||
#ifndef ENABLE_ONEDNN_V3
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg ||
#endif  // !ENABLE_ONEDNN_V3
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg_include_padding ||
         bwdParams.alg_kind == dnnl::algorithm::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";
  context_.alg_kind = bwdParams.alg_kind;

  // Create memory descriptor.
  context_.src_md.reset(new memory::desc({bwdParams.src_dims}, MklDnnType<T>(),
                                         memory::format_tag::any));
  context_.src_md.reset(new memory::desc(bwdParams.GET_MEMORY_DESC(src_md)));
  context_.dst_md.reset(new memory::desc({bwdParams.dst_dims}, MklDnnType<T>(),
                                         bwdParams.native_format
                                             ? bwdParams.src_format
                                             : memory::format_tag::any));

#ifndef ENABLE_ONEDNN_V3
  // Create a backward primitive. The implementation for backward must comply to
  // the workspace format it gets from forward pass, so we directly use src_md
  // and dst_md here.
  context_.bwd_desc.reset(new pooling_backward::desc(
      bwdParams.alg_kind, *context_.src_md, *context_.dst_md, bwdParams.strides,
      bwdParams.filter_dims, bwdParams.padding_left, bwdParams.padding_right));
  // Create a forward primitive,
  // which will be used as a hint for creating backward primitive.
  context_.fwd_desc.reset(new pooling_forward::desc(
      bwdParams.prop_kind, bwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, bwdParams.strides, bwdParams.filter_dims,
      bwdParams.padding_left, bwdParams.padding_right));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
  context_.bwd_pd.reset(new pooling_backward::primitive_desc(
      *context_.bwd_desc, cpu_engine_, *context_.fwd_pd));
#else
  context_.fwd_pd.reset(new pooling_forward::primitive_desc(
      cpu_engine_, bwdParams.prop_kind, bwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, bwdParams.strides, bwdParams.filter_dims,
      bwdParams.dilations, bwdParams.padding_left, bwdParams.padding_right));
  context_.bwd_pd.reset(new pooling_backward::primitive_desc(
      cpu_engine_, bwdParams.alg_kind, *context_.src_md, *context_.dst_md,
      bwdParams.strides, bwdParams.filter_dims, bwdParams.dilations,
      bwdParams.padding_left, bwdParams.padding_right, *context_.fwd_pd));
#endif  // !ENABLE_ONEDNN_V3

  // Create oneDNN internal memory object with dummy data.
  context_.diff_src_mem.reset(new memory(context_.bwd_pd.get()->diff_src_desc(),
                                         cpu_engine_, DummyData));
  context_.diff_dst_mem.reset(new memory(context_.bwd_pd.get()->diff_dst_desc(),
                                         cpu_engine_, DummyData));

  // For max pooling, need to return workspace for backward computing.
  if (bwdParams.alg_kind == dnnl::algorithm::pooling_max) {
    context_.ws_mem.reset(
        new memory(context_.fwd_pd.get()->workspace_desc(), cpu_engine_));
    context_.net_args.push_back({{DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
                                 {DNNL_ARG_WORKSPACE, *context_.ws_mem},
                                 {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
  } else {
    context_.net_args.push_back({{DNNL_ARG_DIFF_DST, *context_.diff_dst_mem},
                                 {DNNL_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
  }
  context_.bwd_primitives.push_back(*context_.bwd);
}

template <typename T>
void MklPoolingBwdPrimitive<T>::Execute(const T* diff_dst_data,
                                        T* diff_src_data, const void* ws_data,
                                        std::shared_ptr<stream> bwd_stream) {
#ifdef DNNL_AARCH64_USE_ACL
  mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)), *bwd_stream);
  context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data),
                                         *bwd_stream);
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data), *bwd_stream);
  }
#else
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)));
  context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data));
  }
#endif  // !ENABLE_ONEDNN_OPENMP

  execute_primitives(context_.bwd_primitives, bwd_stream, context_.net_args);

  // Set back data handle.
  context_.diff_dst_mem->set_data_handle(DummyData);
  context_.diff_src_mem->set_data_handle(DummyData);
  if (context_.alg_kind == dnnl::algorithm::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingBwdPrimitive<float>;
template class MklPoolingBwdPrimitive<bfloat16>;

// Initialization for TensorFlow format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const TensorShape& tensor_in_shape) {
  // For max pooling, tensor_in should have 4 or 5 dimensions.
  OP_REQUIRES(context,
              tensor_in_shape.dims() == 4 || tensor_in_shape.dims() == 5,
              errors::InvalidArgument("tensor_in must be 4 or 5-dimensional"));

  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  if (tensor_in_shape.dims() == 4) {
    // Pool2D
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  } else {
    // Pool3D
    tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
  }
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  Init(context, ksize, stride, padding, data_format);
}

// Initialization for oneDNN format.
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklDnnShape* mklInputShape) {
  // Get the input sizes.
  if (ksize.size() == 4) {
    // Pool2D
    depth = mklInputShape->GetDimension('C');
    tensor_in_cols = mklInputShape->GetDimension('W');
    tensor_in_rows = mklInputShape->GetDimension('H');
    tensor_in_batch = mklInputShape->GetDimension('N');
  } else {
    // Pool3D
    depth = mklInputShape->GetDimension3D('C');
    tensor_in_cols = mklInputShape->GetDimension3D('W');
    tensor_in_rows = mklInputShape->GetDimension3D('H');
    tensor_in_planes = mklInputShape->GetDimension3D('D');
    tensor_in_batch = mklInputShape->GetDimension3D('N');
  }

  Init(context, ksize, stride, padding, data_format);
}

// Common Initialization for TensorFlow and MKL formats.
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format) {
  // Get the data format.
  this->data_format = data_format;

  bool is_pool2d = (ksize.size() == 4);
  if (is_pool2d) {
    // Pool2D
    // Get the output sizes.
    window_rows = GetTensorDim(ksize, data_format, 'H');
    window_cols = GetTensorDim(ksize, data_format, 'W');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    row_stride = GetTensorDim(stride, data_format, 'H');
    col_stride = GetTensorDim(stride, data_format, 'W');
    depth_stride = GetTensorDim(stride, data_format, 'C');

#ifdef ENABLE_ONEDNN_V3
    // TODO(intel-tf): we are setting dilations to 0 to mimic the behavior of
    // oneDNN v2.x integration code. We can extend this in the future to support
    // dilations != 0
    row_dilation = 0;
    col_dilation = 0;
#endif  // ENABLE_ONEDNN_V3

    // We only support 2D pooling across width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
                errors::Unimplemented(
                    "MaxPooling supports exactly one of pooling across depth "
                    "or pooling across width/height."));
  } else {
    // Pool3D
    // Get the output sizes.
    window_planes = GetTensorDim(ksize, data_format, '0');
    window_rows = GetTensorDim(ksize, data_format, '1');
    window_cols = GetTensorDim(ksize, data_format, '2');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    planes_stride = GetTensorDim(stride, data_format, '0');
    row_stride = GetTensorDim(stride, data_format, '1');
    col_stride = GetTensorDim(stride, data_format, '2');
    depth_stride = GetTensorDim(stride, data_format, 'C');

#ifdef ENABLE_ONEDNN_V3
    // TODO(intel-tf): TensorFlow's 3D-pooling API does not support dilations
    planes_dilation = 0;
    row_dilation = 0;
    col_dilation = 0;
#endif  // ENABLE_ONEDNN_V3

    // We only support 3D pooling across depth/width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 ||
                 (window_rows == 1 && window_cols == 1 && window_planes == 1)),
                errors::Unimplemented(
                    "AvgPooling3D supports exactly one of pooling across depth "
                    "or pooling across depth/width/height."));
  }

  if (depth_window == 1) {  // We are pooling in the D (Pool3D only), H and W.
    if (!is_pool2d) {
      OP_REQUIRES_OK(
          context, GetWindowedOutputSizeVerbose(tensor_in_planes, window_planes,
                                                planes_stride, padding,
                                                &out_planes, &pad_P1, &pad_P2));
    }

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));

    // TF can work with int64, but oneDNN only supports int32.
    // Fail if the depth, height or width are greater than MAX_INT.
    // We check depth only for 3D pooling case.
    if (!is_pool2d) {
      OP_REQUIRES(context,
                  FastBoundsCheck(out_planes, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("output depth/planes is too large"));
    }

    OP_REQUIRES(context,
                FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output height is too large"));

    OP_REQUIRES(context,
                FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output width is too large"));

    out_depth = depth;  // Output will have the same depth as the input.
  } else {              // We are pooling in the depth dimension.
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the depth
    // stride (no overlapping).
    OP_REQUIRES(context, depth % depth_window == 0,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to evenly divide the"
                                      " input depth"));
    OP_REQUIRES(context, depth_stride == depth_window,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to equal the depth"
                                      " stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    out_depth = depth / depth_window;
  }
}

#undef GET_MEMORY_DESC

}  // namespace tensorflow

#endif  // INTEL_MKL
