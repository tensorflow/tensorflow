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

#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"

#include <limits>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace tensorflow {
using mkldnn::prop_kind;

template <typename T>
void MklPoolingFwdPrimitive<T>::Setup(const MklPoolingParams& fwdParams) {
  DCHECK(fwdParams.alg_kind == ALGORITHM::pooling_max ||
         fwdParams.alg_kind == ALGORITHM::pooling_avg ||
         fwdParams.alg_kind == ALGORITHM::pooling_avg_include_padding ||
         fwdParams.alg_kind == ALGORITHM::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";

  context_.alg_kind = fwdParams.alg_kind;
  context_.prop_kind = fwdParams.prop_kind;

  // Create memory descriptor
  // FIXME: Pooling doesn't expose to get the src_primitive_desc,
  //        so src format is currently hard-coded.
  //        A utility function is used to do this,
  //        which may be broken with future CPU architectures
#ifndef ENABLE_MKLDNN_V1
  bool is_2d = (fwdParams.src_dims.size() == 4);
  if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value)
    context_.src_fmt = is_2d ? MEMORY_FORMAT::nhwc : MEMORY_FORMAT::ndhwc;
  else
    context_.src_fmt = fwdParams.src_format;

  context_.src_md.reset(new memory::desc({fwdParams.src_dims}, MklDnnType<T>(),
                                         context_.src_fmt));
#else
  context_.src_md.reset(new memory::desc(fwdParams.src_md.data));
#endif  //  !ENABLE_MKLDNN_V1
  context_.dst_md.reset(new memory::desc({fwdParams.dst_dims}, MklDnnType<T>(),
                                         MEMORY_FORMAT::any));

#ifndef ENABLE_MKLDNN_V1
  // Create a pooling descriptor.
  context_.fwd_desc.reset(new pooling_forward::desc(
      fwdParams.prop_kind, fwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, fwdParams.strides, fwdParams.filter_dims,
      fwdParams.padding_left, fwdParams.padding_right, padding_kind::zero));
#else
  context_.fwd_desc.reset(new pooling_forward::desc(
      fwdParams.prop_kind, fwdParams.alg_kind, *context_.src_md,
      *context_.dst_md, fwdParams.strides, fwdParams.filter_dims,
      fwdParams.padding_left, fwdParams.padding_right));
#endif  // !ENABLE_MKLDNN_V1
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
#ifndef ENABLE_MKLDNN_V1
  context_.dst_fmt = static_cast<MEMORY_FORMAT>(
      context_.fwd_pd.get()->PRIMITIVE_DESC_DST.desc().data.format);
#else
  context_.dst_fmt = static_cast<MEMORY_FORMAT>(MEMORY_FORMAT::any);
#endif  // ENABLE_MKLDNN_V1

  // Create MKL-DNN internal memory object with dummy data.
  context_.src_mem.reset(new MEMORY_CONSTRUCTOR(
      context_.fwd_pd.get()->PRIMITIVE_DESC_SRC, cpu_engine_, DummyData));
  context_.dst_mem.reset(new MEMORY_CONSTRUCTOR(
      context_.fwd_pd.get()->PRIMITIVE_DESC_DST, cpu_engine_, DummyData));
  // For max pooling, need to return workspace (ws) for backward computing.
  if (fwdParams.alg_kind == ALGORITHM::pooling_max &&
      fwdParams.prop_kind == prop_kind::forward_training) {
#ifdef ENABLE_MKLDNN_V1
    context_.ws_mem.reset(
        new MEMORY_CONSTRUCTOR(context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE,
                               cpu_engine_, DummyData));
    context_.net_args.push_back({{MKLDNN_ARG_SRC, *context_.src_mem},
                                 {MKLDNN_ARG_DST, *context_.dst_mem},
                                 {MKLDNN_ARG_WORKSPACE, *context_.ws_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
#else
    auto ws_pd = context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE.desc().data;
    // Store workspace's dims and format to create workspace tensor.
    context_.ws_fmt = static_cast<MEMORY_FORMAT>(ws_pd.format);
    context_.ws_dims.assign(ws_pd.dims, ws_pd.dims + ws_pd.ndims);
    context_.ws_dt = static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
    context_.ws_size =
        context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE.get_size();

    context_.ws_mem.reset(
        new memory(context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE, DummyData));
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd, *context_.src_mem,
                                           *context_.dst_mem,
                                           *context_.ws_mem));
#endif  // ENABLE_MKLDNN_V1
  } else {
#ifdef ENABLE_MKLDNN_V1
    context_.net_args.push_back({{MKLDNN_ARG_SRC, *context_.src_mem},
                                 {MKLDNN_ARG_DST, *context_.dst_mem}});
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd));
#else
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd, *context_.src_mem,
                                           *context_.dst_mem));
#endif  // ENABLE_MKLDNN_V1
  }

  context_.fwd_primitives.push_back(*context_.fwd);
}

template <typename T>
void MklPoolingFwdPrimitive<T>::Execute(const T* src_data, T* dst_data,
                                        void* ws_data) {
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)));
  context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
  if (context_.alg_kind == ALGORITHM::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(ws_data);
  }

#ifdef ENABLE_MKLDNN_V1
  execute_primitives(context_.fwd_primitives, context_.fwd_stream,
                     context_.net_args);
#else
  context_.fwd_stream->submit(context_.fwd_primitives);
#endif  // ENABLE_MKLDNN_V1

  // Set back data handle.
  context_.src_mem->set_data_handle(DummyData);
  context_.dst_mem->set_data_handle(DummyData);
  if (context_.alg_kind == ALGORITHM::pooling_max &&
      context_.prop_kind ==
          prop_kind::forward_training) {  // Max pooling must have workspace.
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingFwdPrimitive<float>;
template class MklPoolingFwdPrimitive<quint8>;
template class MklPoolingFwdPrimitive<qint8>;
template class MklPoolingFwdPrimitive<bfloat16>;

template <typename T>
void MklPoolingBwdPrimitive<T>::Setup(const MklPoolingParams& bwdParams) {
  DCHECK(bwdParams.alg_kind == ALGORITHM::pooling_max ||
         bwdParams.alg_kind == ALGORITHM::pooling_avg ||
         bwdParams.alg_kind == ALGORITHM::pooling_avg_include_padding ||
         bwdParams.alg_kind == ALGORITHM::pooling_avg_exclude_padding)
      << "Pooling algorithm kind is not supported";
  context_.alg_kind = bwdParams.alg_kind;

  // Create memory descriptor.
  context_.diff_src_md.reset(new memory::desc(
      {bwdParams.src_dims}, MklDnnType<T>(), MEMORY_FORMAT::any));
  context_.diff_dst_md.reset(new memory::desc(
      {bwdParams.dst_dims}, MklDnnType<T>(), bwdParams.src_format));

#ifndef ENABLE_MKLDNN_V1
  context_.bwd_desc.reset(new pooling_backward::desc(
      bwdParams.alg_kind, *context_.diff_src_md, *context_.diff_dst_md,
      bwdParams.strides, bwdParams.filter_dims, bwdParams.padding_left,
      bwdParams.padding_right, padding_kind::zero));

  // Create a forward primitive,
  // which will be used as a hint for creating backward primitive.
  context_.fwd_desc.reset(new pooling_forward::desc(
      bwdParams.prop_kind, bwdParams.alg_kind, *context_.diff_src_md,
      *context_.diff_dst_md, bwdParams.strides, bwdParams.filter_dims,
      bwdParams.padding_left, bwdParams.padding_right, padding_kind::zero));
#else
  context_.bwd_desc.reset(new pooling_backward::desc(
      bwdParams.alg_kind, *context_.diff_src_md, *context_.diff_dst_md,
      bwdParams.strides, bwdParams.filter_dims, bwdParams.padding_left,
      bwdParams.padding_right));
  // Create a forward primitive,
  // which will be used as a hint for creating backward primitive.
  context_.fwd_desc.reset(new pooling_forward::desc(
      bwdParams.prop_kind, bwdParams.alg_kind, *context_.diff_src_md,
      *context_.diff_dst_md, bwdParams.strides, bwdParams.filter_dims,
      bwdParams.padding_left, bwdParams.padding_right));
#endif  // !ENABLE_MKLDNN_V1
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));
  context_.bwd_pd.reset(new pooling_backward::primitive_desc(
      *context_.bwd_desc, cpu_engine_, *context_.fwd_pd));

#ifndef ENABLE_MKLDNN_V1
  context_.diff_src_fmt = static_cast<MEMORY_FORMAT>(
      context_.bwd_pd.get()->PRIMITIVE_DESC_DIFF_SRC.desc().data.format);
  context_.diff_dst_fmt = bwdParams.src_format;
#endif  // ENABLE_MKLDNN_V1

#ifdef ENABLE_MKLDNN_V1
  // Create MKL-DNN internal memory object with dummy data.
  context_.diff_src_mem.reset(new memory(context_.bwd_pd.get()->diff_src_desc(),
                                         cpu_engine_, DummyData));
  context_.diff_dst_mem.reset(new memory(context_.bwd_pd.get()->diff_dst_desc(),
                                         cpu_engine_, DummyData));
#else
  context_.diff_src_mem.reset(
      new memory(context_.bwd_pd.get()->diff_src_primitive_desc(), DummyData));
  context_.diff_dst_mem.reset(new memory(
      {{{bwdParams.dst_dims}, MklDnnType<T>(), context_.diff_dst_fmt},
       cpu_engine_},
      DummyData));
#endif  // ENABLE_MKLDNN_V1

  // For max pooling, need to return workspace for backward computing.
  if (bwdParams.alg_kind == ALGORITHM::pooling_max) {
#ifdef ENABLE_MKLDNN_V1
    auto ws_pd = context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE.data;
    context_.ws_mem.reset(
        new memory(context_.fwd_pd.get()->workspace_desc(), cpu_engine_));
    context_.net_args.push_back(
        {{MKLDNN_ARG_DIFF_DST, *context_.diff_dst_mem},
         {MKLDNN_ARG_WORKSPACE, *context_.ws_mem},
         {MKLDNN_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
#else
    auto ws_pd = context_.fwd_pd.get()->PRIMITIVE_DESC_WORKSPACE.desc().data;
    context_.ws_dims.assign(ws_pd.dims, ws_pd.dims + ws_pd.ndims);
    context_.ws_fmt = static_cast<memory::format>(ws_pd.format);
    context_.ws_dt = static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
    context_.ws_mem.reset(new memory(
        {{{context_.ws_dims}, context_.ws_dt, context_.ws_fmt}, cpu_engine_},
        DummyData));
    context_.bwd.reset(
        new pooling_backward(*context_.bwd_pd, *context_.diff_dst_mem,
                             *context_.ws_mem, *context_.diff_src_mem));
#endif  // ENABLE_MKLDNN_V1
  } else {
#ifdef ENABLE_MKLDNN_V1
    context_.net_args.push_back(
        {{MKLDNN_ARG_DIFF_DST, *context_.diff_dst_mem},
         {MKLDNN_ARG_DIFF_SRC, *context_.diff_src_mem}});
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd));
#else
    context_.bwd.reset(new pooling_backward(
        *context_.bwd_pd, *context_.diff_dst_mem, *context_.diff_src_mem));
#endif  // ENABLE_MKLDNN_V1
  }
  context_.bwd_primitives.push_back(*context_.bwd);
}

template <typename T>
void MklPoolingBwdPrimitive<T>::Execute(const T* diff_dst_data,
                                        T* diff_src_data, const void* ws_data) {
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)));
  context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
  if (context_.alg_kind == ALGORITHM::pooling_max) {
    DCHECK(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data));
  }

#ifdef ENABLE_MKLDNN_V1
  execute_primitives(context_.bwd_primitives, context_.bwd_stream,
                     context_.net_args);
#else
  context_.bwd_stream->submit(context_.bwd_primitives);
#endif  // ENABLE_MKLDNN_V1

  // Set back data handle.
  context_.diff_dst_mem->set_data_handle(DummyData);
  context_.diff_src_mem->set_data_handle(DummyData);
  if (context_.alg_kind == ALGORITHM::pooling_max) {
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

// Initialization for MKL format.
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

    // TF can work with int64, but mkldnn only supports int32.
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

}  // namespace tensorflow

#endif  // INTEL_MKL
