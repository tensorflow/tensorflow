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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

#ifndef INTEL_MKL_ML

using mkldnn::pooling_max;
using mkldnn::pooling_avg;
using mkldnn::pooling_avg_include_padding;
using mkldnn::pooling_avg_exclude_padding;
using mkldnn::prop_kind;

template <typename T>
void MklPoolingFwdPrimitive<T>::Setup(const MklPoolingParams& fwdParams) {
  if (fwdParams.alg_kind != pooling_max &&
    fwdParams.alg_kind != pooling_avg &&
    fwdParams.alg_kind != pooling_avg_include_padding &&
    fwdParams.alg_kind != pooling_avg_exclude_padding) {
    assert("Pooling algorithm kind is not supported\n");
  }

  context_.alg_kind = fwdParams.alg_kind;
  // create memory desc
  // FIXME: Pooling doesn't expose to get the src_primitive_desc,
  //        so src format is currently hard-coded.
  //        A utility function is used to do this,
  //        which may be broken with future CPU architectures
  context_.src_md.reset(new memory::desc({fwdParams.src_dims},
      MklDnnType<T>(), get_desired_format(fwdParams.src_dims[1])));
  context_.dst_md.reset(new memory::desc({fwdParams.dst_dims},
      MklDnnType<T>(), memory::format::any));

  // create a pooling descriptor
  context_.fwd_desc.reset(new pooling_forward::desc(prop_kind::forward_training,
      fwdParams.alg_kind, *context_.src_md, *context_.dst_md, fwdParams.strides,
      fwdParams.filter_dims, fwdParams.padding_left,
      fwdParams.padding_right, padding_kind::zero));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine_));

  // store expected primitive format
  context_.src_fmt = get_desired_format(fwdParams.src_dims[1]);
  context_.dst_fmt = static_cast<mkldnn::memory::format>(
      context_.fwd_pd.get()->dst_primitive_desc().desc().data.format);

  // create MKL-DNN internal memory object with dummy data
  context_.src_mem.reset(
      new memory({{{fwdParams.src_dims}, MklDnnType<T>(), context_.src_fmt},
      cpu_engine_}, DummyData));
  context_.dst_mem.reset(
      new memory(context_.fwd_pd.get()->dst_primitive_desc(), DummyData));

  // for max pooling, need to return workspace(ws) for backward computing
  if (fwdParams.alg_kind == pooling_max) {
    auto ws_pd = context_.fwd_pd.get()->workspace_primitive_desc().desc().data;
    // store workspace's dims and format to create workspace tensor
    context_.ws_fmt = static_cast<mkldnn::memory::format>(ws_pd.format);
    context_.ws_dims.assign(ws_pd.dims, ws_pd.dims + ws_pd.ndims);
    context_.ws_dt = static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
    context_.ws_size = 
      context_.fwd_pd.get()->workspace_primitive_desc().get_size();
    context_.ws_mem.reset(
        new memory(context_.fwd_pd.get()->workspace_primitive_desc(),
                  DummyData));
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd, 
          *context_.src_mem, *context_.dst_mem, *context_.ws_mem));
  } else {
    context_.fwd.reset(new pooling_forward(*context_.fwd_pd, 
          *context_.src_mem, *context_.dst_mem));
  }

  context_.fwd_primitives.push_back(*context_.fwd);
}

template <typename T>
void MklPoolingFwdPrimitive<T>::Execute(const T* src_data, const T* dst_data, 
                                        const void* ws_data) {
  context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)));
  context_.dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(dst_data)));
  if (context_.alg_kind == pooling_max) {  // max pooling must have ws
    assert(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data));
  }
  context_.fwd_stream->submit(context_.fwd_primitives);

  // set back data handle
  context_.src_mem->set_data_handle(DummyData);
  context_.dst_mem->set_data_handle(DummyData);
  if (context_.alg_kind == pooling_max) {  // max pooling must have ws
    assert(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingFwdPrimitive<float>;

template <typename T>
void MklPoolingBwdPrimitive<T>::Setup(const MklPoolingParams& bwdParams) {
  if (bwdParams.alg_kind != pooling_max && bwdParams.alg_kind != pooling_avg
     && bwdParams.alg_kind != pooling_avg_include_padding
     && bwdParams.alg_kind != pooling_avg_exclude_padding) {
    assert("Pooling algorithm kind is not supported\n");
  }
  context_.alg_kind = bwdParams.alg_kind;

  // Create memory desc
  context_.diff_src_md.reset(new memory::desc({bwdParams.src_dims}, 
        MklDnnType<T>(), memory::format::any));
  context_.diff_dst_md.reset(new memory::desc({bwdParams.dst_dims}, 
        MklDnnType<T>(), get_desired_format(bwdParams.dst_dims[1])));
  context_.bwd_desc.reset(new pooling_backward::desc(bwdParams.alg_kind,
      *context_.diff_src_md, *context_.diff_dst_md, bwdParams.strides, 
      bwdParams.filter_dims, bwdParams.padding_left, bwdParams.padding_right, 
      padding_kind::zero));

  // create a forward primitive,
  // which will be used as a hint for creating backward primitive
  context_.fwd_desc.reset(new pooling_forward::desc(prop_kind::forward_training,
      bwdParams.alg_kind, *context_.diff_src_md, *context_.diff_dst_md,
      bwdParams.strides, bwdParams.filter_dims, bwdParams.padding_left,
      bwdParams.padding_right, padding_kind::zero));
  context_.fwd_pd.reset(
      new pooling_forward::primitive_desc(*context_.fwd_desc, cpu_engine));
  context_.bwd_pd.reset(new pooling_backward::primitive_desc(
      *context_.bwd_desc, cpu_engine, *context_.fwd_pd));

  // store expected primitive format
  context_.diff_src_fmt = static_cast<mkldnn::memory::format>(
      context_.bwd_pd.get()->diff_src_primitive_desc().desc().data.format);
  context_.diff_dst_fmt = get_desired_format(bwdParams.dst_dims[1]);

  // create MKL-DNN internal memory object with dummy data
  context_.diff_src_mem.reset(
      new memory(context_.bwd_pd.get()->diff_src_primitive_desc(), DummyData));
  context_.diff_dst_mem.reset(new memory({{{bwdParams.dst_dims},
        MklDnnType<T>(), context_.diff_dst_fmt}, cpu_engine}, DummyData));

  // for max pooling, need to return workspace for backward
  if (bwdParams.alg_kind == pooling_max) {
    auto ws_pd = context_.fwd_pd.get()->workspace_primitive_desc().desc().data;
    context_.ws_dims.assign(ws_pd.dims, ws_pd.dims + ws_pd.ndims);
    context_.ws_fmt = get_desired_format(context_.ws_dims[1]);
    context_.ws_dt =  static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
    context_.ws_mem.reset(new memory({{{context_.ws_dims}, context_.ws_dt, 
          context_.ws_fmt}, cpu_engine}, DummyData));
    context_.bwd.reset(new pooling_backward(
               *context_.bwd_pd, *context_.diff_dst_mem, *context_.ws_mem, 
               *context_.diff_src_mem));
  } else {
    context_.bwd.reset(new pooling_backward(*context_.bwd_pd,
          *context_.diff_dst_mem, *context_.diff_src_mem));
  }
  context_.bwd_primitives.push_back(*context_.bwd);
}

template<typename T>
void MklPoolingBwdPrimitive<T>::Execute(const T* diff_dst_data,
                            const T* diff_src_data, const void* ws_data) {
  context_.diff_dst_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_dst_data)));
  context_.diff_src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(diff_src_data)));
  if (context_.alg_kind == pooling_max) {
    assert(ws_data != nullptr);
    context_.ws_mem->set_data_handle(const_cast<void*>(ws_data));
  }

  context_.bwd_stream->submit(context_.bwd_primitives);
  //  set back data handle
  context_.diff_dst_mem->set_data_handle(DummyData);
  context_.diff_src_mem->set_data_handle(DummyData);
  if (context_.alg_kind == pooling_max) {
    assert(ws_data != nullptr);
    context_.ws_mem->set_data_handle(DummyData);
  }
}

template class MklPoolingBwdPrimitive<float>;

#endif

// Initialization for TensorFlow format
void MklPoolParameters::Init(
        OpKernelContext* context,
        const std::vector<int32>& ksize,
        const std::vector<int32>& stride, Padding padding,
        TensorFormat data_format,
        const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 4,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  Init(context, ksize, stride, padding, data_format);
}

#ifdef INTEL_MKL_ML
// Initialization for MKL format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklShape* mklInputShape) {
  // Get the input sizes
  depth = mklInputShape->GetSizes()[2];
  tensor_in_cols = mklInputShape->GetSizes()[0];
  tensor_in_rows = mklInputShape->GetSizes()[1];
  tensor_in_batch = mklInputShape->GetSizes()[3];

  Init(context, ksize, stride, padding, data_format);
}
#else
// Initialization for MKL format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklDnnShape* mklInputShape) {
  // Get the input sizes
  depth = mklInputShape->GetDimension('C');
  tensor_in_cols = mklInputShape->GetDimension('W');
  tensor_in_rows = mklInputShape->GetDimension('H');
  tensor_in_batch = mklInputShape->GetDimension('N');

  Init(context, ksize, stride, padding, data_format);
}
#endif  // INTEL_MKL_ML
// Common Initialization for TensorFlow and MKL formats
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format) {
  // Get the data format
  this->data_format = data_format;

  // Get the output sizes
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');

  // Get the strides
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

  if (depth_window == 1) {  // we are pooling in the H and W
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));
#ifndef INTEL_MKL_ML
    // TF can work with int64, but mkldnn only supports int32
    // Fail if the height or width are greater than MAX_INT

    OP_REQUIRES(context,
                FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output height is too large"));

    OP_REQUIRES(context,
                FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output width is too large"));

#endif
    out_depth = depth;  // output will have the same depth as the input
  } else {              // we are pooling in the depth dimension
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

// Transfers the right parameters for pooling to the op parameters
// Updates context->status if there is an invalid input.
void ExtractMklOpParams(OpKernelContext* context, TensorFormat data_format,
                        const MklPoolParameters& params,
                        MklPoolingOpParams* mkl_params) {
  mkl_params->in_sizes[0] = params.tensor_in_cols;
  mkl_params->in_sizes[1] = params.tensor_in_rows;
  mkl_params->in_sizes[2] = params.depth;
  mkl_params->in_sizes[3] = params.tensor_in_batch;

  GetStridesFromSizes(data_format, mkl_params->in_strides,
                      mkl_params->in_sizes);

  mkl_params->out_sizes[0] = params.out_width;
  mkl_params->out_sizes[1] = params.out_height;
  mkl_params->out_sizes[2] = params.depth;
  mkl_params->out_sizes[3] = params.tensor_in_batch;

  GetStridesFromSizes(data_format, mkl_params->out_strides,
                      mkl_params->out_sizes);

  mkl_params->in_offset[0] = -params.pad_left;
  mkl_params->in_offset[1] = -params.pad_top;
  mkl_params->in_offset[2] = -params.pad_right;
  mkl_params->in_offset[3] = -params.pad_bottom;

  mkl_params->kernel_stride[0] = params.col_stride;
  mkl_params->kernel_stride[1] = params.row_stride;

  mkl_params->kernel_size[0] = params.window_cols;
  mkl_params->kernel_size[1] = params.window_rows;
}
}  // namespace tensorflow
#endif  // INTEL_MKL
