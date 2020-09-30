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

#ifndef TENSORFLOW_CORE_UTIL_MKL_TYPES_H_
#define TENSORFLOW_CORE_UTIL_MKL_TYPES_H_
#ifdef INTEL_MKL

namespace tensorflow {
// MKL DNN 0.x will not be supported. So all related macro's have been removed
// This file will be removed once MKL DNN 0.x related source code is cleaned and
// all MKL DNN 1.x related macro's have been replaced.

#ifdef ENABLE_MKLDNN_V1
#define ADD_MD add_md
#define ALGORITHM mkldnn::algorithm
#define ALGORITHM_UNDEF ALGORITHM::undef
#define BN_FLAGS mkldnn::normalization_flags
#define CPU_STREAM(engine) stream(engine)
#define DATA_WITH_ENGINE(data, engine) data, engine
#define DST_MD dst_md
#define ENGINE_CPU engine::kind::cpu
#define GET_CHECK_REORDER_MEM_ARGS(md, tensor, net, net_args, engine) \
  md, tensor, net, net_args, engine
#define GET_CHECK_REORDER_TO_OP_MEM_ARGS(md, tensor, net, net_args, engine) \
  md, tensor, net, net_args, engine
#define GET_DESC get_desc()
#define GET_FORMAT_FROM_SHAPE(src_mkl_shape) MklTensorFormat::FORMAT_BLOCKED
#define GET_BLOCK_STRIDES(strides, idx) strides
#define GET_MEMORY_DESC_CONSTRUCTOR(dims, type, fm) \
  { {dims}, MklDnnType<type>(), fm }
#define GET_MEMORY_DESC_FROM_MEM_PTR(mem_ptr) mem_ptr->get_desc()
#define GET_MEMORY_PRIMITIVE_DESC_FROM_MEM_PTR(mem_ptr) \
  GET_MEMORY_DESC_FROM_MEM_PTR(mem_ptr)
#define GET_MEMORY_SIZE_FROM_MD(md, engine) md.get_size()
#define GET_SRC_DESC_FROM_OP_PD(op_pd) op_pd->src_desc()
#define GET_DST_DESC_FROM_OP_PD(op_pd) op_pd->dst_desc()
#define GET_BIAS_DESC_FROM_OP_PD(op_pd) op_pd->bias_desc()
#define GET_DIFF_DST_DESC_FROM_OP_PD(op_pd) op_pd->diff_dst_desc()
#define GET_WORKSPACE_DESC_FROM_OP_PD(op_pd) op_pd->workspace_desc()
#define GET_TENSOR_FORMAT(fmt) MklTensorFormatToMklDnnDataFormat(fmt)
#define GET_TF_DATA_FORMAT(shape, mem_desc) shape.GetTfDataFormat()
#define GET_USR_MEM_PRIM_DESC(src) src.GetUsrMemDesc()
#define GET_WEIGHTS_DESC_FROM_OP_PD(op_pd) op_pd->weights_desc()
#define GET_WEIGHTS_FORMAT_FROM_OP_PD(op_pd, op) \
  GET_WEIGHTS_DESC_FROM_OP_PD(op_pd)
#define IS_DIFF_DST_REORDER_NEEDED(diff_dst_md, op_pd, op) \
  diff_dst_md != op_pd->diff_dst_desc()
#define IS_DIFF_FILTER_REORDER_NEEDED(diff_filter_md, fmt, op_pd, op) \
  diff_filter_md != op_pd->diff_weights_desc()
#define IS_FILTER_REORDER_NEEDED(filter_md, op_pd, op) \
  filter_md != op_pd->weights_desc()
#define IS_SRC_REORDER_NEEDED(src_md, op_pd, op) src_md != op_pd->src_desc()
#define IS_WEIGHTS_REORDER_NEEDED(weights_md, op_pd, op) \
  weights_md != op_pd->weights_desc()
#define MEMORY_CONSTRUCTOR(mem_desc, engine, data) \
  memory(mem_desc, engine, data)
#define MEMORY_CONSTRUCTOR_PD(mem_desc, engine, data) \
  MEMORY_CONSTRUCTOR(mem_desc, engine, data)
#define MEMORY_CONSTRUCTOR_USING_MEM_PD(dims, type, fm, engine, data) \
  memory(GET_MEMORY_DESC_CONSTRUCTOR(dims, type, fm), engine, data)
#define MEMORY_CONSTRUCTOR_USING_MD(md, engine, data) memory(md, engine, data)
#define MEMORY_CONSTRUCTOR_WITH_MEM_PD(mem_ptr, cpu_engine, data) \
  memory(GET_MEMORY_DESC_FROM_MEM_PTR(mem_ptr), cpu_engine, data)
#define MEMORY_CONSTRUCTOR_WITHOUT_DATA(mem_desc, engine) \
  memory(mem_desc, engine)
#define MEMORY_DATA_TYPE_UNDEF memory::data_type::undef
#define MEMORY_DESC memory::desc
#define MEMORY_FORMAT mkldnn::memory::format_tag
#define MEMORY_FORMAT_DESC format_desc
#define MEMORY_FORMAT_UNDEF mkldnn::memory::format_tag::undef
#define MEMORY_PD_CONSTRUCTOR(dims, type, fm, engine) \
  memory::desc({dims}, MklDnnType<type>(), fm)
#define MEMORY_PD_WITHOUT_DATA(md, engine) md, engine
#define MEMORY_PRIMITIVE_DESC memory::desc
#define MEMORY_PD_CONSTRUCTOR_2_PARAMS(md, engine) MEMORY_PRIMITIVE_DESC(md)
#define MKL_FMT_TAG mkl_fmt_tag
#define MKL_TENSOR_FORMAT MklTensorFormat
#define MKL_TENSOR_FORMAT_BLOCKED MklTensorFormat::FORMAT_BLOCKED
#define MKL_TENSOR_FORMAT_IN_C MKL_TENSOR_FORMAT
#define MKL_TENSOR_FORMAT_INVALID MklTensorFormat::FORMAT_INVALID
#define MKL_TENSOR_FORMAT_NC MklTensorFormat::FORMAT_NC
#define MKL_TENSOR_FORMAT_NCHW MklTensorFormat::FORMAT_NCHW
#define MKL_TENSOR_FORMAT_NCDHW MklTensorFormat::FORMAT_NCDHW
#define MKL_TENSOR_FORMAT_NDHWC MklTensorFormat::FORMAT_NDHWC
#define MKL_TENSOR_FORMAT_NHWC MklTensorFormat::FORMAT_NHWC
#define MKL_TENSOR_FORMAT_TNC MklTensorFormat::FORMAT_TNC
#define MKL_TENSOR_FORMAT_X MklTensorFormat::FORMAT_X
#define MKL_TENSOR_FORMAT_UNDEF MKL_TENSOR_FORMAT_BLOCKED
#define NET_ARGS_PTR &net_args
#define OUTPUT_TF_MD output_tf_md
#define PRIMITIVE_DESC_BIAS bias_desc()
#define PRIMITIVE_DESC_DIFF_DST diff_dst_desc()
#define PRIMITIVE_DESC_DIFF_SRC diff_src_desc()
#define PRIMITIVE_DESC_DIFF_WEIGHTS diff_weights_desc()
#define PRIMITIVE_DESC_DST dst_desc()
#define PRIMITIVE_DESC_SRC src_desc()
#define PRIMITIVE_DESC_WORKSPACE workspace_desc()
#define PRIMITIVE_DESC_WEIGHTS weights_desc()
#define REORDER_PD_CONSTRUCTOR(src_md, dst_md, engine) \
  ReorderPd(engine, src_md, engine, dst_md)
#define REORDER_PD_CONSTRUCTOR_WITH_ATTR(src_md, dst_md, engine, prim_attr) \
  ReorderPd(engine, src_md, engine, dst_md, prim_attr)
#define SKIP_INPUT_REORDER(input_mkl_shape, input_md) \
  input_mkl_shape.GetTfDataFormat() == MKL_TENSOR_FORMAT_BLOCKED
#define SUMMAND_MD summand_md
#define TENSOR_FORMAT MKL_TENSOR_FORMAT
#define TENSOR_FORMAT_NHWC MKL_TENSOR_FORMAT_NHWC
#define TENSOR_MAX_DIMS MKLDNN_MAX_NDIMS

#endif  // ENABLE_MKLDNN_V1

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_TYPES_H_
