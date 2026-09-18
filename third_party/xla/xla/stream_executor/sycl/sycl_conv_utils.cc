/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_conv_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace sycl {
using dnn::DataLayout;
using dnn::FilterDescriptor;
using dnn::FilterLayout;
using ConvFwdPd = dnnl::convolution_forward::primitive_desc;
using ConvBwdInputPd = dnnl::convolution_backward_data::primitive_desc;
using ConvBwdFilterPd = dnnl::convolution_backward_weights::primitive_desc;

namespace {

ReorderOp CreateReorderOp(const dnnl::memory& src, const dnnl::memory& dst) {
  ReorderOp reorder;
  reorder.primitive = dnnl::reorder(src, dst);
  reorder.args = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
  return reorder;
}

// Pointers to the input, filter, and output buffers of a conv primitive.
//
// Two things depend on the conv primitive kind:
//   * Which XLA buffer backs each slot: see GetConvBufferPointers.
//   * Which DNNL_ARG_* key each slot uses: see CreateConv*Primitive.
//
// Arg-key mapping (`DNNL_ARG_` prefix omitted):
//
//   +---------------+------------+--------------+-------------+
//   | PrimitiveKind | input_data | filter_data  | output_data |
//   +---------------+------------+--------------+-------------+
//   | Fwd/FwdAct    | SRC        | WEIGHTS      | DST         |
//   | BwdInput      | DIFF_SRC   | WEIGHTS      | DIFF_DST    |
//   | BwdFilter     | SRC        | DIFF_WEIGHTS | DIFF_DST    |
//   +---------------+------------+--------------+-------------+
struct ConvBufferPointers {
  void* input_data;
  void* filter_data;
  void* output_data;
};

absl::StatusOr<ConvBufferPointers> GetConvBufferPointers(
    xla::gpu::CudnnConvKind conv_kind,
    absl::Span<const DeviceAddressBase> operand_buffers,
    const DeviceAddressBase& result_buffer) {
  auto opaque_ptr = [](const DeviceAddressBase& addr) {
    return const_cast<void*>(addr.opaque());
  };
  void* op0 = opaque_ptr(operand_buffers[0]);
  void* op1 = opaque_ptr(operand_buffers[1]);
  void* res = opaque_ptr(result_buffer);
  switch (conv_kind) {
    case xla::gpu::CudnnConvKind::kForward:
    case xla::gpu::CudnnConvKind::kForwardActivation:
      return ConvBufferPointers{op0, op1, res};
    case xla::gpu::CudnnConvKind::kBackwardInput:
      return ConvBufferPointers{res, op1, op0};
    case xla::gpu::CudnnConvKind::kBackwardFilter:
      return ConvBufferPointers{op0, res, op1};
    default:
      return absl::InvalidArgumentError("Unknown convolution kind");
  }
}

// Converts XLA's DataLayout enum to oneDNN's memory format tag.
// For 2D convolutions: NCHW (batch, channels, height, width) or NHWC.
// For 3D convolutions: NCDHW (batch, channels, depth, height, width)
// or NDHWC.
// Note: XLA's DataLayout "Depth" refers to feature channels. In oneDNN,
// this is called "Channel" (C). oneDNN's "Depth" (D) refers to the additional
// spatial dimension in 3D/volumetric convolutions.
absl::StatusOr<dnnl::memory::format_tag> ToOneDnnDataFormatTag(
    DataLayout layout, bool is_conv3d) {
  switch (layout) {
    case DataLayout::kBatchDepthYX:
      return is_conv3d ? dnnl::memory::format_tag::ncdhw
                       : dnnl::memory::format_tag::nchw;
    case DataLayout::kBatchYXDepth:
      return is_conv3d ? dnnl::memory::format_tag::ndhwc
                       : dnnl::memory::format_tag::nhwc;
    default:
      return absl::InvalidArgumentError("Unsupported data layout");
  }
}

// Converts XLA's FilterLayout enum to oneDNN's memory format tag for
// filter/weight tensors.
// For 2D: OIHW (output channels, input channels, height, width) or variations.
// For 3D: OIDHW (output, input, depth, height, width) or variations.
// Grouped convolutions add 'g' prefix (e.g., GOIHW: groups, output per group,
// input per group, height, width).
absl::StatusOr<dnnl::memory::format_tag> ToOneDnnFilterFormatTag(
    FilterLayout layout, bool is_conv3d, bool is_group_conv) {
  switch (layout) {
    case FilterLayout::kOutputInputYX:
      if (is_conv3d) {
        return is_group_conv ? dnnl::memory::format_tag::goidhw
                             : dnnl::memory::format_tag::oidhw;
      }
      return is_group_conv ? dnnl::memory::format_tag::goihw
                           : dnnl::memory::format_tag::oihw;
    case FilterLayout::kOutputYXInput:
      if (is_conv3d) {
        return is_group_conv ? dnnl::memory::format_tag::godhwi
                             : dnnl::memory::format_tag::odhwi;
      }
      return is_group_conv ? dnnl::memory::format_tag::gohwi
                           : dnnl::memory::format_tag::ohwi;
    case FilterLayout::kYXInputOutput:
      if (is_conv3d) {
        return absl::InvalidArgumentError("Unsupported conv weight format");
      }
      return is_group_conv ? dnnl::memory::format_tag::hwigo
                           : dnnl::memory::format_tag::hwio;
    default:
      return absl::InvalidArgumentError("Unsupported conv weight format");
  }
}

// Builds the oneDNN filter dims from an XLA FilterDescriptor. For a plain
// conv the shape is [O, I, spatial...]; for group conv it becomes
// [G, O/G, I, spatial...]. `group_count` == 1 keeps the plain shape.
dnnl::memory::dims ToOneDnnFilterDims(const FilterDescriptor& descriptor,
                                      int64_t group_count) {
  std::vector<int64_t> dims =
      descriptor.full_dims(FilterLayout::kOutputInputYX);
  if (group_count > 1) {
    dims[0] /= group_count;                  // O -> O/G
    dims.insert(dims.begin(), group_count);  // prepend G
  }
  return dnnl::memory::dims(dims.begin(), dims.end());
}

// Allocates a temporary buffer and wraps it in a dnnl::memory.
// Such buffers are used for oneDNN scratchpad and pre-packed filter.
absl::StatusOr<dnnl::memory> AllocateDnnlBuffer(
    const dnnl::memory::desc& desc, const dnnl::engine& engine,
    ScratchAllocator* scratch_allocator) {
  stream_executor::DeviceMemory<uint8_t> buffer;
  ABSL_ASSIGN_OR_RETURN(buffer, scratch_allocator->AllocateBytes(desc.get_size()));
  return CreateDnnlMemory(desc, engine, buffer.opaque());
}

// Builds a forward convolution primitive descriptor. `bias_md` is present only
// when bias is fused into the pd (forward path with alpha == 1);
// omit it otherwise. `post_ops_attr` defaults to empty for the bwd pds.
ConvFwdPd CreateConvFwdPd(
    const dnnl::engine& engine, const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& filter_md_prefer,
    const std::optional<dnnl::memory::desc>& bias_md,
    const dnnl::memory::desc& dst_md, const dnnl::memory::dims& stride_dims,
    const dnnl::memory::dims& dilation_dims,
    const dnnl::memory::dims& padding_dims_l,
    const dnnl::memory::dims& padding_dims_r,
    const dnnl::primitive_attr& post_ops_attr = dnnl::primitive_attr()) {
  if (bias_md.has_value()) {
    return ConvFwdPd(
        engine, dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
        src_md, filter_md_prefer, *bias_md, dst_md, stride_dims, dilation_dims,
        padding_dims_l, padding_dims_r, post_ops_attr);
  }
  return ConvFwdPd(engine, dnnl::prop_kind::forward,
                   dnnl::algorithm::convolution_direct, src_md,
                   filter_md_prefer, dst_md, stride_dims, dilation_dims,
                   padding_dims_l, padding_dims_r, post_ops_attr);
}

// Constructs a convolution operation with distinct oneDNN primitive kind
// selected by the template arguments (forward, backward-data, or
// backward-filter). Wires up the src, filter, and destination memory,
// allocates the scratchpad, and pre-packs the filter when the primitive's
// preferred layout does not match the one the caller provides.
template <typename Pd, typename Primitive, typename ConvOp>
absl::StatusOr<ConvOp> BuildConvOp(
    const Pd& pd, dnnl::memory src, int src_arg_key, dnnl::memory filter,
    int weights_arg_key, dnnl::memory dst, int dst_arg_key,
    const dnnl::memory::desc& filter_md,
    const dnnl::memory::desc& weights_target_desc, bool prepack_filter,
    const dnnl::engine& engine, ScratchAllocator* scratch_allocator,
    std::optional<ReorderOp>* out_filter_reorder) {
  ConvOp op;
  op.src = std::move(src);
  op.filter = std::move(filter);
  op.dst = std::move(dst);
  ABSL_ASSIGN_OR_RETURN(
      op.scratchpad,
      AllocateDnnlBuffer(pd.scratchpad_desc(), engine, scratch_allocator));
  if (filter_md != weights_target_desc) {
    ABSL_ASSIGN_OR_RETURN(
        op.internal_filter,
        AllocateDnnlBuffer(weights_target_desc, engine, scratch_allocator));
    *out_filter_reorder = prepack_filter
                              ? CreateReorderOp(op.filter, op.internal_filter)
                              : CreateReorderOp(op.internal_filter, op.filter);
    op.args.insert({weights_arg_key, op.internal_filter});
  } else {
    op.args.insert({weights_arg_key, op.filter});
  }
  op.args.insert({src_arg_key, op.src});
  op.args.insert({dst_arg_key, op.dst});
  op.args.insert({DNNL_ARG_SCRATCHPAD, op.scratchpad});
  op.primitive = Primitive(pd);
  return op;
}
}  // namespace

absl::StatusOr<OneDnnConvPrimitive> CreateOneDnnConvPrimitive(
    const xla::gpu::GpuConvConfig& config,
    absl::Span<const DeviceAddressBase> operand_buffers,
    DeviceAddressBase result_buffer, Stream* stream,
    ScratchAllocator* scratch_allocator) {
  OneDnnConvPrimitive onednn_conv_primitive;
  ::sycl::queue* sycl_queue =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  onednn_conv_primitive.engine = FindOrCreateEngine(sycl_queue);
  onednn_conv_primitive.stream = dnnl::sycl_interop::make_stream(
      onednn_conv_primitive.engine, *sycl_queue);

  DataLayout input_dl = config.input_descriptor.layout();
  FilterLayout filter_dl = config.filter_descriptor.layout();
  DataLayout output_dl = config.output_descriptor.layout();

  xla::PrimitiveType input_type;
  void* input_data;
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;

  float alpha = config.conv_result_scale;
  bool alpha_is_one = (fabs(alpha - 1.0f) < 1e-6);
  xla::gpu::CudnnConvKind conv_kind = config.kind;

  // Get input type based on convolution kind
  switch (conv_kind) {
    case xla::gpu::CudnnConvKind::kForward:
    case xla::gpu::CudnnConvKind::kForwardActivation:
    case xla::gpu::CudnnConvKind::kBackwardFilter:
      input_type = config.input_type;
      break;
    case xla::gpu::CudnnConvKind::kBackwardInput:
      input_type = config.output_type;
      break;
    default:
      return absl::InvalidArgumentError("Unknown convolution kind");
  }

  // Get buffer pointers
  ABSL_ASSIGN_OR_RETURN(
      auto buffers,
      GetConvBufferPointers(conv_kind, operand_buffers, result_buffer));
  input_data = buffers.input_data;
  filter_data = buffers.filter_data;
  output_data = buffers.output_data;

  float side_input_scale = 0.0f;
  bool side_input_scale_zero = true;
  if (conv_kind == xla::gpu::CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].opaque());
      side_input_scale = config.fusion->side_input_scale;
      side_input_scale_zero = (fabs(side_input_scale - 0.0f) < 1e-6);
    }
  }

  // TODO(intel-tf): depthwise-conv
  const int64_t group_count = config.conv_desc.group_count();
  const bool is_group_conv = group_count > 1;
  const int64_t output_channels = config.output_descriptor.feature_map_count();

  absl::Span<const int64_t> padding_dimensions = config.conv_desc.padding();
  absl::Span<const int64_t> stride_dimensions = config.conv_desc.strides();
  absl::Span<const int64_t> dilations_dimensions = config.conv_desc.dilations();

  bool is_conv3d = (config.conv_desc.ndims() == 3);
  try {
    std::vector<int64_t> src_full =
        config.input_descriptor.full_dims(DataLayout::kBatchDepthYX);
    std::vector<int64_t> dst_full =
        config.output_descriptor.full_dims(DataLayout::kBatchDepthYX);
    dnnl::memory::dims src_dims(src_full.begin(), src_full.end());
    dnnl::memory::dims dst_dims(dst_full.begin(), dst_full.end());
    dnnl::memory::dims filter_dims =
        ToOneDnnFilterDims(config.filter_descriptor, group_count);
    dnnl::memory::dims bias_dims = {output_channels};
    dnnl::memory::dims stride_dims(stride_dimensions.begin(),
                                   stride_dimensions.end());
    dnnl::memory::dims padding_dims_l(padding_dimensions.begin(),
                                      padding_dimensions.end());
    dnnl::memory::dims padding_dims_r = padding_dims_l;
    dnnl::memory::dims dilation_dims(dilations_dimensions.size());
    std::transform(dilations_dimensions.begin(), dilations_dimensions.end(),
                   dilation_dims.begin(), [](int64_t d) { return d - 1; });
    dnnl::memory::format_tag src_fmt, weight_fmt, dst_fmt;
    ABSL_ASSIGN_OR_RETURN(src_fmt, ToOneDnnDataFormatTag(input_dl, is_conv3d));
    ABSL_ASSIGN_OR_RETURN(weight_fmt, ToOneDnnFilterFormatTag(filter_dl, is_conv3d,
                                                         is_group_conv));
    ABSL_ASSIGN_OR_RETURN(dst_fmt, ToOneDnnDataFormatTag(output_dl, is_conv3d));
    ABSL_ASSIGN_OR_RETURN(dnnl::memory::data_type data_type,
                     ToOneDnnDataType(input_type));

    dnnl::memory::desc src_md =
        dnnl::memory::desc({src_dims}, data_type, src_fmt);
    dnnl::memory::desc filter_md =
        dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
    dnnl::memory::desc dst_md =
        dnnl::memory::desc({dst_dims}, data_type, dst_fmt);

    bool use_plain_weight = false;
    ABSL_RETURN_IF_ERROR(tsl::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false,
                                            &use_plain_weight));
    dnnl::memory::desc filter_md_prefer = dnnl::memory::desc(
        {filter_dims}, data_type, dnnl::memory::format_tag::any);
    if (use_plain_weight) {
      filter_md_prefer =
          dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
    }

    dnnl::memory src_memory =
        CreateDnnlMemory(src_md, onednn_conv_primitive.engine, input_data);
    dnnl::memory filter_memory =
        CreateDnnlMemory(filter_md, onednn_conv_primitive.engine, filter_data);
    dnnl::memory dst_memory =
        CreateDnnlMemory(dst_md, onednn_conv_primitive.engine, output_data);

    // oneDNN's `sum` post-op computes `dst = conv + beta * dst`, reading the
    // current dst buffer. The side input lives in a separate operand, so we
    // create a reorder that copies it into dst before each conv execute.
    dnnl::memory side_input_memory;
    if (side_input_data && !side_input_scale_zero) {
      side_input_memory = CreateDnnlMemory(dst_md, onednn_conv_primitive.engine,
                                           side_input_data);
      onednn_conv_primitive.side_input_reorder =
          CreateReorderOp(side_input_memory, dst_memory);
    }

    // Fused forward conv computes `out = activation(alpha * conv(x, w) +
    // beta * side + bias)`. How each term is expressed depends on `alpha`:
    //
    //   alpha == 1: bias is fused into the pd (DNNL_ARG_BIAS); the post-op
    //               chain reduces to (sum beta, activation).
    //   alpha != 1: bias-in-pd is not supported, so alpha and bias are both
    //               applied as post-ops. The chain is
    //               (eltwise_linear alpha, sum beta, binary_add bias,
    //                activation).
    //
    // Post-ops are applied in append order, and the terms below are appended
    // to `po` accordingly.
    dnnl::post_ops po;
    dnnl::primitive_attr post_ops_attr;
    if (!alpha_is_one) {
      po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0);
    }
    if (side_input_data && !side_input_scale_zero) {
      po.append_sum(side_input_scale);
    }
    // Post-op bias goes only on the forward path; staged here, applied below.
    dnnl::memory post_op_bias_memory;
    int post_op_bias_arg_key = 0;
    bool has_post_op_bias = false;
    // Bias can be fused directly into the primitive descriptor (via
    // bias_md) only when alpha == 1. Otherwise, it's applied as
    // a post-op.
    if (!alpha_is_one && bias_data) {
      dnnl::memory::dims bias_post_dims(dst_dims.size(), 1);
      bias_post_dims[1] =
          bias_dims[0];  // Logical dimension 1 is always channels (C) in oneDNN
      auto bias_post_md =
          dnnl::memory::desc(bias_post_dims, data_type, dst_fmt);
      po.append_binary(dnnl::algorithm::binary_add, bias_post_md);
      post_op_bias_memory = CreateDnnlMemory(
          bias_post_md, onednn_conv_primitive.engine, bias_data);
      post_op_bias_arg_key =
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(po.len() - 1) | DNNL_ARG_SRC_1;
      has_post_op_bias = true;
    }
    if (conv_kind == xla::gpu::CudnnConvKind::kForwardActivation) {
      switch (config.fusion->mode) {
        case stream_executor::dnn::kSigmoid:
          po.append_eltwise(dnnl::algorithm::eltwise_logistic, 1, 0);
          break;
        case stream_executor::dnn::kRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
          break;
        case stream_executor::dnn::kRelu6:
          po.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0, 6);
          break;
        case stream_executor::dnn::kTanh:
          po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0, 0);
          break;
        case stream_executor::dnn::kElu:
          po.append_eltwise(dnnl::algorithm::eltwise_elu, 1, 0);
          break;
        case stream_executor::dnn::kLeakyRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu,
                            config.fusion->leakyrelu_alpha, 0);
          break;
        case stream_executor::dnn::kNone:
          break;
        default:
          return absl::InvalidArgumentError("Unsupported Activation mode");
      }
    }
    post_ops_attr.set_post_ops(po);
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // Set fp32 mode.
    dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
    if (input_type == xla::F32) {
      post_ops_attr.set_fpmath_mode(fp32_math_mode);
    }
    switch (conv_kind) {
      case xla::gpu::CudnnConvKind::kForward:
      case xla::gpu::CudnnConvKind::kForwardActivation: {
        // Bias can be fused directly into the primitive descriptor (via
        // bias_md) only when alpha == 1.
        std::optional<dnnl::memory::desc> bias_md;
        if (bias_data != nullptr && alpha_is_one) {
          bias_md = dnnl::memory::desc(bias_dims, data_type,
                                       dnnl::memory::format_tag::x);
        }
        // Create the convolution forward primitive descriptor with the
        // appropriate bias memory descriptor.
        ConvFwdPd fwd_pd = CreateConvFwdPd(
            onednn_conv_primitive.engine, src_md, filter_md_prefer, bias_md,
            dst_md, stride_dims, dilation_dims, padding_dims_l, padding_dims_r,
            post_ops_attr);

        // Build a convolution primitive
        ConvFwd fwd;
        ABSL_ASSIGN_OR_RETURN(
            fwd,
            (BuildConvOp<ConvFwdPd, dnnl::convolution_forward, ConvFwd>(
                fwd_pd, std::move(src_memory), DNNL_ARG_SRC,
                std::move(filter_memory), DNNL_ARG_WEIGHTS,
                std::move(dst_memory), DNNL_ARG_DST, filter_md,
                fwd_pd.weights_desc(),
                /*prepack_filter=*/true, onednn_conv_primitive.engine,
                scratch_allocator, &onednn_conv_primitive.filter_reorder)));

        fwd.side_input = std::move(side_input_memory);
        if (bias_md.has_value()) {
          fwd.bias = CreateDnnlMemory(bias_md.value(),
                                      onednn_conv_primitive.engine, bias_data);
          fwd.args.insert({DNNL_ARG_BIAS, fwd.bias});
        }
        if (has_post_op_bias) {
          fwd.bias = std::move(post_op_bias_memory);
          fwd.args.insert({post_op_bias_arg_key, fwd.bias});
        }
        onednn_conv_primitive.op = std::move(fwd);
        break;
      }
      case xla::gpu::CudnnConvKind::kBackwardInput: {
        // Create a forward convolution primitive descriptor for the backward
        // input convolution.
        dnnl::primitive_attr attr;
        if (input_type == xla::F32) {
          attr.set_fpmath_mode(fp32_math_mode);
        }
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        ConvFwdPd fwd_pd = CreateConvFwdPd(
            onednn_conv_primitive.engine, src_md, filter_md_prefer,
            /*bias_md=*/std::nullopt, dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, attr);
        ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
            onednn_conv_primitive.engine, dnnl::algorithm::convolution_direct,
            src_md, filter_md_prefer, dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, fwd_pd, attr);

        ConvBwdData bwd;
        ABSL_ASSIGN_OR_RETURN(
            bwd,
            (BuildConvOp<ConvBwdInputPd, dnnl::convolution_backward_data,
                         ConvBwdData>(
                bwd_input_pd, std::move(src_memory), DNNL_ARG_DIFF_SRC,
                std::move(filter_memory), DNNL_ARG_WEIGHTS,
                std::move(dst_memory), DNNL_ARG_DIFF_DST, filter_md,
                bwd_input_pd.weights_desc(),
                /*prepack_filter=*/true, onednn_conv_primitive.engine,
                scratch_allocator, &onednn_conv_primitive.filter_reorder)));

        onednn_conv_primitive.op = std::move(bwd);
        break;
      }
      case xla::gpu::CudnnConvKind::kBackwardFilter: {
        // Create a forward convolution primitive descriptor for the backward
        // weights convolution.
        dnnl::primitive_attr attr;
        if (input_type == xla::F32) {
          attr.set_fpmath_mode(fp32_math_mode);
        }
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        ConvFwdPd fwd_pd = CreateConvFwdPd(
            onednn_conv_primitive.engine, src_md, filter_md_prefer,
            /*bias_md=*/std::nullopt, dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, attr);
        ConvBwdFilterPd bwd_filter_pd = ConvBwdFilterPd(
            onednn_conv_primitive.engine, dnnl::algorithm::convolution_direct,
            src_md, filter_md_prefer, dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, fwd_pd, attr);
        ConvBwdWeights bwd;
        ABSL_ASSIGN_OR_RETURN(
            bwd,
            (BuildConvOp<ConvBwdFilterPd, dnnl::convolution_backward_weights,
                         ConvBwdWeights>(
                bwd_filter_pd, std::move(src_memory), DNNL_ARG_SRC,
                std::move(filter_memory), DNNL_ARG_DIFF_WEIGHTS,
                std::move(dst_memory), DNNL_ARG_DIFF_DST, filter_md,
                bwd_filter_pd.diff_weights_desc(),
                /*prepack_filter=*/false, onednn_conv_primitive.engine,
                scratch_allocator, &onednn_conv_primitive.filter_reorder)));

        onednn_conv_primitive.op = std::move(bwd);
        break;
      }
      default:
        return absl::InvalidArgumentError("Unknown convolution kind");
    }
  } catch (dnnl::error& e) {
    return absl::InternalError(absl::StrCat("OneDNN Conv error: ", e.message));
  }
  return onednn_conv_primitive;
}

absl::Status DoOnednnConv(const OneDnnConvPrimitive& onednn_primitive) {
  try {
    auto execute_reorder = [&](const std::optional<ReorderOp>& reorder) {
      if (reorder) {
        reorder->primitive.execute(onednn_primitive.stream, reorder->args);
      }
    };
    std::visit(
        [&](const auto& op) {
          using T = std::decay_t<decltype(op)>;
          if constexpr (std::is_same_v<T, ConvFwd>) {
            execute_reorder(onednn_primitive.filter_reorder);
            // The `sum` post-op accumulates into dst in place, so stage the
            // side input there before running the conv.
            execute_reorder(onednn_primitive.side_input_reorder);
            op.primitive.execute(onednn_primitive.stream, op.args);
          } else if constexpr (std::is_same_v<T, ConvBwdData>) {
            execute_reorder(onednn_primitive.filter_reorder);
            op.primitive.execute(onednn_primitive.stream, op.args);
          } else {
            op.primitive.execute(onednn_primitive.stream, op.args);
            execute_reorder(onednn_primitive.filter_reorder);
          }
        },
        onednn_primitive.op);
  } catch (dnnl::error& e) {
    return absl::InternalError(
        absl::StrCat("OneDNN Conv execution error: ", e.message));
  }

  return absl::OkStatus();
}

}  // namespace sycl
}  // namespace stream_executor
