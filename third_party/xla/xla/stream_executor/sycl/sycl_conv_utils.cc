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
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
    dnn::ConvolutionKind conv_kind,
    absl::Span<const DeviceAddressBase> operand_buffers,
    const DeviceAddressBase& result_buffer) {
  auto opaque_ptr = [](const DeviceAddressBase& addr) {
    return const_cast<void*>(addr.opaque());
  };
  if (operand_buffers.size() < 2) {
    return absl::InvalidArgumentError("Insufficient operand buffers");
  }
  void* op0 = opaque_ptr(operand_buffers[0]);
  void* op1 = opaque_ptr(operand_buffers[1]);
  void* res = opaque_ptr(result_buffer);
  switch (conv_kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      return ConvBufferPointers{op0, op1, res};
    case dnn::ConvolutionKind::BACKWARD_DATA:
      return ConvBufferPointers{res, op1, op0};
    case dnn::ConvolutionKind::BACKWARD_FILTER:
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
absl::StatusOr<dnnl::memory> AllocateDnnlBuffer(const dnnl::memory::desc& desc,
                                                const dnnl::engine& engine) {
  return CreateDnnlMemory(desc, engine);
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
    const dnnl::engine& engine, std::optional<ReorderOp>* out_filter_reorder) {
  ConvOp op;
  op.src = std::move(src);
  op.filter = std::move(filter);
  op.dst = std::move(dst);
  ABSL_ASSIGN_OR_RETURN(op.scratchpad,
                   AllocateDnnlBuffer(pd.scratchpad_desc(), engine));
  if (filter_md != weights_target_desc) {
    ABSL_ASSIGN_OR_RETURN(op.internal_filter,
                     AllocateDnnlBuffer(weights_target_desc, engine));
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

absl::StatusOr<OneDnnConvPrimitiveDesc> CreateOneDnnConvPrimitiveDesc(
    const OneDnnConvConfig& config, Stream* stream) {
  OneDnnConvPrimitiveDesc pd;
  ::sycl::queue* sycl_queue =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);

  const DataLayout input_dl = config.input_descriptor.layout();
  const FilterLayout filter_dl = config.filter_descriptor.layout();
  const DataLayout output_dl = config.output_descriptor.layout();

  dnn::DataType input_type;
  const float alpha = config.conv_result_scale;
  const bool alpha_is_one = std::fabs(alpha - 1.0f) < 1e-6;
  const dnn::ConvolutionKind conv_kind = config.kind;
  switch (conv_kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
    case dnn::ConvolutionKind::BACKWARD_FILTER:
      input_type = config.input_type;
      break;
    case dnn::ConvolutionKind::BACKWARD_DATA:
      input_type = config.output_type;
      break;
    default:
      return absl::InvalidArgumentError("Unknown convolution kind");
  }

  // TODO(intel-tf): depthwise-conv
  const int64_t group_count = config.conv_desc.group_count();
  const bool is_group_conv = group_count > 1;
  const int64_t output_channels = config.output_descriptor.feature_map_count();
  const bool is_conv3d = (config.conv_desc.ndims() == 3);

  const absl::Span<const int64_t> padding_dimensions =
      config.conv_desc.padding();
  const absl::Span<const int64_t> stride_dimensions =
      config.conv_desc.strides();
  const absl::Span<const int64_t> dilations_dimensions =
      config.conv_desc.dilations();

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
  ABSL_ASSIGN_OR_RETURN(
      weight_fmt, ToOneDnnFilterFormatTag(filter_dl, is_conv3d, is_group_conv));
  ABSL_ASSIGN_OR_RETURN(dst_fmt, ToOneDnnDataFormatTag(output_dl, is_conv3d));
  ABSL_ASSIGN_OR_RETURN(dnnl::memory::data_type data_type,
                   ToOneDnnDataType(input_type));
  pd.kind = conv_kind;
  try {
    pd.engine = FindOrCreateEngine(sycl_queue);
    pd.src_md = dnnl::memory::desc({src_dims}, data_type, src_fmt);
    pd.dst_md = dnnl::memory::desc({dst_dims}, data_type, dst_fmt);
    pd.filter_md = dnnl::memory::desc({filter_dims}, data_type, weight_fmt);

    // ONEDNN_PLAIN_WEIGHT forces the primitive descriptor to keep the
    // caller's filter layout; otherwise let oneDNN pick an internal
    // pre-packed layout via format_tag::any.
    bool use_plain_weight = false;
    ABSL_RETURN_IF_ERROR(tsl::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false,
                                            &use_plain_weight));
    dnnl::memory::desc filter_md_prefer =
        use_plain_weight
            ? dnnl::memory::desc({filter_dims}, data_type, weight_fmt)
            : dnnl::memory::desc({filter_dims}, data_type,
                                 dnnl::memory::format_tag::any);

    const bool has_fusion = config.fusion.has_value();
    const double side_input_scale =
        has_fusion ? config.fusion->side_input_scale : 0.0;
    pd.has_side_input_sum = has_fusion && std::fabs(side_input_scale) > 1e-6;

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
    // Post-ops are applied in append order.
    dnnl::post_ops po;
    if (!alpha_is_one) {
      po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0);
    }
    if (pd.has_side_input_sum) {
      po.append_sum(side_input_scale);
    }
    if (has_fusion && !alpha_is_one) {
      dnnl::memory::dims bias_post_dims(dst_dims.size(), 1);
      // Logical dimension 1 is always channels (C) in oneDNN.
      bias_post_dims[1] = bias_dims[0];
      dnnl::memory::desc bias_post_md(bias_post_dims, data_type, dst_fmt);
      po.append_binary(dnnl::algorithm::binary_add, bias_post_md);
      pd.bias = OneDnnConvPrimitiveDesc::Bias{
          bias_post_md,
          DNNL_ARG_ATTR_MULTIPLE_POST_OP(po.len() - 1) | DNNL_ARG_SRC_1};
    }
    if (has_fusion) {
      switch (config.fusion->mode) {
        case dnn::kSigmoid:
          po.append_eltwise(dnnl::algorithm::eltwise_logistic, 1, 0);
          break;
        case dnn::kRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
          break;
        case dnn::kRelu6:
          po.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0, 6);
          break;
        case dnn::kTanh:
          po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0, 0);
          break;
        case dnn::kElu:
          po.append_eltwise(dnnl::algorithm::eltwise_elu, 1, 0);
          break;
        case dnn::kLeakyRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu,
                            config.fusion->leakyrelu_alpha, 0);
          break;
        case dnn::kNone:
          break;
        default:
          return absl::InvalidArgumentError("Unsupported Activation mode");
      }
    }
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_post_ops(po);
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (input_type == dnn::DataType::kFloat) {
      post_ops_attr.set_fpmath_mode(GetFP32MathMode());
    }

    // Backward primitives don't take post-ops; their pds only need
    // scratchpad + fpmath, plus a forward hint built with the same attrs.
    dnnl::primitive_attr bwd_attr;
    bwd_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (input_type == dnn::DataType::kFloat) {
      bwd_attr.set_fpmath_mode(GetFP32MathMode());
    }

    switch (conv_kind) {
      case dnn::ConvolutionKind::FORWARD:
      case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
        if (has_fusion && alpha_is_one) {
          pd.bias = OneDnnConvPrimitiveDesc::Bias{
              dnnl::memory::desc(bias_dims, data_type,
                                 dnnl::memory::format_tag::x),
              DNNL_ARG_BIAS};
        }
        std::optional<dnnl::memory::desc> primitive_bias_md;
        if (pd.bias.has_value() && pd.bias->arg_key == DNNL_ARG_BIAS) {
          primitive_bias_md = pd.bias->md;
        }
        pd.conv_pd = CreateConvFwdPd(pd.engine, pd.src_md, filter_md_prefer,
                                     primitive_bias_md, pd.dst_md, stride_dims,
                                     dilation_dims, padding_dims_l,
                                     padding_dims_r, post_ops_attr);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        ConvFwdPd fwd_hint = CreateConvFwdPd(
            pd.engine, pd.src_md, filter_md_prefer, /*bias_md=*/std::nullopt,
            pd.dst_md, stride_dims, dilation_dims, padding_dims_l,
            padding_dims_r, bwd_attr);
        pd.conv_pd = ConvBwdInputPd(
            pd.engine, dnnl::algorithm::convolution_direct, pd.src_md,
            filter_md_prefer, pd.dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, fwd_hint, bwd_attr);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        ConvFwdPd fwd_hint = CreateConvFwdPd(
            pd.engine, pd.src_md, filter_md_prefer, /*bias_md=*/std::nullopt,
            pd.dst_md, stride_dims, dilation_dims, padding_dims_l,
            padding_dims_r, bwd_attr);
        pd.conv_pd = ConvBwdFilterPd(
            pd.engine, dnnl::algorithm::convolution_direct, pd.src_md,
            filter_md_prefer, pd.dst_md, stride_dims, dilation_dims,
            padding_dims_l, padding_dims_r, fwd_hint, bwd_attr);
        break;
      }
      default:
        return absl::InvalidArgumentError("Unknown convolution kind");
    }
  } catch (const dnnl::error& e) {
    return absl::InternalError(absl::StrCat("OneDNN Conv error: ", e.message));
  }
  return pd;
}

absl::StatusOr<OneDnnConvPrimitive> CreateOneDnnConvPrimitive(
    const OneDnnConvPrimitiveDesc& pd,
    absl::Span<const DeviceAddressBase> operand_buffers,
    DeviceAddressBase result_buffer, Stream* stream) {
  OneDnnConvPrimitive onednn_conv_primitive;
  ::sycl::queue* sycl_queue =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  onednn_conv_primitive.engine = pd.engine;

  const dnn::ConvolutionKind conv_kind = pd.kind;
  ABSL_ASSIGN_OR_RETURN(
      ConvBufferPointers buffers,
      GetConvBufferPointers(conv_kind, operand_buffers, result_buffer));

  void* bias_data = nullptr;
  void* side_input_data = nullptr;
  if (pd.bias.has_value() && operand_buffers.size() >= 3) {
    bias_data = const_cast<void*>(operand_buffers[2].opaque());
  }
  if (pd.has_side_input_sum && operand_buffers.size() >= 4) {
    side_input_data = const_cast<void*>(operand_buffers[3].opaque());
  }
  if (pd.has_side_input_sum && side_input_data == nullptr) {
    return absl::InvalidArgumentError(
        "Fused convolution requires a side input buffer");
  }
  if (pd.bias.has_value() && bias_data == nullptr) {
    return absl::InvalidArgumentError(
        "Fused convolution requires a bias buffer");
  }
  try {
    onednn_conv_primitive.stream = dnnl::sycl_interop::make_stream(
        onednn_conv_primitive.engine, *sycl_queue);
    dnnl::memory src_memory = CreateDnnlMemory(
        pd.src_md, onednn_conv_primitive.engine, buffers.input_data);
    dnnl::memory filter_memory = CreateDnnlMemory(
        pd.filter_md, onednn_conv_primitive.engine, buffers.filter_data);
    dnnl::memory dst_memory = CreateDnnlMemory(
        pd.dst_md, onednn_conv_primitive.engine, buffers.output_data);

    // oneDNN's `sum` post-op computes `dst = conv + beta * dst`, reading the
    // current dst buffer. When the side input uses a separate buffer, copy it
    // into dst before each conv execute.
    dnnl::memory side_input_memory;
    if (pd.has_side_input_sum && side_input_data != nullptr &&
        side_input_data != buffers.output_data) {
      side_input_memory = CreateDnnlMemory(
          pd.dst_md, onednn_conv_primitive.engine, side_input_data);
      onednn_conv_primitive.side_input_reorder =
          CreateReorderOp(side_input_memory, dst_memory);
    }

    switch (conv_kind) {
      case dnn::ConvolutionKind::FORWARD:
      case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
        const ConvFwdPd& fwd_pd = std::get<ConvFwdPd>(pd.conv_pd);
        ConvFwd fwd;
        ABSL_ASSIGN_OR_RETURN(
            fwd, (BuildConvOp<ConvFwdPd, dnnl::convolution_forward, ConvFwd>(
                     fwd_pd, std::move(src_memory), DNNL_ARG_SRC,
                     std::move(filter_memory), DNNL_ARG_WEIGHTS,
                     std::move(dst_memory), DNNL_ARG_DST, pd.filter_md,
                     fwd_pd.weights_desc(),
                     /*prepack_filter=*/true, onednn_conv_primitive.engine,
                     &onednn_conv_primitive.filter_reorder)));

        fwd.side_input = std::move(side_input_memory);
        if (pd.bias.has_value() && bias_data != nullptr) {
          fwd.bias = CreateDnnlMemory(pd.bias->md, onednn_conv_primitive.engine,
                                      bias_data);
          fwd.args.insert({pd.bias->arg_key, fwd.bias});
        }
        onednn_conv_primitive.op = std::move(fwd);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        const ConvBwdInputPd& bwd_input_pd =
            std::get<ConvBwdInputPd>(pd.conv_pd);
        ConvBwdData bwd;
        ABSL_ASSIGN_OR_RETURN(
            bwd, (BuildConvOp<ConvBwdInputPd, dnnl::convolution_backward_data,
                              ConvBwdData>(
                     bwd_input_pd, std::move(src_memory), DNNL_ARG_DIFF_SRC,
                     std::move(filter_memory), DNNL_ARG_WEIGHTS,
                     std::move(dst_memory), DNNL_ARG_DIFF_DST, pd.filter_md,
                     bwd_input_pd.weights_desc(),
                     /*prepack_filter=*/true, onednn_conv_primitive.engine,
                     &onednn_conv_primitive.filter_reorder)));
        onednn_conv_primitive.op = std::move(bwd);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        const ConvBwdFilterPd& bwd_filter_pd =
            std::get<ConvBwdFilterPd>(pd.conv_pd);
        ConvBwdWeights bwd;
        ABSL_ASSIGN_OR_RETURN(
            bwd,
            (BuildConvOp<ConvBwdFilterPd, dnnl::convolution_backward_weights,
                         ConvBwdWeights>(
                bwd_filter_pd, std::move(src_memory), DNNL_ARG_SRC,
                std::move(filter_memory), DNNL_ARG_DIFF_WEIGHTS,
                std::move(dst_memory), DNNL_ARG_DIFF_DST, pd.filter_md,
                bwd_filter_pd.diff_weights_desc(),
                /*prepack_filter=*/false, onednn_conv_primitive.engine,
                &onednn_conv_primitive.filter_reorder)));
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

absl::Status DoOneDnnConv(const OneDnnConvPrimitive& onednn_primitive) {
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
