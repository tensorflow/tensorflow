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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kNHWC[] = "NHWC";
constexpr char kNCHW[] = "NCHW";
constexpr float kGPURatioThreshold = 0.5;
constexpr float kConvGPUExpectedDtypeThreshold = 0.5;

struct MutableNodeViewFormatter {
  void operator()(std::string* out, utils::MutableNodeView* node_view) const {
    absl::StrAppend(out, node_view->node()->name());
  }
};

struct GpuStats {
  int num_gpus;
  int num_voltas;
  int num_amperes;
};

inline GpuStats GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  GpuStats gpu_stats{};
  for (const auto& device : devices) {
    if (device.second.type() != kGPU) {
      continue;
    }
    gpu_stats.num_gpus++;
    auto compute_capability_it =
        device.second.environment().find("architecture");
    if (compute_capability_it == device.second.environment().end()) {
      continue;
    }
    double compute_capability = 0.0;
    if (absl::SimpleAtod(compute_capability_it->second, &compute_capability)) {
      if (compute_capability >= 7.0) gpu_stats.num_voltas++;
      if (compute_capability >= 8.0) gpu_stats.num_amperes++;
    }
  }
  return gpu_stats;
}

inline bool ConvBackpropExists(const TransposeContext& context,
                               absl::string_view device,
                               const DataType& data_type) {
  for (const auto& node : context.graph_view->GetNodes()) {
    const auto* node_def = node.node();
    if (!IsConv2DBackpropFilter(*node_def) &&
        !IsConv2DBackpropInput(*node_def) &&
        !IsConv3DBackpropFilterV2(*node_def) &&
        !IsConv3DBackpropInputV2(*node_def)) {
      continue;
    }

    const string& device_name = GetDeviceName(*node_def);
    string device_type;
    string task;
    if (!DeviceNameUtils::SplitDeviceName(device_name, &task, &device_type) ||
        !absl::StrContains(absl::AsciiStrToLower(device_type),
                           absl::AsciiStrToLower(device))) {
      continue;
    }

    const auto* t_attr = node.GetAttr("T");
    if (t_attr == nullptr) {
      continue;
    }
    if (t_attr->type() == data_type) {
      return true;
    }
  }
  return false;
}

inline std::pair<string, string> GetSrcAndDstDataFormats(
    const TransposeContext& context, GpuStats gpu_stats) {
  string src_format = kNHWC;
  string dst_format = kNCHW;

  const bool is_NHWC_enforced =
      (!context.enforced_layout.empty() && context.enforced_layout == "NHWC");
  const bool volta_ready =
      (static_cast<float>(gpu_stats.num_voltas) /
       static_cast<float>(gpu_stats.num_gpus)) >= kGPURatioThreshold;
  const bool ampere_ready =
      (static_cast<float>(gpu_stats.num_amperes) /
       static_cast<float>(gpu_stats.num_gpus)) >= kGPURatioThreshold;

  // We swap the src_format and dst_format when >= 50% of gpu conv nodes are
  //   (1): half-dtype and we are tuning for Volta+ GPUs
  //   (2): TF32-dtype with TensorCores enabled and tuning for Ampere+ GPUs
  //        (but only if no backward conv in fp32 exists)
  //   (3): blfoat16-dtype and tuning for Ampere+ GPUs
  int num_conv_gpu = 0;
  int num_conv_gpu_prefer_swap = 0;
  bool fp32_backprop = ConvBackpropExists(context, kGPU, DT_FLOAT);

  for (const auto& node : context.graph_view->GetNodes()) {
    const auto* node_def = node.node();
    if (!IsConv2D(*node_def) && !IsConv3D(*node_def)) {
      continue;
    }
    const string& device_name = GetDeviceName(*node_def);
    string device_type;
    string task;
    if (!DeviceNameUtils::SplitDeviceName(device_name, &task, &device_type) ||
        !absl::StrContains(absl::AsciiStrToLower(device_type),
                           absl::AsciiStrToLower(kGPU))) {
      continue;
    }
    num_conv_gpu++;
    const auto* t_attr = node.GetAttr("T");
    if (t_attr == nullptr) {
      continue;
    }
    const DataType dtype = t_attr->type();
    if ((volta_ready && dtype == DT_HALF) ||
        (ampere_ready && dtype == DT_BFLOAT16) ||
        (ampere_ready && dtype == DT_FLOAT &&
         tsl::tensor_float_32_execution_enabled() && !fp32_backprop)) {
      num_conv_gpu_prefer_swap++;
    }
  }

  // Check ratio of ops preferring swap.
  const bool should_swap =
      num_conv_gpu > 0 &&
      (static_cast<float>(num_conv_gpu_prefer_swap) /
       static_cast<float>(num_conv_gpu)) >= kConvGPUExpectedDtypeThreshold;

  // We swap only if NHWC is enforced or no layout is enforced and the devices
  // config meet the thresholds
  if (is_NHWC_enforced || (context.enforced_layout.empty() && should_swap)) {
    std::swap(src_format, dst_format);
  }

  VLOG(2) << "Layout conversion of " << src_format << " to " << dst_format
          << " will take place.";

  return {src_format, dst_format};
}

Status ExpandLayoutSensitiveOp(TransposeContext* context,
                               TransposerFactory* transposer_factory) {
  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutSensitiveOp(*node_def)) {
      std::shared_ptr<Transposer> transposer =
          transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            absl::StatusCode::kNotFound,
            absl::StrCat(
                "Layout sensitive operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return absl::OkStatus();
}

Status ExpandLayoutAgnosticOp(TransposeContext* context,
                              TransposerFactory* transposer_factory) {
  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutAgnosticOp(*node_def)) {
      const auto& transposer = transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            absl::StatusCode::kNotFound,
            absl::StrCat(
                "Layout agnostic operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return absl::OkStatus();
}

inline bool IsCancellableConstPermTransposeNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  Tensor fanout_tensor;
  if (!GetValueAttrFromConstInputNode(fanout_transpose, IsTranspose, 1,
                                      &fanout_tensor)) {
    return false;
  }
  Tensor fanin_tensor;
  if (!GetValueAttrFromConstInputNode(fanin_transpose, IsTranspose, 1,
                                      &fanin_tensor)) {
    return false;
  }
  if (fanout_tensor.NumElements() != fanin_tensor.NumElements()) {
    return false;
  }

  // Using dst->src to permute on src->dst will result in
  // seq(0, ..., num_elements - 1) if they are cancellable.
  const auto& fanout_tensor_data = fanout_tensor.unaligned_flat<int32>();
  const auto& fanin_tensor_data = fanin_tensor.unaligned_flat<int32>();
  const int num_elements = fanout_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    if (fanout_tensor_data(fanin_tensor_data(i)) != i) {
      return false;
    }
  }
  return true;
}

inline bool IsCancellableDataFormatNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  if (!IsDataFormatOp(fanout_transpose) || !IsDataFormatOp(fanin_transpose)) {
    return false;
  }

  auto src_dst_match = [](const utils::MutableNodeView& src,
                          const utils::MutableNodeView& dst) {
    const auto* src_format = src.GetAttr(kAttrSrcFormat);
    if (src_format == nullptr) {
      return false;
    }
    const auto* dst_format = dst.GetAttr(kAttrDstFormat);
    if (dst_format == nullptr) {
      return false;
    }
    return src_format->s() == dst_format->s();
  };

  // If src_format node A is equal to dst_format of node B and dst_format of
  // node A is equal to src_format of node B, then they are cancellable.
  return src_dst_match(fanin_transpose, fanout_transpose) &&
         src_dst_match(fanout_transpose, fanin_transpose);
}

inline bool IsCancellableNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  return IsCancellableConstPermTransposeNodePair(fanout_transpose,
                                                 fanin_transpose) ||
         IsCancellableDataFormatNodePair(fanout_transpose, fanin_transpose);
}

Status EraseCancellableNodes(TransposeContext* context) {
  const int original_num_nodes = context->num_nodes;
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();

  for (int i = original_num_nodes; i < num_nodes; ++i) {
    auto* node = graph_view->GetNode(i);
    if (node->NumRegularFanins() < 1) {
      continue;
    }
    const auto& regular_fanin_0 = node->GetRegularFanin(0);
    auto* fanin_node = regular_fanin_0.node_view();
    // TODO(lyandy): Lift restriction once original nodes in the graph can be
    // pruned away.
    if (fanin_node->node_index() < original_num_nodes) {
      continue;
    }
    if (!IsCancellableNodePair(*node, *fanin_node)) {
      continue;
    }
    const auto& fanin_to_forward = fanin_node->GetRegularFanin(0);
    TensorId fanin_id_to_forward(fanin_to_forward.node_view()->GetName(),
                                 fanin_to_forward.index());
    for (const auto& regular_fanout : node->GetRegularFanout(0)) {
      mutation->AddOrUpdateRegularFanin(regular_fanout.node_view(),
                                        regular_fanout.index(),
                                        fanin_id_to_forward);
    }
    mutation->RemoveNode(node);
    if (node->NumRegularFanins() > 1) {
      mutation->RemoveNode(node->GetRegularFanin(1).node_view());
    }
    mutation->RemoveNode(fanin_node);
    if (fanin_node->NumRegularFanins() > 1) {
      mutation->RemoveNode(fanin_node->GetRegularFanin(1).node_view());
    }
  }
  return mutation->Apply();
}

// TODO(ezhulenev): This is a temporary workaround for a graph pattern
// in Resnet models. We should be able to push down transpose nodes across Pad
// and many other ops, and then rely on cancellation to remove them.
//
// From: Transpose[NHWC->NCHW] -> Pad[paddings] -> Transpose[NCHW->NHWC]
// To:   Pad[Permute(paddings)]
Status EraseCancellableNodesAroundPad(TransposeContext* context) {
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  absl::flat_hash_set<utils::MutableNodeView*> cancelled_transposes;

  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    // Transpose node after Pad.
    auto* transpose_after = graph_view->GetNode(i);
    if (!IsTranspose(*transpose_after->node())) continue;

    // This transpose was already cancelled in previous loop iteration.
    if (cancelled_transposes.contains(transpose_after)) continue;

    // Pad node.
    const auto& transpose_after_fanin = transpose_after->GetRegularFanin(0);
    auto* pad = transpose_after_fanin.node_view();
    if (!IsPad(*pad->node())) continue;

    // Transpose node before Pad.
    const auto& pad_fanin_0 = pad->GetRegularFanin(0);
    auto* transpose_before = pad_fanin_0.node_view();
    if (!IsTranspose(*transpose_before->node())) continue;

    // Transpose before output used once by the Pad node.
    if (transpose_before->NumRegularFanouts() != 1) continue;

    // Transposes are cancellable.
    if (!IsCancellableConstPermTransposeNodePair(*transpose_after,
                                                 *transpose_before))
      continue;

    // Paddings are known constant values.
    Tensor paddings_t;
    if (!GetValueAttrFromConstInputNode(*pad, IsPad, 1, &paddings_t)) continue;

    // Paddings value used once by the pad node only.
    const auto& pad_fanin_1 = pad->GetRegularFanin(1);
    auto* paddings = pad_fanin_1.node_view();
    if (paddings->NumRegularFanouts() != 1) continue;

    // Get permutation after the padding.
    Tensor permute_t;
    if (!GetValueAttrFromConstInputNode(*transpose_after, IsTranspose, 1,
                                        &permute_t))
      continue;

    // Pad output might be used multiple times by different Transpose nodes. If
    // they all have identical permutation, we can cancel all of them.
    std::vector<utils::MutableNodeView*> pad_fanout_transposes;
    pad_fanout_transposes.emplace_back(transpose_after);

    bool pad_has_unsupported_fanout = false;
    for (auto& fanout : pad->GetRegularFanout(0)) {
      auto* extra_transpose = fanout.node_view();
      if (extra_transpose == transpose_after) continue;

      // Check that fanout is a Transpose identical to the transpose_after.
      Tensor extra_permute_t;
      if (!GetValueAttrFromConstInputNode(*extra_transpose, IsTranspose, 1,
                                          &extra_permute_t) ||
          extra_permute_t.tensor_data() != permute_t.tensor_data()) {
        pad_has_unsupported_fanout = true;
        break;
      }

      pad_fanout_transposes.emplace_back(extra_transpose);
    }
    if (pad_has_unsupported_fanout) continue;

    VLOG(0) << "Cancel Transpose nodes around Pad:"
            << " transpose_before=" << transpose_before->node()->name()
            << " pad=" << pad->node()->name() << " transpose_after="
            << absl::StrJoin(pad_fanout_transposes, ",",
                             MutableNodeViewFormatter());

    // Permute paddings in place according to permutation in second transpose.
    auto permutation_s = absl::Span<int32>(permute_t.flat<int32>().data(),
                                           permute_t.NumElements());
    auto paddings_s = absl::Span<int32>(paddings_t.flat<int32>().data(),
                                        paddings_t.NumElements());
    TF_RETURN_IF_ERROR(
        PermuteDouble(absl::StrCat("paddings in ", pad->GetName()),
                      permutation_s, &paddings_s));

    // Update paddings constant value with a permuted tensor.
    AttrValue permuted_paddings_tensor;
    paddings_t.AsProtoTensorContent(permuted_paddings_tensor.mutable_tensor());
    mutation->AddOrUpdateNodeAttr(paddings, "value", permuted_paddings_tensor);

    // Transform Transpose nodes into Identity nodes.
    const auto transpose_to_identity =
        [&cancelled_transposes,
         &mutation](utils::MutableNodeView* transpose) -> void {
      mutation->UpdateNodeOp(transpose, "Identity");
      mutation->RemoveNodeAttr(transpose, "Tperm");
      mutation->RemoveRegularFanin(transpose, 1);
      cancelled_transposes.insert(transpose);
    };

    transpose_to_identity(transpose_before);
    absl::c_for_each(pad_fanout_transposes, transpose_to_identity);
  }

  return mutation->Apply();
}

Status EraseOutputShapeAttrs(TransposeContext* context) {
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    auto* node = graph_view->GetNode(i);
    if (IsArg(*node->node())) {
      continue;
    }
    mutation->RemoveNodeAttr(node, kAttrOutputShape);
    TF_RETURN_IF_ERROR(mutation->Apply());
  }
  return absl::OkStatus();
}

}  // namespace

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status GenericLayoutOptimizer::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
  if (cluster == nullptr) {
    LOG(WARNING)
        << "generic layout optimizer was called with cluster == nullptr";
    return errors::Aborted("cluster == nullptr.");
  }
  if (!enforced_layout_.empty() && enforced_layout_ != "NHWC" &&
      enforced_layout_ != "NCHW") {
    return Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Invalid value for enforced_layout: ", enforced_layout_,
                     ". Supported layouts: 'NHWC', 'NCHW'."));
  }
  const auto gpu_stats = GetNumGPUs(*cluster);

  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;

  TransposeContext context;
  context.enforced_layout = enforced_layout_;

  if (gpu_stats.num_gpus > 0) {
    TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
        /*assume_valid_feeds=*/is_aggressive, item, cluster, &context));

    const auto src_dst_formats = GetSrcAndDstDataFormats(context, gpu_stats);
    context.AssignDeviceAndDataFormats(kGPU, src_dst_formats.first,
                                       src_dst_formats.second);
  } else {
    TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
        /*assume_valid_feeds=*/is_aggressive, item, cluster, &context));
    switch (cpu_layout_conversion_) {
      case RewriterConfig::NCHW_TO_NHWC:
        context.AssignDeviceAndDataFormats(kCPU, kNCHW, kNHWC);
        break;
      // TODO(intel-tf): Add functionality for NHWC_TO_NCHW layout conversion on
      // CPU.
      case RewriterConfig::NHWC_TO_NCHW:
        return errors::Aborted(
            "Conversion from NHWC to NCHW is currently not  available for "
            "CPU.");
      default:
        *output = item.graph;
        VLOG(2) << "No layout conversion will take place for CPU.";
        return absl::OkStatus();
    }
  }

  TransposerFactory transposer_factory;
  TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
  if (context.graph.node_size() > context.num_nodes || is_aggressive) {
    TF_RETURN_IF_ERROR(ExpandLayoutAgnosticOp(&context, &transposer_factory));
    TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
    TF_RETURN_IF_ERROR(EraseCancellableNodesAroundPad(&context));
    // TODO(lyandy): Remove sorting once other optimizers are migrated to using
    // `utils::GraphView`.
    TF_RETURN_IF_ERROR(
        context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));
  }
  TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

  *output = context.graph;
  return absl::OkStatus();
}

}  // end namespace grappler
}  // end namespace tensorflow
