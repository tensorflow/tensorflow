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

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

namespace {

inline int GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  for (const auto& device : devices) {
    if (device.second.type() == "GPU") {
      num_gpus++;
    }
  }
  return num_gpus;
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
            error::NOT_FOUND,
            absl::StrCat(
                "Layout sensitive operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
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
            error::NOT_FOUND,
            absl::StrCat(
                "Layout agnostic operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
}

inline bool IsCancellableConstPermTransposeNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  Tensor fanout_tensor;
  if (!GetValueAttrIfConstPermTransposeNode(fanout_transpose, &fanout_tensor)) {
    return false;
  }
  Tensor fanin_tensor;
  if (!GetValueAttrIfConstPermTransposeNode(fanin_transpose, &fanin_tensor)) {
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

Status EraseOutputShapeAttrs(TransposeContext* context) {
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    mutation->RemoveNodeAttr(graph_view->GetNode(i), kAttrOutputShape);
    TF_RETURN_IF_ERROR(mutation->Apply());
  }
  return Status::OK();
}

}  // namespace

Status GenericLayoutOptimizer::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
  if (cluster == nullptr) {
    LOG(WARNING)
        << "generic layout optimizer was called with cluster == nullptr";
    return errors::Aborted("cluster == nullptr.");
  }
  if (GetNumGPUs(*cluster) < 1) {
    return errors::Aborted(
        "No GPUs found: GenericLayoutOptimizer is currently only tuned for "
        "GPU.");
  }

  TransposeContext context;
  TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
      item, cluster, src_format_, dst_format_, target_device_, &context));
  TransposerFactory transposer_factory;
  TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
  TF_RETURN_IF_ERROR(ExpandLayoutAgnosticOp(&context, &transposer_factory));
  TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
  TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

  *output = context.graph;
  return Status::OK();
}

void GenericLayoutOptimizer::Feedback(Cluster* cluster,
                                      const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // Takes no feedback.
}

}  // end namespace grappler
}  // end namespace tensorflow
