/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/preperation_pass.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Graph optimizer context extension specific to TRTConditioner optimization
// pass.
struct TRTPreperationContext {
  explicit TRTPreperationContext(
      grappler::SetVector<NodeDef*>* nodes_to_simplify)
      : nodes_to_simplify(nodes_to_simplify) {}

  auto& NodesToSimplify() { return *nodes_to_simplify; }
  const auto& NodesToSimplify() const { return *nodes_to_simplify; }

  grappler::SetVector<NodeDef*>* nodes_to_simplify;
};

namespace {

// Extracts values from a Const op to `values`. Returns true if succeeds.
template <typename T>
bool TensorFromConstNode(const NodeDef& node, Tensor* tensor) {
  if (node.op() != "Const") {
    return false;
  }
  if (node.attr().count("dtype") == 0 || node.attr().count("value") == 0) {
    return false;
  }
  if (node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
    LOG(ERROR) << "  node attr dtype is " << node.attr().at("dtype").type()
               << " requested dtype is " << DataTypeToEnum<T>::value;
    return false;
  }
  return tensor->FromProto(node.attr().at("value").tensor());
}

// Base class for conditioning rewrites.
class TRTOptimizerRewriteStage : public grappler::GraphOptimizerStage<string> {
 public:
  explicit TRTOptimizerRewriteStage(const string& name,
                                    const grappler::GraphOptimizerContext& ctx,
                                    const TRTPreperationContext ctx_ext)
      : GraphOptimizerStage("TRTOptimizer", name, ctx), ctx_ext_(ctx_ext) {}
  ~TRTOptimizerRewriteStage() override = default;

 protected:
  // Graph rewrites can create new nodes that are inputs
  // to final simplified node. The created nodes can be also added to the
  // optimizer queue for further optimization.
  void AddToOptimizationQueue(NodeDef* node) const {
    ctx_ext_.nodes_to_simplify->PushBack(node);
  }

  bool IsInPreserveSet(const NodeDef& node) const {
    return ctx().nodes_to_preserve->find(node.name()) !=
           ctx().nodes_to_preserve->end();
  }

 private:
  // Extended context required for TRTOptimizer.
  const TRTPreperationContext ctx_ext_;
};

// Rewrites "x -> cast(SrcT, FP32)-> y" where SrcT != fp16 to
// "x -> cast(srcT, fp16) -> cast(fp16, fp32) -> y". The fp16 to fp32 cast
// is specifically recognized by the TF-TRT cast op converter in FP16 mode so
// that an engine input will be created between the two cast operations. The
// second cast is then implicitly added based on the TRT network's input
// datatype. This allows for avoiding unnecessary casts when converting models
// with int8 inputs in fp16 mode.
// IMPORTANT: This stage must be ordered after any other stage which creates or
// moves cast operations.
class RewriteFP32CastsStage : public TRTOptimizerRewriteStage {
 public:
  explicit RewriteFP32CastsStage(const grappler::GraphOptimizerContext& ctx,
                                 const TRTPreperationContext& ctx_ext)
      : TRTOptimizerRewriteStage("RewriteFP32CastsStage", ctx, ctx_ext) {}
  ~RewriteFP32CastsStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return (grappler::IsCastLike(*node) && !IsInPreserveSet(*node));
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* producer{nullptr};
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &producer));
    const OpDef* cast_def{nullptr};
    TF_RETURN_IF_ERROR(
        OpRegistry::Global()->LookUpOpDef(node->op(), &cast_def));

    // We only rewrite casts to FP16 from a non-FP16 type.
    DataType cast_src_type;
    TF_RETURN_IF_ERROR(InputTypeForNode(*node, *cast_def, 0, &cast_src_type));
    DataType cast_dst_type;
    TF_RETURN_IF_ERROR(OutputTypeForNode(*node, *cast_def, 0, &cast_dst_type));
    if (cast_dst_type != DataType::DT_FLOAT ||
        cast_src_type == DataType::DT_HALF) {
      return Status::OK();
    }

    VLOG(1) << "Rewriting cast to FP32 " << node->DebugString();
    (*node->mutable_attr())["DstT"].set_type(DataType::DT_HALF);

    // Create the SrcT -> FP16 cast operation and wire it together.
    NodeDef* cast_to_fp16 = AddCopyNode(
        OptimizedNodeName(grappler::ParseNodeScopeAndName(node->name()),
                          DataTypeString(DataType::DT_HALF)),
        node);
    (*cast_to_fp16->mutable_attr())["SrcT"].set_type(cast_src_type);
    (*cast_to_fp16->mutable_attr())["DstT"].set_type(DataType::DT_HALF);
    cast_to_fp16->set_input(0, node->input(0));
    ctx().node_map->AddOutput(producer->name(), cast_to_fp16->name());
    ctx().node_map->AddOutput(cast_to_fp16->name(), node->name());
    ctx().node_map->RemoveOutput(producer->name(), node->name());

    // Update the consumer cast types and input.
    (*node->mutable_attr())["SrcT"].set_type(DataType::DT_HALF);
    (*node->mutable_attr())["DstT"].set_type(DataType::DT_FLOAT);
    node->set_input(0, cast_to_fp16->name());
    *simplified_node_name = node->name();
    return Status::OK();
  }
};
}  // namespace

Status TRTPreperationPass::RunPipeline(
    grappler::GraphOptimizerStagePipeline<string>& pipeline,
    const grappler::GraphOptimizerContext& context,
    const TRTPreperationContext& trt_context) {
  auto& nodes_to_simplify = *trt_context.nodes_to_simplify;
  GraphDef* optimized_graph = context.optimized_graph;

  // Reset the nodes to simplify.
  nodes_to_simplify.Reserve(optimized_graph->node_size());
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph->mutable_node(i));
  }

  VLOG(1) << "Run " << pipeline.NumStages()
          << " TRT optimizer conditioner stages: "
          << absl::StrJoin(pipeline.StageNames(), ",");

  // Run the optimization.
  while (!nodes_to_simplify.Empty()) {
    NodeDef* node = nodes_to_simplify.PopBack();
    string simplified_tensor;
    bool optimized = pipeline.PassThroughAllStages(node, &simplified_tensor);

    // If the node was not optimized by any of the stages, go to the next one.
    if (!optimized) continue;

    // re-wire consumers of an old node to the new one
    if (grappler::NodeName(simplified_tensor) != node->name()) {
      // Always consider simplified_tensor for further optimizations.
      NodeDef* simplified_node = node_map_->GetNode(simplified_tensor);
      if (simplified_node != nullptr)
        nodes_to_simplify.PushBack(simplified_node);

      // When `node` is simplified to another node rather than in-place, the
      // consumers of `node` are already redirected to `simplified_tensor`.
      // Re-push the consumers into `nodes_to_simplify` for further
      // optimizations.
      const std::vector<NodeDef*> consumers =
          node_map_->GetOutputsOrderedByNodeName(node->name());
      for (NodeDef* consumer : consumers) {
        // Update `consumer`'s use of `node` to `input`'s operand.
        for (int i = 0; i < consumer->input_size(); ++i) {
          int operand_pos = 0;
          string operand_node_name =
              grappler::ParseNodeName(consumer->input(i), &operand_pos);
          if (operand_node_name == node->name()) {
            *consumer->mutable_input(i) =
                (operand_pos < 0 ? grappler::AsControlDependency(
                                       grappler::NodeName(simplified_tensor))
                                 : simplified_tensor);
          }
        }
        node_map_->UpdateInput(consumer->name(), node->name(),
                               simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
  }
  return Status::OK();
}

Status TRTPreperationPass::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  if (config == nullptr) {
    return Status::OK();
  }
  const auto params = config->parameter_map();
  if (params.count("precision_mode")) {
    TF_RETURN_IF_ERROR(TrtPrecisionModeFromName(
        absl::AsciiStrToUpper(params.at("precision_mode").s()),
        &precision_mode_));
  }
  return Status::OK();
}

Status TRTPreperationPass::ConditionGraphConversion(bool can_use_shapes) {
  grappler::SetVector<NodeDef*> nodes_to_simplify;

  // Stop pipeline after first stage returning non-empty simplified tensor
  // name.
  const auto stop = [](const string& result) { return !result.empty(); };

  // Run the main pipeline.
  const grappler::GraphOptimizerContext ctx(
      &nodes_to_preserve_, optimized_graph_, graph_properties_.get(),
      node_map_.get(), &feed_nodes_, opt_level_);
  const TRTPreperationContext ctx_ext(&nodes_to_simplify);
  grappler::GraphOptimizerStagePipeline<string> pipeline(stop);
  TF_RETURN_IF_ERROR(RunPipeline(pipeline, ctx, ctx_ext));

  // Perform cast rewrites at the end.
  if (precision_mode_ == TrtPrecisionMode::FP16) {
    const grappler::GraphOptimizerContext ctx(
        &nodes_to_preserve_, optimized_graph_, graph_properties_.get(),
        node_map_.get(), &feed_nodes_, opt_level_);
    const TRTPreperationContext ctx_ext(&nodes_to_simplify);
    grappler::GraphOptimizerStagePipeline<string> pipeline(stop);
    pipeline.AddStage<RewriteFP32CastsStage>(ctx, ctx_ext);
    TF_RETURN_IF_ERROR(RunPipeline(pipeline, ctx, ctx_ext));
  }

  return Status::OK();
}

Status TRTPreperationPass::Optimize(grappler::Cluster* cluster,
                                     const grappler::GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  // Set up helper data structures.
  grappler::GrapplerItem optimized_item(item);
  optimized_graph_ = &optimized_item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  node_map_ = std::make_unique<grappler::NodeMap>(optimized_graph_);
  for (const auto& feed : item.feed) {
    feed_nodes_.insert(grappler::NodeName(feed.first));
  }

  // Perform topological sort on the graph in order to help DedupComputations
  // and AddOpsRewrite to optimize larger subgraphs starting from the roots
  // with more inputs.
  TF_RETURN_IF_ERROR(grappler::TopologicalSort(optimized_graph_));
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  // Setup the graph properties handle.
  graph_properties_ =
      std::make_unique<grappler::GraphProperties>(optimized_item);
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  const Status status =
      graph_properties_->InferStatically(assume_valid_feeds,
                                         /*aggressive_shape_inference=*/false,
                                         /*include_tensor_values=*/false);
  const bool can_use_shapes = status.ok();
  if (!can_use_shapes) {
    VLOG(1) << "Shape inference failed." << status.error_message();
  }

  // Perform conditioning optimization.
  TF_RETURN_IF_ERROR(ConditionGraphConversion(can_use_shapes));
  *optimized_graph = *optimized_graph_;
  return Status::OK();
}

static grappler::CustomGraphOptimizerRegistrar TRTPreperationPass_Registrar(
    []() { return new TRTPreperationPass("TensorRTConditioner"); },
    ("TensorRTConditioner"));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT