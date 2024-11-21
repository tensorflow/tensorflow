/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

namespace {
// Internal encapsulation of state for the MLIR bridge graph analyzer. Steps
// through the nodes in the graph and reachable functions, tracking whether
// each feature of interest is found.
//
// Tracks the presence of each feature of interest in the corresponding streamz
// metric. Note that the graph traversal does not terminate early so as to
// capture all of these features.
class MlirBridgeGraphAnalyzer {
 public:
  explicit MlirBridgeGraphAnalyzer(bool single_core_inference_mode)
      : single_core_inference_mode_(single_core_inference_mode) {}
  ~MlirBridgeGraphAnalyzer() = default;
  // Not copyable or movable.
  MlirBridgeGraphAnalyzer(const MlirBridgeGraphAnalyzer&) = delete;
  MlirBridgeGraphAnalyzer& operator=(const MlirBridgeGraphAnalyzer&) = delete;

  // Analyzes whether the graph has features not guaranteed to be supported by
  // the MLIR-based TF XLA bridge.
  bool HasUnsupportedFeatures(const Graph& graph,
                              const FunctionLibraryDefinition* function_library,
                              std::optional<ConfigProto> config_proto,
                              tensorflow::TF2XLABridgeVersion bridge_version) {
    // Non-ok status is considered as "unsupported" since this means something
    // is wrong or unexpected with the graph itself.
    invalid_graph_ =
        invalid_graph_ || !AnalyzeGraphAndReachableFunctions(
                               graph, function_library, config_proto)
                               .ok();

    // We conservatively consider the graph to be unsupported if it's not
    // *known* to be TF2. That is, graphs that have kNotTracked construction
    // context are considered unsupported, even though they might in fact be
    // TF2 models.
    auto construction_context = graph.GetConstructionContextInternal();
    bool is_tf2 = construction_context == ConstructionContext::kEagerRuntime;
    auto is_tf2_execution_enabled = tensorflow::tf2_execution_enabled();
    auto has_unsupported_features = false;
    auto is_v1_compat = bridge_version == TF2XLABridgeVersion::kV1Compat;
    auto is_nominal_bridge = bridge_version == TF2XLABridgeVersion::kNominal;
    auto is_tfrt_bridge = bridge_version == TF2XLABridgeVersion::kTFRTNominal;
    is_eager_compliant_ = is_tf2_execution_enabled || is_tf2 ||
                          is_nominal_bridge || is_tfrt_bridge;

    is_eager_compliant_ |= (is_v1_compat && contains_partitioned_call_);

    has_unsupported_features = contains_ref_type_ || invalid_graph_;

    // For non single core inference mode, checking conditions:
    if (!single_core_inference_mode_) {
      has_unsupported_features |=
          !is_eager_compliant_ || uses_v1_control_flow_ ||
          HasTpuReplicatedCoreUnsupportedFeature(is_nominal_bridge,
                                                 is_v1_compat, is_tfrt_bridge);
    }

    PrintGraphUnsupportedFeatures(is_tf2, is_tf2_execution_enabled,
                                  is_v1_compat, is_tfrt_bridge,
                                  is_nominal_bridge, has_unsupported_features);

    // Determine whether or not the graph contains unsupported features.
    return has_unsupported_features;
  }

 private:
  static constexpr char kPartitionedCall[] = "TPUPartitionedCall";

  bool HasTPUReplicatedCoreAttr(const Node& node) {
    constexpr absl::string_view kTPUReplicatedCore = "TPU_REPLICATED_CORE";
    const std::string& device = node.requested_device();
    if (!device.empty()) {
      DeviceNameUtils::ParsedName name;
      if (DeviceNameUtils::ParseFullName(device, &name)) {
        // The TPU_REPLICATED_CORE attrs is not relevant for single TPU core
        // inference.
        // TODO(b/201091475): this can be generalized to check
        // num_cores_per_replica != 1, rather than being special cased for
        // single core inference.
        if (name.type == kTPUReplicatedCore && !single_core_inference_mode_) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasTpuReplicatedCoreUnsupportedFeature(bool is_nominal_bridge,
                                              bool is_v1_compat,
                                              bool is_tfrt_bridge) {
    if (!has_tpu_replicated_core_) {
      return false;
    }
    return has_infeed_dequeue_tuple_with_tpu_replicated_core_;
  }

  void PrintGraphUnsupportedFeatures(bool is_tf2, bool is_tf2_execution_enabled,
                                     bool is_v1_compat, bool is_tfrt_bridge,
                                     bool is_nominal_bridge,
                                     bool has_unsupported_features) {
    if (!has_unsupported_features) {
      VLOG(1) << "Graph doesn't have unsupported features";
      return;
    }

    LOG(INFO)
        << "Graph has unsupported features: " << (is_tf2 ? "" : "not is_tf2, ")
        << (is_tf2_execution_enabled ? "" : "not tf2_execution, ")
        << (is_nominal_bridge ? "" : "not nominal bridge, ")
        << (is_tfrt_bridge ? "" : "not tfrt bridge, ")
        << (is_v1_compat && contains_partitioned_call_
                ? "contains partitioned calls at v1 compat bridge call site, "
                : "")
        << (contains_ref_type_ ? "contains ref variables, " : "")
        << (invalid_graph_ ? "Invalid graph, " : "")
        << (uses_v1_control_flow_ ? "uses control flow v1 " : "")
        << ((has_tpu_replicated_core_ &&
             has_infeed_dequeue_tuple_with_tpu_replicated_core_)
                ? "InfeedDequeueTuple op with TPU_REPLICATED_CORE attr, "
                : "");
  }

  // Traverses each node in the graph and gathers information about each of the
  // features. Specifically, sets the relevant class variable to true when a
  // feature is found.
  void AnalyzeGraphNodes(const Graph& graph) {
    constexpr absl::string_view kIdentityOp = "Identity";
    constexpr absl::string_view kIdentityNOp = "IdentityN";
    constexpr absl::string_view kCastOp = "Cast";
    constexpr absl::string_view kInfeedDequeueTuple = "InfeedDequeueTuple";
    constexpr absl::string_view kOutsideCompilationAttr =
        "_xla_outside_compilation";
    constexpr absl::string_view kAllowSoftPlacementAttr =
        "allow_soft_placement";
    constexpr absl::string_view kManualControlDepsAttr =
        "_has_manual_control_dependencies";

    auto has_ref_type = [](const DataTypeVector& types) {
      for (const DataType& dtype : types)
        if (IsRefType(dtype)) return true;
      return false;
    };

    for (const Node* node : graph.nodes()) {
      contains_ref_type_ =
          (contains_ref_type_ || has_ref_type(node->input_types()) ||
           has_ref_type(node->output_types()));
      contains_partitioned_call_ = (contains_partitioned_call_ ||
                                    node->type_string() == kPartitionedCall);
      uses_v1_control_flow_ = (uses_v1_control_flow_ || node->IsControlFlow());
      uses_outside_compilation_ =
          (uses_outside_compilation_ ||
           node->attrs().Find(kOutsideCompilationAttr) != nullptr);
      has_manual_control_deps_ = (has_manual_control_deps_ ||
                                  node->attrs().Find(kManualControlDepsAttr));

      auto soft_placement_attr = node->attrs().Find(kAllowSoftPlacementAttr);
      if (soft_placement_attr != nullptr) {
        uses_outside_compilation_ =
            (uses_outside_compilation_ || soft_placement_attr->b());
      }

      // TODO(b/187611527): Add support for the ops with explicit device
      // assignment on the TPU_REPLICATED_CORE.
      if (node->type_string() == kIdentityOp ||
          node->type_string() == kCastOp ||
          node->type_string() == kIdentityNOp) {
        if (HasTPUReplicatedCoreAttr(*node)) {
          has_tpu_replicated_core_ = true;
          VLOG(2) << node->type_string()
                  << " node has TPU_REPLICATED_CORE attribute.";
        }
      }
      if (node->type_string() == kInfeedDequeueTuple &&
          HasTPUReplicatedCoreAttr(*node)) {
        has_infeed_dequeue_tuple_with_tpu_replicated_core_ = true;
      }
    }
  }

  // Analyze all functions from the flib_def if there are any that belong to
  // the inference graph.
  void AnalyzeInferenceGraphs(const FunctionLibraryDefinition& flib_def) {
    if (contains_partitioned_call_) return;

    for (const std::string& func_name : flib_def.ListFunctionNames()) {
      const FunctionDef* func_def = flib_def.Find(func_name);
      for (const NodeDef& node_def : func_def->node_def()) {
        contains_partitioned_call_ = node_def.op() == kPartitionedCall;
        if (contains_partitioned_call_) return;
      }
    }
  }

  // Checks any reachable functions from `graph_def` in `flib_def`
  // for unsupported features in the MLIR-based bridge.
  //
  // Returns failure in the event that the FunctionDef fails to convert to
  // FunctionBody. Otherwise returns success.
  absl::Status AnalyzeReachableFunctions(
      const GraphDef& graph_def, const FunctionLibraryDefinition& flib_def) {
    // Check the inputs and outputs of a function for reference variables.
    auto signature_contains_ref_type = [](const OpDef& signature) {
      for (const auto& args : {signature.input_arg(), signature.output_arg()}) {
        for (const auto& arg : args) {
          if (IsRefType(arg.type())) return true;
        }
      }
      return false;
    };

    for (const std::string& func_name :
         flib_def.ReachableDefinitions(graph_def).ListFunctionNames()) {
      const FunctionDef* func_def = flib_def.Find(func_name);
      if (func_def->has_signature()) {
        contains_ref_type_ = contains_ref_type_ ||
                             signature_contains_ref_type(func_def->signature());
      }
      // Check the function body.
      std::unique_ptr<FunctionBody> func_body;
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          *func_def, AttrSlice(&func_def->attr()), &flib_def, &func_body));
      AnalyzeGraphNodes(*func_body->graph);
    }
    return absl::OkStatus();
  }

  // Checks the inputted graph for any features which aren't supported in the
  // MLIR-based bridge, stepping through each node in the graph as well as any
  // reachable functions (inputs, outputs, and function body).
  //
  // Note that this analysis does not terminate early because we care about
  // collecting all of these metrics.
  //
  // Returns failure in the event that the FunctionDef fails to convert to
  // FunctionBody. Otherwise returns success.
  absl::Status AnalyzeGraphAndReachableFunctions(
      const Graph& graph, const FunctionLibraryDefinition* function_library,
      std::optional<ConfigProto> config_proto) {
    // First, check whether soft placement is enabled. This means that auto
    // outside compilation may be used.
    uses_outside_compilation_ =
        uses_outside_compilation_ ||
        (config_proto.has_value() && config_proto->allow_soft_placement());

    // Analyze each node in this graph.
    AnalyzeGraphNodes(graph);

    // Then check any associated functions in the graph
    // FunctionLibraryDefinition.
    GraphDef graph_def;
    graph.ToGraphDef(&graph_def);
    TF_RETURN_IF_ERROR(AnalyzeReachableFunctions(graph_def, graph.flib_def()));
    // Analyze whether there is an inference graph, including non reachable
    // from the `graph` itself. This happens when there is a sequence of
    // TPUPartitionedCall()->main()->PartitionedCall() and only second part
    // of the graph is processed by the MLIR bridge.
    AnalyzeInferenceGraphs(graph.flib_def());

    // Check any associated function in the graph defined in a separate
    // FunctionLibraryDefinition.
    if (function_library != nullptr) {
      TF_RETURN_IF_ERROR(
          AnalyzeReachableFunctions(graph_def, *function_library));
      AnalyzeInferenceGraphs(*function_library);
    }

    return absl::OkStatus();
  }

  bool contains_partitioned_call_ = false;
  bool contains_ref_type_ = false;
  bool invalid_graph_ = false;
  bool uses_outside_compilation_ = false;
  bool uses_v1_control_flow_ = false;
  bool has_manual_control_deps_ = false;
  bool single_core_inference_mode_ = false;
  bool is_eager_compliant_ = false;
  bool has_tpu_replicated_core_ = false;
  bool has_infeed_dequeue_tuple_with_tpu_replicated_core_ = false;
};

}  // namespace

bool GraphHasUnsupportedFeaturesInMlirBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library,
    std::optional<ConfigProto> config_proto, TF2XLABridgeVersion bridge_version,
    bool single_core_inference_mode) {
  return MlirBridgeGraphAnalyzer(single_core_inference_mode)
      .HasUnsupportedFeatures(graph, function_library, config_proto,
                              bridge_version);
}

}  // namespace tensorflow
