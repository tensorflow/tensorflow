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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor_util.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

auto* mlir_first_phase_unsupported_feature_count_v2 =
    tensorflow::monitoring::Counter<4>::New(
        /* metric name */
        "/tensorflow/core/mlir_first_phase_unsupported_feature_count_v2",
        /* metric description */
        "Tracks if the Graph contains any specified feature in the first "
        "phase of the MLIR bridge",
        /* metric field */ "feature",
        /* metric field */ "bridge_version",
        /* metric field*/ "is_inference_graph",
        /* metric field */ "has_unsupported_feature");

constexpr char kNonTfrtInferenceCallSite[] = "kNonTfrtInferenceCallSite";
constexpr char kTfrtInferenceCallSite[] = "kTfrtInferenceCallSite";
constexpr char kTpuReplicatedCore[] = "tpu_replicated_core";
constexpr char kTpuReplicatedCoreWithInference[] =
    "kTpuReplicatedCoreWithInference";
constexpr char kOutsideCompilation[] = "kOutsideCompilation";
constexpr char kReferenceVariable[] = "kReferenceVariable";
constexpr char kV1ControlFlow[] = "kV1ControlFlow";
constexpr char kUninitializedResourceArgs[] = "kUninitializedResourceArgs";
constexpr char kManualControlDepsNoTpuReplicatedCoreAttr[] =
    "kManualControlDepsNoTpuReplicatedCoreAttr";
constexpr char kManualControlDepsWithTpuReplicatedCoreAttr[] =
    "kManualControlDepsWithTpuReplicatedCoreAttr";
constexpr char kInvalidGraph[] = "kInvalidGraph";
constexpr char kNotAvailable[] = "N/A";
constexpr char kSupportedFeature[] = "kSupportedFeature";
constexpr char kUnsupportedFeature[] = "kUnsupportedFeature";
constexpr char kNotTf2[] = "kNotTf2";  // will either be session or trft
constexpr char kNotTf2WithInference[] = "kNotTf2WithInference";
constexpr char kNotTf2WithoutInference[] = "kNotTf2WithoutInference";
constexpr char kDynamicPadderOps[] = "kDynamicPadderOps";
constexpr char kReshapeOutsideCompilation[] = "kReshapeOutsideCompilation";
constexpr char kInfeedDequeueTupleWithTpuReplicateCoreAttr[] =
    "kInfeedDequeueTupleWithTpuReplicateCoreAttr";
constexpr char kInference[] = "kInference";
constexpr char kNotInference[] = "kNotInference";

namespace {

// Record a specified feature in metrics.
// Also, this method can be used to record if the gragh contains unsupported
// feature in the first phase.
static inline void RecordFeature(const char feature[],
                                 TF2XLABridgeVersion bridge_version,
                                 bool inference_graph,
                                 const char has_unsupported_feature[]) {
  std::string is_inference_graph = inference_graph ? kInference : kNotInference;
  mlir_first_phase_unsupported_feature_count_v2
      ->GetCell(feature, BridgeVersionToString(bridge_version),
                is_inference_graph, has_unsupported_feature)
      ->IncrementBy(1);
}

// Returns true if the graph has any op that is unsupported by the phase1 of the
// bridge.
bool IsUnsupportedOp(absl::string_view op_name, bool is_tf2) {
  static auto* exact_matched_ops_tf1 =
      new absl::flat_hash_set<absl::string_view>{};
  static auto* exact_matched_ops_tf2 =
      new absl::flat_hash_set<absl::string_view>{};

  if ((!is_tf2 && !exact_matched_ops_tf1->empty() &&
       exact_matched_ops_tf1->contains(op_name)) ||
      (is_tf2 && !exact_matched_ops_tf2->empty() &&
       exact_matched_ops_tf2->contains(op_name))) {
    VLOG(2) << op_name
            << " is not supported due to exact match to unsupported ops.";
    return true;
  }

  return false;
}
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
  ~MlirBridgeGraphAnalyzer() {}
  // Not copyable or movable.
  MlirBridgeGraphAnalyzer(const MlirBridgeGraphAnalyzer&) = delete;
  MlirBridgeGraphAnalyzer& operator=(const MlirBridgeGraphAnalyzer&) = delete;

  // Analyzes whether the graph has features not guaranteed to be supported by
  // the MLIR-based TF XLA bridge.
  bool HasUnsupportedFeatures(const Graph& graph,
                              const FunctionLibraryDefinition* function_library,
                              std::optional<ConfigProto> config_proto,
                              tensorflow::TF2XLABridgeVersion bridge_version,
                              bool record_stats) {
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
    bool use_session =
        construction_context == ConstructionContext::kDirectSession;

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

    RecordStatus(record_stats, is_tf2, is_tf2_execution_enabled, bridge_version,
                 is_v1_compat, is_tfrt_bridge, is_nominal_bridge,
                 has_unsupported_features, construction_context);

    UpdateGraphAnalysisPerOpInstrumentation(
        graph, is_tf2, use_session, is_tf2_execution_enabled, is_v1_compat,
        has_unsupported_features);
    // Determine whether or not the graph contains unsupported features.
    return has_unsupported_features;
  }

  void AssessForTF1Features(const Graph& graph, const std::string& device,
                            const std::string& context) {
    AnalyzeGraphNodes(graph);

    metrics::RecordTFVersionByGraphFeatures(
        device, context, uses_v1_control_flow_, contains_ref_type_,
        has_manual_control_deps_);
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

  std::string GetCallsiteStackTrace(std::string& stack_info) {
    std::vector<std::string> vec_of_stack = absl::StrSplit(stack_info, '\n');
    std::vector<std::string> callsite_stack_trace;
    for (int i = vec_of_stack.size() - 1; i > 0; --i) {
      const std::string& phase_one_stack_info = vec_of_stack[i];
      auto first_idx = phase_one_stack_info.find("tensorflow");
      auto last_idx = phase_one_stack_info.find_last_of('(');
      if (first_idx == std::string::npos) continue;
      if (last_idx == std::string::npos)
        last_idx = phase_one_stack_info.size() - 1;
      callsite_stack_trace.push_back(
          phase_one_stack_info.substr(first_idx, last_idx - first_idx));
    }
    return absl::StrJoin(callsite_stack_trace, " --> ");
  }

  bool HasTpuReplicatedCoreUnsupportedFeature(bool is_nominal_bridge,
                                              bool is_v1_compat,
                                              bool is_tfrt_bridge) {
    if (!has_tpu_replicated_core_) {
      return false;
    }
    return has_infeed_dequeue_tuple_with_tpu_replicated_core_;
  }

  void RecordStatus(bool record_stats, bool is_tf2,
                    bool is_tf2_execution_enabled,
                    tensorflow::TF2XLABridgeVersion bridge_version,
                    bool is_v1_compat, bool is_tfrt_bridge,
                    bool is_nominal_bridge, bool has_unsupported_features,
                    ConstructionContext construction_context) {
    if (!record_stats) return;
    // Valid features we are logging for tracking
    // Outside compilation is supported but tracking the number of graphs with
    // outside compilation is still a useful metric to have.
    if (uses_outside_compilation_)
      RecordFeature(kOutsideCompilation, bridge_version,
                    contains_partitioned_call_, kSupportedFeature);
    if (has_manual_control_deps_) {
      if (has_tpu_replicated_core_)
        RecordFeature(kManualControlDepsWithTpuReplicatedCoreAttr,
                      bridge_version, contains_partitioned_call_,
                      kSupportedFeature);
      else
        RecordFeature(kManualControlDepsNoTpuReplicatedCoreAttr, bridge_version,
                      contains_partitioned_call_, kSupportedFeature);
    }

    if (!has_unsupported_features) {
      if (contains_partitioned_call_) {
        RecordFeature(kInference, bridge_version, contains_partitioned_call_,
                      kSupportedFeature);
      } else {
        RecordFeature(kNotInference, bridge_version, contains_partitioned_call_,
                      kSupportedFeature);
      }
    }

    // Invalid Features
    if (contains_ref_type_)
      RecordFeature(kReferenceVariable, bridge_version,
                    contains_partitioned_call_, kUnsupportedFeature);
    if (!is_eager_compliant_) {
      if (contains_partitioned_call_) {
        RecordFeature(
            kNotTf2WithInference, bridge_version, contains_partitioned_call_,
            has_unsupported_features ? kUnsupportedFeature : kSupportedFeature);
      } else {
        RecordFeature(
            kNotTf2WithoutInference, bridge_version, contains_partitioned_call_,
            has_unsupported_features ? kUnsupportedFeature : kSupportedFeature);
      }
    }

    if (uses_v1_control_flow_)
      RecordFeature(kV1ControlFlow, bridge_version, contains_partitioned_call_,
                    kUnsupportedFeature);
    if (has_tpu_replicated_core_) {
      if (has_infeed_dequeue_tuple_with_tpu_replicated_core_)
        RecordFeature(kInfeedDequeueTupleWithTpuReplicateCoreAttr,
                      bridge_version, contains_partitioned_call_,
                      kUnsupportedFeature);
    }

    if (contains_partitioned_call_) {
      std::string stack_info = tensorflow::CurrentStackTrace();
      std::string callsite_stack_trace = GetCallsiteStackTrace(stack_info);
    }

    if (invalid_graph_)
      RecordFeature(kInvalidGraph, bridge_version, contains_partitioned_call_,
                    kUnsupportedFeature);

    RecordFeature(kNotAvailable, bridge_version, contains_partitioned_call_,
                  has_unsupported_features ? "YES" : "NO");
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
        << (is_tfrt_bridge && contains_partitioned_call_
                ? "contains partitioned calls at tfrt bridge call site, "
                : "")
        << (is_nominal_bridge && contains_partitioned_call_
                ? "contains partitioned calls at non-tfrt bridge call site, "
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
    auto construction_context = graph.GetConstructionContextInternal();
    bool is_tf2 = construction_context == ConstructionContext::kEagerRuntime;

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
      bool unsupported_op = IsUnsupportedOp(node->type_string(), is_tf2);
      if (unsupported_op) unsupported_ops_name_.insert(node->type_string());

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

  // Checks any reacheable functions from `graph_def` in `flib_def`
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

  void CollectTPUReplicateMetaDataOpInfo(
      const Node* node, std::string& num_replicas_str,
      std::string& num_cores_per_relica_str, std::string& use_tpu_str,
      std::string& allow_soft_placement_str,
      std::string& use_spmd_for_xla_partitioning_str) {
    int num_replicas;
    int num_cores_per_replica;
    bool use_tpu;
    bool allow_soft_placement;
    bool use_spmd_for_xla_partitioning;
    if (GetNodeAttr(node->attrs(), "num_replicas", &num_replicas).ok()) {
      num_replicas_str = std::to_string(num_replicas);
    }
    if (GetNodeAttr(node->attrs(), "num_cores_per_replica",
                    &num_cores_per_replica)
            .ok()) {
      num_cores_per_relica_str = std::to_string(num_cores_per_replica);
    }
    if (GetNodeAttr(node->attrs(), "use_tpu", &use_tpu).ok()) {
      use_tpu_str = use_tpu ? "True" : "False";
    }
    if (GetNodeAttr(node->attrs(), "allow_soft_placement",
                    &allow_soft_placement)
            .ok()) {
      allow_soft_placement_str = allow_soft_placement ? "True" : "False";
    }
    if (GetNodeAttr(node->attrs(), "use_spmd_for_xla_partitioning",
                    &use_spmd_for_xla_partitioning)
            .ok()) {
      use_spmd_for_xla_partitioning_str =
          use_spmd_for_xla_partitioning ? "True" : "False";
    }
  }
  void UpdateGraphAnalysisPerOpInstrumentation(const Graph& graph, bool is_tf2,
                                               bool use_session,
                                               bool is_tf2_execution_enabled,
                                               bool is_v1_compat,
                                               bool has_unsupported_features) {
    auto ops_in_instrumentation = absl::flat_hash_set<std::string>{
        "ResourceScatterUpdate", "TPUPartitionedOutput", "TPUPartitionedInput",
        "TPUPartitionedCall", "TPUReplicateMetadata"};
    auto ops_with_tpu_replicated_core_attr_set =
        absl::flat_hash_set<std::string>{"Cast", "Identity", "IdentityN"};

    auto visited_ops = absl::flat_hash_set<std::string>{};

    // compute the reasons once because they are at the graph not op level
    std::vector<std::string> reasons;
    if (!is_tf2 && !is_tf2_execution_enabled) {
      reasons.push_back("not eager mode and not tf2_execution");
    }
    if (contains_ref_type_) {
      reasons.push_back("contains ref_type");
    }
    if (invalid_graph_) {
      reasons.push_back("invalid graph");
    }
    if (uses_v1_control_flow_) {
      reasons.push_back("uses v1 control flow");
    }
    string construction_context = "Not tracked";
    if (is_tf2) {
      construction_context = "eager";
    } else if (use_session) {
      construction_context = "session";
    }
    if (!is_v1_compat && contains_partitioned_call_) {
      reasons.push_back("inference graph uses non session api");
    }
    if (!unsupported_ops_name_.empty()) {
      for (const std::string& s : unsupported_ops_name_) {
        reasons.push_back("unsupported op/attribute: " + s);
      }
    }
    if (reasons.empty()) {
      reasons.push_back("not available");
    }

    for (const Node* node : graph.nodes()) {
      std::string op_name = node->type_string();
      if ((ops_in_instrumentation.contains(op_name) ||
           (ops_with_tpu_replicated_core_attr_set.contains(op_name) &&
            HasTPUReplicatedCoreAttr(*node))) &&
          !visited_ops.contains(op_name)) {
        std::string num_replicas_str = "N/A";
        std::string num_cores_per_relica_str = "N/A";
        std::string use_tpu_str = "N/A";
        std::string allow_soft_placement_str = "N/A";
        std::string use_spmd_for_xla_partitioning_str = "N/A";
        visited_ops.insert(op_name);
        if (op_name == "TPUReplicateMetadata") {
          CollectTPUReplicateMetaDataOpInfo(
              node, num_replicas_str, num_cores_per_relica_str, use_tpu_str,
              allow_soft_placement_str, use_spmd_for_xla_partitioning_str);
        }
        for (const std::string& r : reasons) {
          // print a line out for each reason
          metrics::UpdateTfMlirBridgeGraphAnalysisPerOp(
              op_name, construction_context, single_core_inference_mode_,
              num_replicas_str, num_cores_per_relica_str, use_tpu_str,
              allow_soft_placement_str, use_spmd_for_xla_partitioning_str, r,
              has_unsupported_features);
        }
      }
    }
  }

  bool contains_partitioned_call_ = false;
  bool contains_ref_type_ = false;
  bool invalid_graph_ = false;
  bool uses_outside_compilation_ = false;
  bool uses_v1_control_flow_ = false;
  std::set<std::string> unsupported_ops_name_;
  bool has_manual_control_deps_ = false;
  bool single_core_inference_mode_ = false;
  bool is_eager_compliant_ = false;
  bool has_tpu_replicated_core_ = false;
  bool has_infeed_dequeue_tuple_with_tpu_replicated_core_ = false;
};

}  // namespace

std::string BridgeVersionToString(TF2XLABridgeVersion bridge_version) {
  switch (bridge_version) {
    case TF2XLABridgeVersion::kV1Compat:
      return "v1Compat";
    case TF2XLABridgeVersion::kNominal:
      return "nominal";
    case TF2XLABridgeVersion::kTFRTNominal:
      return "tfrtNominal";
    case TF2XLABridgeVersion::kNotBridgeUseCase:
      return "notBridgeUseCase";
  }

  // C++'s enum's are unsafe even when using an "exhaustive switch" like the one
  // above. Per go/totw/147, bridge_version may take on values not listed in the
  // enum definition even though the compiler doesn't think so. This code should
  // ensure that execution does not fall through the end of this function w/o
  // returning, therefor preventing undefined behavior.
  static constexpr char kUnknownBridgeVersionString[] =
      "unknown_bridge_version";
  VLOG(2) << "Unknown BridgeVersion: " << static_cast<int>(bridge_version);
  return kUnknownBridgeVersionString;
}

bool GraphHasFeaturesUnsupportedByMlirBridge(
    const Graph& graph, const FunctionLibraryDefinition* function_library,
    std::optional<ConfigProto> config_proto, TF2XLABridgeVersion bridge_version,
    bool record_stats, bool single_core_inference_mode) {
  return MlirBridgeGraphAnalyzer(single_core_inference_mode)
      .HasUnsupportedFeatures(graph, function_library, config_proto,
                              bridge_version, record_stats);
}

}  // namespace tensorflow
