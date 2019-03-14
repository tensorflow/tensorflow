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

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

#include <atomic>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {
// The clusters we create here are eventually lowered into an
// _XlaCompile/_XlaRun pair with a TF executor "fallback" that uses the
// PartitionedCall op to execute the cluster in the regular graph executor if
// need be.  PartitionedCall, however, reruns the entire TF graph optimization
// pipeline over the cluster which includes this mark for compilation pass.  To
// avoid endlessly recursing we tag nodes that we've already visited with this
// attribute so that we can bail out if we see them a second time.
//
// TODO(sanjoy): This method is not robust since it is possible that the
// optimizations run by PartitionedCall can mutate the cluster arbitrarily,
// dropping the kXlaAlreadyClustered attributes from all nodes in the process.
// The correct fix is to use the ConfigProto to pass in some sort of flag into
// the PartitionedCall kernel that tells it to not rerun auto-clustering on the
// cluster.
const char* kXlaAlreadyClustered = "_XlaAlreadyClustered";

// Aggregates information about what kinds of ops are allowed.
struct OperationFilter {
  // Whether resource variable ops are allowed.  We do not allow resource
  // variable ops in called functions (either as direct TF calls or as higher
  // order control flow ops) because we do not yet model their memory effects in
  // jit/resource_variable_safety_analysis.
  bool allow_resource_ops;

  // Whether stateful RNG ops are allowed.  XLA's RNG does not have the same
  // seeding behavior as TensorFlow's RNG (b/34749654).  So we avoid
  // auto-clustering stateful RNG ops.
  bool allow_stateful_rng_ops;

  // TODO(b/118970344): Whether ControlTrigger ops are allowed.  It is unsound
  // to cluster ControlTrigger because of how we use deadness analysis.
  bool allow_control_trigger;

  // Whether ops with dummy implementations are allowed. We avoid
  // auto-clustering these ops so that the user is not surprised when XLA is
  // implicitly enabled. If the user explicitly specifies to use XLA, it is fine
  // to resort to a dummy implementation. Currently Assert and CheckNumerics ops
  // have dummy XLA implementations.
  bool allow_dummy_ops;

  // Whether ops that produce or consume DT_VARIANT values are allowed.  We
  // don't auto-cluster these ops because we don't yet support live-in or
  // live-out DT_VARIANT values.
  bool allow_ops_producing_or_consuming_variant;
};

bool IsDummyImplOp(absl::string_view op_name) {
  return op_name == "Assert" || op_name == "CheckNumerics";
}

bool IsStatefulRandomOp(absl::string_view op_name) {
  return op_name == "RandomUniform" || op_name == "RandomShuffle" ||
         op_name == "RandomUniformInt" || op_name == "RandomStandardNormal" ||
         op_name == "TruncatedNormal" || op_name == "Multinomial";
}

bool OpProducesOrConsumesVariant(const Node& node) {
  auto is_variant = [](DataType dtype) { return dtype == DT_VARIANT; };
  return absl::c_any_of(node.input_types(), is_variant) ||
         absl::c_any_of(node.output_types(), is_variant);
}

bool HasXLAKernel(const Node& node, const DeviceType& jit_device_type) {
  // There is a SymbolicGradient kernel on the XLA_JIT device, but the gradient
  // is really a kind of function call and will be handled by
  // IsCompilableCall().
  if (node.type_string() == "SymbolicGradient") return false;
  if (node.type_string() == "Const") {
    // Skip Const op with type DT_STRING, since XLA doesn't support it, but the
    // registered Const KernelDef says that it does, to support no-op Assert for
    // tfcompile.
    const AttrValue* attr = node.attrs().Find("dtype");
    if (attr != nullptr && attr->type() == DT_STRING) {
      return false;
    }
  }

  // XLA does not offer guaranteed aliasing between the input and output of the
  // XLA cluster so it can't implement the forward-tensor-ref semantic.  Leave
  // such nodes out of XLA clusters.
  if (HasForwardedRefInput(node)) {
    VLOG(2) << "Rejecting " << node.name() << ": Identity with unsafe cast.";
    return false;
  }

  return FindKernelDef(jit_device_type, node.def(), nullptr, nullptr).ok();
}

bool HasResourceOutput(const Node& node) {
  return std::find(node.output_types().begin(), node.output_types().end(),
                   DT_RESOURCE) != node.output_types().end();
}

bool HasResourceInput(const Node& node) {
  return std::find(node.input_types().begin(), node.input_types().end(),
                   DT_RESOURCE) != node.input_types().end();
}

// Returns true if `node` is a resource operation recognized by tf2xla that
// operates on something other than resource variables.
bool IsNonResourceVarResourceOp(const Node& node) {
  // TODO(b/112837194): We can't cluster these because we only support
  // snapshotting resource variables (and we can't e.g. snapshot stacks).  This
  // limitation may be fixable with some work.
  const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(node.type_string());
  return op_info && op_info->resource_kind() != XlaResourceKind::kVariable;
}

// Make sure we don't recurse infinitely on recursive functions.
const int kMaxRecursionDepth = 10;

bool IsCompilableCall(const NodeDef& call_def,
                      const DeviceType& jit_device_type,
                      const OperationFilter& op_filter, int depth,
                      FunctionLibraryRuntime* lib_runtime);

// Tests whether 'while_node' is a completely compilable loop.
// Every operator in the condition and body functions must be compilable for a
// while loop to be compilable.
bool IsCompilableWhile(const Node& while_node,
                       const DeviceType& jit_device_type,
                       const OperationFilter& op_filter, int depth,
                       FunctionLibraryRuntime* lib_runtime) {
  const NameAttrList* name_attr;
  NodeDef call;
  Status status;
  status = GetNodeAttr(while_node.attrs(), "cond", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": missing 'cond' attribute on While node.";
    return false;
  }
  const string cond_func = name_attr->name();
  call.set_name("while_cond");
  call.set_op(cond_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, jit_device_type, op_filter, depth + 1,
                        lib_runtime)) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": can't compile loop condition: " << cond_func;
    return false;
  }
  status = GetNodeAttr(while_node.attrs(), "body", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": missing 'body' attribute on While node.";
    return false;
  }
  const string body_func = name_attr->name();
  call.set_name("while_body");
  call.set_op(body_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, jit_device_type, op_filter, depth + 1,
                        lib_runtime)) {
    VLOG(2) << "Rejecting While " << while_node.name()
            << ": can't compile loop body: " << body_func;
    return false;
  }
  return true;
}

// Tests whether 'call_def' is a call to a completely compilable function.
// Every operator in the function must be compilable for a function to be
// compilable.
bool IsCompilableCall(const NodeDef& call_def,
                      const DeviceType& jit_device_type,
                      const OperationFilter& op_filter, int depth,
                      FunctionLibraryRuntime* lib_runtime) {
  if (depth > kMaxRecursionDepth) {
    VLOG(2) << "Rejecting " << call_def.op()
            << ": function depth limit exceeded.";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status status = InstantiateFunctionCall(call_def, *lib_runtime, &handle);
  if (!status.ok()) {
    VLOG(2) << "Rejecting " << call_def.DebugString()
            << ": could not instantiate: " << status;
    return false;
  }

  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });

  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  CHECK(fbody);
  const FunctionDef& fdef = fbody->fdef;
  bool noinline = false;
  if (GetNodeAttr(AttrSlice(&fdef.attr()), "_noinline", &noinline).ok() &&
      noinline) {
    // The underlying mechanism that calls non-inlined functions uses
    // LocalExecutor, which interacts poorly with the LocalExecutor used by
    // tf2xla to translate the TF graph into XLA.  So we avoid this for now.
    //
    // TODO(b/36139787): Create a mechanism to set inlining hints.
    VLOG(2) << "Rejecting " << call_def.op()
            << ": can't compile noinline function.";
    return false;
  }

  for (Node* node : fbody->graph->op_nodes()) {
    if (node->type_string() == "_Arg" || node->type_string() == "_Retval")
      continue;
    if (node->type_string() == "While") {
      // Handle functional While loop.
      return IsCompilableWhile(*node, jit_device_type, op_filter, depth + 1,
                               lib_runtime);
    }
    if (!op_filter.allow_resource_ops &&
        (HasResourceInput(*node) || HasResourceOutput(*node))) {
      return false;
    }
    if (!op_filter.allow_stateful_rng_ops &&
        IsStatefulRandomOp(node->type_string())) {
      return false;
    }
    if (!op_filter.allow_control_trigger && node->IsControlTrigger()) {
      return false;
    }
    if (!op_filter.allow_dummy_ops && IsDummyImplOp(node->type_string())) {
      return false;
    }
    if (!op_filter.allow_ops_producing_or_consuming_variant &&
        OpProducesOrConsumesVariant(*node)) {
      return false;
    }
    if (!HasXLAKernel(*node, jit_device_type) &&
        !IsCompilableCall(node->def(), jit_device_type, op_filter, depth + 1,
                          lib_runtime)) {
      VLOG(2) << "Rejecting " << call_def.op() << ": unsupported op "
              << node->name() << ": " << node->def().ShortDebugString();
      return false;
    }
  }
  return true;
}

// Returns true if the op can be decomposed into XLA ops for which
// there are fusable elemental implementations.
//
// TODO(hpucha): Remove this code since this functionality is subsumed by
// Grappler XlaFusionOptimizer.
bool IsXlaFusable(const NodeDef& node) {
  static const std::unordered_set<std::string>* elementwise_ops =
      new std::unordered_set<std::string>(
          {// tf2xla/kernels/aggregate_ops.cc
           "AddN",
           // tf2xla/kernels/batchtospace_op.cc
           "BatchToSpace", "BatchToSpaceND",
           // tf2xla/kernels/bcast_ops.cc
           "BroadcastArgs", "BroadcastGradientArgs",
           // tf2xla/kernels/bias_ops.cc
           "BiasAdd", "BiasAddV1", "BiasAddGrad" /*(Reduce)*/,
           // tf2xla/kernels/binary_ops.cc
           "Add", "Sub", "Mul", "Div", "Atan2", "Complex", "FloorDiv",
           "FloorMod", "BitwiseAnd", "BitwiseOr", "LeftShift", "RightShift",
           "LogicalAnd", "LogicalOr", "Mod", "Maximum", "Minimum", "RealDiv",
           "ReciprocalGrad", "RsqrtGrad", "SqrtGrad", "SquaredDifference",
           "TruncateDiv", "TruncateMod", "Equal", "NotEqual", "Greater",
           "GreaterEqual", "Less", "LessEqual", "SigmoidGrad", "SoftplusGrad",
           "SoftsignGrad", "TanhGrad", "Pow", "ApproximateEqual",
           // tf2xla/kernels/cast_op.cc
           "Cast",
           // tf2xla/kernels/categorical_op.cc
           "Multinomial" /* (Rng ops are disabled on GPU backend currently)*/,
           // tf2xla/kernels/concat_op.cc
           "Concat", "ConcatV2", "ConcatOffset",
           // tf2xla/kernels/const_op.cc
           "Const",
           // tf2xla/kernels/cross_op.cc
           "Cross",
           // tf2xla/kernels/depthtospace_op.cc
           "DepthToSpace",
           // tf2xla/kernels/diag_op.cc
           "Diag", "DiagPart", "MatrixDiag", "MatrixDiagPart",
           // tf2xla/kernels/dynamic_stitch_op.cc
           "DynamicStitch", "ParallelDynamicStitch",
           // tf2xla/kernels/elu_op.cc
           "Elu", "EluGrad", "Selu", "SeluGrad",
           // tf2xla/kernels/fake_quantize_ops.cc
           "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxArgsGradient",
           "FakeQuantWithMinMaxVars",
           "FakeQuantWithMinMaxVarsGradient" /*(Reduce)*/,
           // tf2xla/kernels/fill_op.cc
           "Fill",
           // tf2xla/kernels/gather_op.cc
           "Gather", "GatherV2", "GatherNd",
           // tf2xla/kernels/identity_op.cc
           "Identity", "IdentityN", "PreventGradient", "StopGradient",
           "Snapshot",
           // tf2xla/kernels/image_ops.cc
           "RGBToHSV", "HSVToRGB", "AdjustContrastv2" /*(Reduce)*/,
           "AdjustSaturation", "AdjustHue",
           // tf2xla/kernels/index_ops.cc
           "ArgMax", "ArgMin",
           // tf2xla/kernels/l2loss_op.cc
           "L2Loss" /*(Reduce)*/,
           // tf2xla/kernels/lrn_ops.cc (ReduceWindow)
           "LRN", "LRNGrad",
           // tf2xla/kernels/matrix_band_part_op.cc
           "MatrixBandPart",
           // tf2xla/kernels/matrix_set_diag_op.cc
           "MatrixSetDiag",
           // tf2xla/kernels/mirror_pad_op.cc
           "MirrorPad",
           // tf2xla/kernels/no_op.cc
           "NoOp", "ControlTrigger",
           // tf2xla/kernels/one_hot_op.cc
           "OneHot",
           // tf2xla/kernels/pack_op.cc
           "Pack",
           // tf2xla/kernels/pad_op.cc
           "Pad", "PadV2",
           // tf2xla/kernels/pooling_ops.cc
           "MaxPool", "MaxPoolV2", "MaxPool3D", "AvgPool",
           "AvgPool3D", /*(all the pooling ops use ReduceWindow)*/
           "MaxPoolGrad", "MaxPoolGradV2", "MaxPool3DGrad", "AvgPoolGrad",
           "AvgPool3DGrad",
           // tf2xla/kernels/quantize_and_dequantize_op.cc (Reduce)
           "QuantizeAndDequantizeV2",
           // tf2xla/kernels/random_ops.cc (Rng ops are disabled on GPU backend
           // currently)
           "RandomUniform", "RandomUniformInt", "RandomStandardNormal",
           "TruncatedNormal",
           // tf2xla/kernels/reduction_ops.cc (Reduce)
           "Sum", "Prod", "Min", "Max", "Mean", "All", "Any",
           // tf2xla/kernels/relu_op.cc
           "Relu", "Relu6", "ReluGrad", "Relu6Grad",
           // tf2xla/kernels/reshape_op.cc
           "Reshape",
           // tf2xla/kernels/reverse_op.cc
           "Reverse", "ReverseV2",
           // tf2xla/kernels/reverse_sequence_op.cc
           "ReverseSequence",
           // tf2xla/kernels/scan_ops.cc (ReduceWindow)
           "Cumsum", "Cumprod",
           // tf2xla/kernels/scatter_nd_op.cc (Reduce)
           "ScatterNd",
           // tf2xla/kernels/segment_reduction_ops.cc (Reduce)
           "UnsortedSegmentSum",
           // tf2xla/kernels/select_op.cc
           "Select",
           // tf2xla/kernels/sequence_ops.cc
           "Range", "LinSpace",
           // tf2xla/kernels/shape_op.cc
           "Shape", "ShapeN", "Rank", "Size", "ExpandDims", "Squeeze",
           "ZerosLike", "OnesLike",
           // tf2xla/kernels/slice_op.cc
           "Slice",
           // tf2xla/kernels/softmax_op.cc (Reduce)
           "Softmax", "LogSoftmax", "SoftmaxCrossEntropyWithLogits",
           "SparseSoftmaxCrossEntropyWithLogits",
           // tf2xla/kernels/spacetobatch_op.cc
           "SpaceToBatchND", "SpaceToBatch",
           // tf2xla/kernels/spacetodepth_op.cc
           "SpaceToDepth",
           // tf2xla/kernels/split_op.cc
           "Split", "SplitV",
           // tf2xla/kernels/stack_ops.cc
           "StackV2", "StackPushV2", "StackPopV2", "StackCloseV2",
           // tf2xla/kernels/stateless_random_ops.cc (Rng ops are disabled on
           // GPU
           // backend currently)
           "StatelessRandomUniform",
           "StatelessRandomNormal"
           // tf2xla/kernels/strided_slice_op.cc
           "StridedSlice",
           "StridedSliceGrad", "ResourceStridedSliceAssign",
           // tf2xla/kernels/tile_ops.cc
           "Tile",
           // tf2xla/kernels/training_ops.cc
           "ResourceApplyGradientDescent", "ResourceApplyMomentum",
           "ResourceApplyAdagrad", "ResourceApplyAdam", "ResourceApplyRMSProp",
           "ResourceApplyFtrl", "ResourceApplyFtrlV2",
           // tf2xla/kernels/transpose_op.cc
           "Transpose", "InvertPermutation",
           // tf2xla/kernels/unary_ops.cc
           "ComplexAbs", "Angle", "Conj", "Abs", "Acos", "Acosh", "Asin",
           "Asinh", "Atan", "Atanh", "Ceil", "Cos", "Cosh", "Sin", "Exp",
           "Expm1", "Floor", "IsFinite", "IsInf", "IsNan", "Inv", "Reciprocal",
           "Log", "Log1p", "Invert", "LogicalNot", "Neg", "Rint", "Round",
           "Rsqrt", "Sigmoid", "Sign", "Sinh", "Softplus", "Softsign", "Sqrt",
           "Square", "Tan", "Tanh", "Real", "Imag",
           // tf2xla/kernels/unpack_op.cc
           "Unpack"});

  return elementwise_ops->count(node.op()) > 0;
}

// Nodes that XLA can compile are put in `candidates`.  Nodes put in
// `isolated_nodes` must either be unclustered or be put in trivial single-node
// clusters.
Status FindCompilationCandidates(
    const Graph& graph, FunctionLibraryDefinition* flib_def, Env* env,
    const std::function<bool(const Node*, const DeviceType&)>& is_compilable_fn,
    OrderedNodeSet* candidates, absl::flat_hash_set<Node*>* isolated_nodes) {
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, env, TF_GRAPH_DEF_VERSION,
                                        flib_def, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  std::vector<bool> compile_time_const_nodes(graph.num_node_ids(), false);
  TF_RETURN_IF_ERROR(
      BackwardsConstAnalysis(graph, /*compile_time_const_arg_indices=*/nullptr,
                             &compile_time_const_nodes, lib_runtime));

  int64& fuel = GetMarkForCompilationPassFlags()->tf_xla_clustering_fuel;

  // Iterate over nodes in sorted order so that compiler fuel is deterministic.
  // We can't simply pass op_nodes().begin() and op_nodes().end to the
  // std::vector constructor because they're not proper iterators, with
  // iterator_traits defined and so on.
  std::vector<Node*> sorted_nodes;
  for (Node* node : graph.op_nodes()) {
    sorted_nodes.push_back(node);
  }
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), NodeComparatorID());

  if (fuel >= std::numeric_limits<int64>::max() / 2) {
    // The assumption is that if fuel started out as INT64_MAX, it will forever
    // stay greater than INT64_MAX / 2.
    VLOG(2) << "Starting fuel: infinity";
  } else {
    VLOG(2) << "Starting fuel: " << fuel;
  }

  for (Node* node : sorted_nodes) {
    if (fuel <= 0) {
      VLOG(1)
          << "Hit fuel limit; not marking any remaining ops as clusterable.";
      break;
    }

    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceToDeviceType(node->assigned_device_name(), &device_type));
    VLOG(4) << "Device type for " << node->name() << ": "
            << device_type.type_string();

    if (is_compilable_fn && !is_compilable_fn(node, device_type)) {
      // is_compilable_fn has already logged the reason if it returned false.
      continue;
    }

    const XlaOpRegistry::DeviceRegistration* registration;
    CHECK(
        XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration));
    DeviceType jit_device_type(registration->compilation_device_name);

    bool always_auto_cluster = registration->autoclustering_policy ==
                               XlaOpRegistry::AutoclusteringPolicy::kAlways;

    OperationFilter op_filter;
    op_filter.allow_resource_ops = registration->compile_resource_ops;
    op_filter.allow_stateful_rng_ops = always_auto_cluster;
    op_filter.allow_control_trigger = always_auto_cluster;
    op_filter.allow_dummy_ops = always_auto_cluster;
    op_filter.allow_ops_producing_or_consuming_variant = always_auto_cluster;

    if (!HasXLAKernel(*node, jit_device_type) &&
        !IsCompilableCall(node->def(), jit_device_type, op_filter, 0,
                          lib_runtime)) {
      VLOG(2) << "Rejecting " << node->name() << ": unsupported op "
              << node->type_string();
      continue;
    }

    if (!op_filter.allow_stateful_rng_ops &&
        IsStatefulRandomOp(node->type_string())) {
      VLOG(2) << "Rejecting " << node->name() << ": stateful random operation";
      continue;
    }
    if (!op_filter.allow_control_trigger && node->IsControlTrigger()) {
      VLOG(2) << "Rejecting " << node->name() << ": is a control trigger op";
      continue;
    }
    if (!op_filter.allow_dummy_ops && IsDummyImplOp(node->type_string())) {
      VLOG(2) << "Rejecting " << node->name() << ": dummy op ("
              << node->type_string() << ")";
      continue;
    }
    if (!op_filter.allow_ops_producing_or_consuming_variant &&
        OpProducesOrConsumesVariant(*node)) {
      VLOG(2) << "Rejecting " << node->name()
              << ": produces or consumes DT_VARIANT";
      continue;
    }

    if (!op_filter.allow_resource_ops &&
        (HasResourceOutput(*node) || IsNonResourceVarResourceOp(*node))) {
      // We don't have a way of returning values of type DT_RESOURCE from XLA
      // computations so we avoid auto-clustering nodes producing DT_RESOURCE.
      // XlaLaunchOp also cannot snapshot resources that are not resource
      // variables so we avoid clustering resource operations that operate on
      // non-resource variables.
      VLOG(2) << "Rejecting: " << node->name() << ": resource output "
              << node->type_string();
      continue;
    }

    if (compile_time_const_nodes[node->id()]) {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(
          graph.op_registry()->LookUpOpDef(node->type_string(), &op_def));
      if (op_def->is_stateful()) {
        // It is easiest to demonstrate the problem we're trying to solve with
        // an example.  Say we have this graph:
        //
        //   shape = RandomUniformInt();
        //   reshape = Reshape(input, shape)
        //
        // Both RandomUniformInt and Reshape are compilable by XLA so, absent
        // any other reason, we will try to put both shape and reshape in the
        // same cluster.  However, since XLA only supports statically shaped
        // values, it will expect to be able to constant fold `shape` to get a
        // static shape for `reshape`.  This is a problem because side-effecting
        // ops like RandomUniformInt() cannot be constant folded.  We fix this
        // by putting `shape` and `reshape` in different clusters, which results
        // in us recompiling `reshape`'s cluster for every new value of `shape`,
        // making `reshape` statically sized within each compilation.  We
        // simplify the solution even further by disallowing operations like
        // `shape` from being part of *any* non-trivial cluster.  They're either
        // not compiled by XLA altogether or, if assigned to an XLA_* device
        // with "must compile" semantics, compiled into a trivial single-op
        // cluster.  This approach leaves some room for improvement, and we can
        // consider implementing a more aggressive data-flow-analysis based
        // solution in the future if needed.
        //
        // One ugly problem we have to contend with: certain sets of ops *have*
        // to be in the same cluster because values flowing between them have
        // types that can't be live-in or live-out of a cluster.  These ops are:
        //
        //  - TensorArray ops operating on the same TensorArray instance.
        //  - Stack ops operating on the same Stack instance.
        //
        // To work around this we avoid isolating these specific ops.  Because
        // of this concession it is unsound to auto-cluster them because then
        // we'd create clusters we could not compile (because we can't constant
        // fold, say, a TensorArrayRead or a StackPopV2).  But we don't
        // auto-cluster these operations today so we're good for now.
        const XlaResourceOpInfo* op_info =
            GetResourceOpInfoForOp(node->type_string());
        bool is_tensor_array_or_stack_op =
            op_info && op_info->resource_kind() != XlaResourceKind::kVariable;
        if (!is_tensor_array_or_stack_op) {
          VLOG(2) << "Isolating " << node->name()
                  << ": must-be-constant stateful op";
          isolated_nodes->insert(node);
          // Keep going and execute all the other checks.
        }
      }
    }
    // We don't auto-cluster functional control flow nodes containing resource
    // operations because safety checks are trickier in this case.
    // registration->compile_resource_ops is true for XLA_CPU/XLA_GPU but not
    // for CPU/GPU.
    if (node->type_string() == "While" &&
        !IsCompilableWhile(*node, jit_device_type, op_filter, 0, lib_runtime)) {
      continue;
    }
    // _Arg nodes in a top-level function represent feeds.
    // Do not compile them.
    if (node->type_string() == "_Arg") {
      continue;
    }
    // _Retval nodes in a top-level function represent fetches.
    // Do not compile them.
    if (node->type_string() == "_Retval") {
      continue;
    }
    candidates->insert(node);
    --fuel;
  }
  VLOG(2) << "candidates->size() = " << candidates->size();
  return Status::OK();
}

struct Cluster {
  // Identifies the node that represents this cluster in the cycle detection
  // graph.
  int representative = -1;

  // The set of devices the nodes in this cluster are placed on.
  absl::flat_hash_set<string> devices;

  // If there are resource operation in the cluster then this is the device that
  // resource operations are placed on.  All resource operations in a cluster
  // must be placed on the same device.
  string resource_op_device;

  // True if any node in the cluster has an _XlaCompile attribute set to true.
  bool has_xla_compile_attr;
};

}  // anonymous namespace

bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef) {
  Device* device = flr->device();
  const XlaOpRegistry::DeviceRegistration* registration;
  CHECK(XlaOpRegistry::GetCompilationDevice(device->device_type(),
                                            &registration));
  DeviceType jit_device_type(registration->compilation_device_name);

  // We can always *compile* resource operations, stateful RNGs and dummy ops,
  // even if we are sometimes unable to auto-cluster them.
  OperationFilter op_filter;
  op_filter.allow_resource_ops = true;
  op_filter.allow_stateful_rng_ops = true;
  op_filter.allow_control_trigger = true;
  op_filter.allow_dummy_ops = true;
  op_filter.allow_ops_producing_or_consuming_variant = true;

  return IsCompilableCall(ndef, jit_device_type, op_filter, 0, flr);
}

Status MarkForCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  // TODO(phawkins): precompute the "GetCompilationDevice" properties of each
  // device ahead of time.
  OptimizerOptions::GlobalJitLevel global_jit_level =
      GetGlobalJitLevel(options);
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  bool fusion_only = flags->tf_xla_fusion_only;

  VLOG(1) << "flags->tf_xla_fusion_only = " << flags->tf_xla_fusion_only;
  VLOG(1) << "flags->tf_xla_auto_jit = " << flags->tf_xla_auto_jit;
  const FunctionLibraryDefinition* fld = options.flib_def;

  // Deadness analysis expects a graph with source and sink edges properly
  // connected but sometimes the incoming graph does not follow this invariant.
  // So fix up the source and sink edges before calling into deadness analysis.
  FixupSourceAndSinkEdges(options.graph->get());

  // See explanation on `kXlaAlreadyClustered`.
  for (Node* n : options.graph->get()->nodes()) {
    if (n->attrs().Find(kXlaAlreadyClustered)) {
      return Status::OK();
    }
  }

  std::unique_ptr<DeadnessAnalysis> deadness;
  {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("DeadnessAnalysis", 1);
    TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(**options.graph, &deadness));
  }

  bool deadness_analysis_disabled =
      GetMarkForCompilationPassFlags()
          ->tf_xla_disable_deadness_safety_checks_for_debugging;

  if (deadness_analysis_disabled) {
    LOG(WARNING) << "Deadness analysis was manually disabled via "
                    "--tf_xla_disable_deadness_safety_checks_for_debugging; "
                    "auto-clustering "
                    "is unsound!";
  }

  auto is_compilable = [&](const Node* node, const DeviceType& device_type) {
    const XlaOpRegistry::DeviceRegistration* registration;
    if (!XlaOpRegistry::GetCompilationDevice(device_type.type(),
                                             &registration)) {
      VLOG(2) << "Rejecting " << node->name() << ": could not find JIT device.";
      return false;
    }

    // If there is a _XlaCompile annotation, use its value.
    bool compile = false;
    Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
    if (status.ok()) {
      if (!compile) {
        VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
                << kXlaCompileAttr << ") is false.";
      }
      return compile;
    }

    status = fld->GetAttr(*node, kXlaCompileAttr, &compile);
    if (status.ok()) {
      if (!compile) {
        VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
                << kXlaCompileAttr << ") on callee is false.";
      }
      return compile;
    }

    // If inputs to `node` can have conflicting deadness (i.e. some are alive
    // and some are dead) then don't compile it.  XLA cannot represent the
    // deadness semantics of these nodes correctly and auto-clustering these
    // nodes can cause deadness to propagate to nodes that should be live.
    if (!deadness_analysis_disabled) {
      if (node->IsMerge() ||
          deadness->HasInputsWithMismatchingDeadness(*node)) {
        VLOG(2) << "Rejecting " << node->name() << ": mismatching deadness.";
        return false;
      }
    }

    // Check for fusable ops only if requested.
    if (global_jit_level > 0 && fusion_only && !IsXlaFusable(node->def())) {
      VLOG(2) << "Rejecting " << node->name()
              << ": not fusable op but fusion_only enabled.";
      return false;
    }

    return true;
  };

  return RunImpl(options, is_compilable);
}

static string RatioToString(int numerator, int denominator) {
  return absl::StrFormat("%d / %d (%.2f%%)", numerator, denominator,
                         (100.0 * numerator) / denominator);
}

static void VLogClusteringSummary(const Graph& g) {
  if (!VLOG_IS_ON(2)) {
    return;
  }

  std::map<absl::string_view, int> cluster_name_to_size;
  std::map<absl::string_view, std::map<absl::string_view, int>>
      cluster_name_to_op_histogram;
  std::map<absl::string_view, int> unclustered_op_histogram;
  int clustered_node_count = 0;

  for (Node* n : g.nodes()) {
    absl::optional<absl::string_view> cluster_name = GetXlaClusterForNode(*n);
    if (cluster_name) {
      clustered_node_count++;
      cluster_name_to_size[*cluster_name]++;
      cluster_name_to_op_histogram[*cluster_name][n->type_string()]++;
    } else {
      unclustered_op_histogram[n->type_string()]++;
    }
  }

  int unclustered_node_count = g.num_nodes() - clustered_node_count;

  VLOG(2) << "*** Clustering info for graph of size " << g.num_nodes();
  VLOG(2) << " Built " << cluster_name_to_size.size() << " clusters, size "
          << RatioToString(clustered_node_count, g.num_nodes());

  for (const auto& cluster_name_size_pair : cluster_name_to_size) {
    absl::string_view cluster_name = cluster_name_size_pair.first;
    int size = cluster_name_size_pair.second;
    VLOG(2) << "  " << cluster_name << " "
            << RatioToString(size, g.num_nodes());
    for (const auto& op_count_pair :
         cluster_name_to_op_histogram[cluster_name]) {
      VLOG(3) << "   " << op_count_pair.first << ": " << op_count_pair.second
              << " instances";
    }
  }

  if (!unclustered_op_histogram.empty()) {
    VLOG(2) << " Unclustered nodes: "
            << RatioToString(unclustered_node_count, g.num_nodes());
    for (const auto& pair : unclustered_op_histogram) {
      VLOG(3) << "  " << pair.first << ": " << pair.second << " instances";
    }
  }

  struct EdgeInfo {
    absl::string_view node_name;
    absl::optional<absl::string_view> cluster_name;

    absl::string_view GetClusterName() const {
      return cluster_name ? *cluster_name : "[none]";
    }

    std::pair<absl::string_view, absl::optional<absl::string_view>> AsPair()
        const {
      return {node_name, cluster_name};
    }

    bool operator<(const EdgeInfo& other) const {
      return AsPair() < other.AsPair();
    }
  };

  using EdgeInfoMap = std::map<absl::string_view, std::map<EdgeInfo, int64>>;

  EdgeInfoMap incoming_edge_infos;
  EdgeInfoMap outgoing_edge_infos;

  std::set<absl::string_view> cluster_names_to_print;

  for (const Edge* e : g.edges()) {
    const Node* from = e->src();
    absl::optional<absl::string_view> from_cluster_name =
        GetXlaClusterForNode(*from);

    const Node* to = e->dst();
    absl::optional<absl::string_view> to_cluster_name =
        GetXlaClusterForNode(*to);

    if (to_cluster_name == from_cluster_name) {
      continue;
    }

    if (to_cluster_name) {
      incoming_edge_infos[*to_cluster_name]
                         [EdgeInfo{from->name(), from_cluster_name}]++;
      cluster_names_to_print.insert(*to_cluster_name);
    }

    if (from_cluster_name) {
      outgoing_edge_infos[*from_cluster_name][{to->name(), to_cluster_name}]++;
      cluster_names_to_print.insert(*from_cluster_name);
    }
  }

  VLOG(2) << "*** Inter-Cluster edges:";
  if (cluster_names_to_print.empty()) {
    VLOG(2) << "   [none]";
  }

  auto print_edge_info_set_for_cluster = [&](absl::string_view cluster_name,
                                             const EdgeInfoMap& edge_info_map,
                                             absl::string_view desc) {
    auto it = edge_info_map.find(cluster_name);
    if (it != edge_info_map.end()) {
      VLOG(2) << "  " << it->second.size() << " " << desc << " edges";
      for (const auto& edge_info_count_pair : it->second) {
        VLOG(2) << "   " << edge_info_count_pair.first.GetClusterName() << " "
                << edge_info_count_pair.first.node_name << " # "
                << edge_info_count_pair.second;
      }
    } else {
      VLOG(2) << "  No " << desc << " edges.";
    }
  };

  for (absl::string_view cluster_name : cluster_names_to_print) {
    VLOG(2) << " ** Cluster " << cluster_name;
    print_edge_info_set_for_cluster(cluster_name, incoming_edge_infos,
                                    "incoming");
    print_edge_info_set_for_cluster(cluster_name, outgoing_edge_infos,
                                    "outgoing");
  }
}

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
static bool IsShapeConsumerOp(const Node& node) {
  return node.type_string() == "Shape" || node.type_string() == "Rank" ||
         node.type_string() == "Size";
}

static Status IgnoreResourceOpForSafetyAnalysis(const Node& n, bool* ignore) {
  // If a resource operation is assigned to XLA_CPU or XLA_GPU explicitly then
  // ignore it during resource operation safety analysis.  We need this hack
  // because of two reasons:
  //
  //  1. Operations assigned to XLA_CPU and XLA_GPU have to always be compiled.
  //  2. We don't support live-out values of type DT_RESOURCE and live-in values
  //     of type DT_RESOURCE that are not resource variables.
  //
  // Together these imply we cannot let resource variable safety analysis
  // constrain e.g. a TensorArrayV3->TensorArrayAssignV3 edge to be in different
  // clusters: both of them will have to be clustered because of (1) and we
  // won't be able to keep the edge between the two as neither the input to the
  // second XLA cluster nor the output from the first XLA cluster are supported
  // because of (2).
  //
  // TODO(b/113100872): This can be fixed if the TensorFlow representation for
  // TensorArray and Stack on the XLA_{C|G}PU devices were the same in XLA; then
  // (2) would no longer hold.

  if (n.assigned_device_name().empty()) {
    *ignore = false;
    return Status::OK();
  }
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceToDeviceType(n.assigned_device_name(), &device_type));

  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    *ignore = true;
  } else {
    *ignore = registration->compile_resource_ops;
  }
  return Status::OK();
}

// Sequence number generator to ensure clusters have unique names.
static std::atomic<int64> cluster_sequence_num;

// Returns true if the devices in `cluster_a` and `cluster_b` are compatible and
// therefore not a hindrance for combining the two clusters into a larger
// cluster.
static Status AreDevicesCompatible(
    const Cluster& cluster_a, const Cluster& cluster_b,
    OptimizerOptions::GlobalJitLevel global_jit_level, bool* result) {
  std::vector<string> devices;
  absl::c_remove_copy(cluster_a.devices, std::back_inserter(devices), "");
  absl::c_remove_copy(cluster_b.devices, std::back_inserter(devices), "");
  absl::c_sort(devices);

  if (devices.empty()) {
    *result = false;
    return Status::OK();
  }

  // First check if we will even be able to pick a device for the larger
  // combined cluster.
  bool can_pick_device;
  TF_RETURN_IF_ERROR(CanPickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &can_pick_device));
  if (!can_pick_device) {
    *result = false;
    return Status::OK();
  }

  string chosen_device;
  TF_RETURN_IF_ERROR(PickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &chosen_device));

  // If we are able to pick a device `chosen_device` for the larger cluster, the
  // resource operations in `cluster_a` and `cluster_b` must be placed on the
  // same device as `chosen_device`.  This is because the _XlaCompile and
  // _XlaRun kernels are going to run on and therefore try to access the
  // resource variables from `chosen_device`, which will be an error if the
  // resource variables are placed on some other device.
  auto resource_op_device_ok = [&](const string& resource_op_device) {
    return resource_op_device.empty() || resource_op_device == chosen_device;
  };

  *result = resource_op_device_ok(cluster_a.resource_op_device) &&
            resource_op_device_ok(cluster_b.resource_op_device);
  if (!*result) {
    return Status::OK();
  }

  // We will check this again later, but here we prune out clusters that would
  // never have been sent to XLA to save compile time.  Without this change we
  // will e.g. create a CPU cluster only to later notice that the user did not
  // enable the CPU JIT via --tf_xla_cpu_global_jit.  With this change we avoid
  // creating the cluster to begin with.
  //
  // TODO(b/126629785): It is possible that this is just papering over O(n^2)
  // behavior in our clustering algorithm.
  const XlaOpRegistry::DeviceRegistration* registration;
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(DeviceToDeviceType(chosen_device, &device_type));
  TF_RET_CHECK(
      XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration))
      << "chosen device = " << chosen_device
      << "; device type = " << device_type.type() << "; devices ("
      << devices.size() << ") = " << absl::StrJoin(devices, ", ");

  *result = cluster_a.has_xla_compile_attr || cluster_b.has_xla_compile_attr ||
            registration->autoclustering_policy ==
                XlaOpRegistry::AutoclusteringPolicy::kAlways ||
            (registration->autoclustering_policy ==
                 XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
             global_jit_level != OptimizerOptions::OFF);

  return Status::OK();
}

// Returns `true` iff we should compile `cluster`.
static Status ShouldCompileClusterImpl(
    const Cluster& cluster, OptimizerOptions::GlobalJitLevel global_jit_level,
    bool* should_compile, string* device) {
  std::vector<string> devices;
  absl::c_remove_copy(cluster.devices, std::back_inserter(devices), "");
  absl::c_sort(devices);

  string chosen_device;
  TF_RETURN_IF_ERROR(PickDeviceForXla(
      devices, /*allow_mixing_unknown_and_cpu=*/false, &chosen_device));

  const XlaOpRegistry::DeviceRegistration* registration;
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(DeviceToDeviceType(chosen_device, &device_type));
  TF_RET_CHECK(
      XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration))
      << "chosen device = " << chosen_device
      << "; device type = " << device_type.type() << "; devices ("
      << devices.size() << ") = " << absl::StrJoin(devices, ", ");

  *should_compile =
      cluster.has_xla_compile_attr ||
      registration->autoclustering_policy ==
          XlaOpRegistry::AutoclusteringPolicy::kAlways ||
      (registration->autoclustering_policy ==
           XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
       global_jit_level != OptimizerOptions::OFF);

  VLOG(3) << (*should_compile ? "Compiling" : "Not compiling")
          << " cluster with device " << chosen_device;

  *device = std::move(chosen_device);
  return Status::OK();
}

static Status ShouldCompileCluster(
    absl::flat_hash_map<int, std::pair<bool, string>>* cache,
    OptimizerOptions::GlobalJitLevel global_jit_level, const Cluster& cluster,
    bool* should_compile, string* device) {
  auto it = cache->find(cluster.representative);
  if (it != cache->end()) {
    *should_compile = it->second.first;
    *device = it->second.second;
    return Status::OK();
  }

  string device_s;
  TF_RETURN_IF_ERROR(ShouldCompileClusterImpl(cluster, global_jit_level,
                                              should_compile, &device_s));
  cache->insert({cluster.representative, {*should_compile, device_s}});
  *device = std::move(device_s);
  return Status::OK();
}

Status MarkForCompilationPass::RunImpl(
    const GraphOptimizationPassOptions& options,
    const std::function<bool(const Node*, const DeviceType&)>&
        is_compilable_fn) {
  VLOG(1) << "MarkForCompilationPass::Run";

  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();

  Graph* graph = options.graph->get();

  OrderedNodeSet compilation_candidates;
  absl::flat_hash_set<Node*> isolated_nodes;
  TF_RETURN_IF_ERROR(FindCompilationCandidates(
      *graph, options.flib_def,
      (options.session_options != nullptr) ? options.session_options->env
                                           : Env::Default(),
      is_compilable_fn, &compilation_candidates, &isolated_nodes));

  if (compilation_candidates.empty()) {
    VLOG(2) << "No compilable candidates";
    return Status::OK();
  }

  GraphCycles cycles;
  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph, &cycles));
  if (!cycle_detection_graph_ok) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(AdjustCycleDetectionGraphForResourceOps(
      graph, options.flib_def, IgnoreResourceOpForSafetyAnalysis, &cycles));

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative
  // names the node in the 'cycles' graph that represents the cluster.
  std::vector<UnionFind<Cluster>> clusters(graph->num_node_ids());
  std::deque<UnionFind<Cluster>*> worklist;
  for (Node* node : compilation_candidates) {
    Cluster& cluster = clusters[node->id()].Get();
    cluster.representative = node->id();
    const string& device = !node->assigned_device_name().empty()
                               ? node->assigned_device_name()
                               : node->requested_device();
    if (HasResourceInput(*node) || HasResourceOutput(*node)) {
      cluster.resource_op_device = device;
    }
    cluster.has_xla_compile_attr = false;
    bool xla_compile_attr;
    if (GetNodeAttr(node->attrs(), kXlaCompileAttr, &xla_compile_attr).ok()) {
      cluster.has_xla_compile_attr |= xla_compile_attr;
    }
    if (options.flib_def->GetAttr(*node, kXlaCompileAttr, &xla_compile_attr)
            .ok()) {
      cluster.has_xla_compile_attr |= xla_compile_attr;
    }

    cluster.devices.insert(device);
    worklist.push_back(&clusters[node->id()]);
  }

  OptimizerOptions::GlobalJitLevel global_jit_level =
      GetGlobalJitLevel(options);
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();

  // Repeatedly contract edges between clusters that are on the same device,
  // provided the contraction would not create a cycle.
  //
  // TODO(hpucha): Handle the case where kXlaClusterAttr is already set (for
  // example, from the Grappler fusion pass).
  while (!worklist.empty()) {
    Cluster* cluster_from = &worklist.front()->Get();
    int from = cluster_from->representative;
    worklist.pop_front();

    Node* node_from = graph->FindNodeId(from);
    if (node_from->IsControlFlow()) {
      // Control flow nodes aren't compilation candidates and should never
      // appear.
      return errors::Internal(
          "Found control flow node in clustering worklist: ",
          node_from->type_string());
    }

    if (isolated_nodes.count(node_from)) {
      continue;
    }

    string from_scope;
    string to_scope;
    for (int to : cycles.Successors(from)) {
      if (to >= graph->num_node_ids()) {
        // Node is a fictitious node that is present only in the cycle detection
        // graph. No clustering is possible.
        continue;
      }

      const Cluster& cluster_to = clusters[to].Get();
      Node* node_to = graph->FindNodeId(to);
      if (compilation_candidates.find(node_to) ==
          compilation_candidates.cend()) {
        continue;
      }
      bool devices_compatible;
      TF_RETURN_IF_ERROR(AreDevicesCompatible(
          *cluster_from, cluster_to, global_jit_level, &devices_compatible));
      if (!devices_compatible) {
        continue;
      }
      if (isolated_nodes.count(node_to)) {
        continue;
      }
      // Look for an _XlaScope on both nodes.  If both nodes have a
      // scope and the scopes do not match, do not cluster along this
      // edge. This restriction is overridden if the global_jit_level is ON. If
      // even one of the nodes lacks an _XlaScope attribute,
      // then it is treated as a "bridge" and a cluster may be created
      // along it.  We may want to restrict this behavior to require
      // all nodes marked with _XlaCompile=true to also have a
      // _XlaScope property set (and raise an error otherwise); but
      // for now we don't do this.
      if (global_jit_level == OptimizerOptions::OFF &&
          GetNodeAttr(node_from->attrs(), kXlaScopeAttr, &from_scope).ok() &&
          GetNodeAttr(node_to->attrs(), kXlaScopeAttr, &to_scope).ok() &&
          from_scope != to_scope) {
        continue;
      }

      // Ops that consume shapes cannot be the root of a cluster. This is an
      // optimization.
      if (clusters[from].Size() == 1 && IsShapeConsumerOp(*node_from)) {
        continue;
      }

      // Don't exceed the maximum cluster size.
      if (clusters[from].Size() + clusters[to].Size() >
          flags->tf_xla_max_cluster_size) {
        continue;
      }

      // If any of the consumer's producers are on a different device, do not
      // cluster these nodes. This prevents other work on this device from being
      // delayed by work on other devices. We consider predecessors of the
      // entire cluster rather than just the inputs to the node to prevent the
      // cluster still being combined in cases where the 'to' cluster has
      // multiple dependencies on the 'from' cluster and another dependency
      // leads to a merging of the clusters.
      //
      // TODO(b/117085735): We probably want to handle the reciprocal of this
      // case where a cluster is producing data for multiple devices.
      bool found_split = false;
      for (const auto& in_id : cycles.Predecessors(to)) {
        if (in_id >= graph->num_node_ids()) continue;

        Node* in = graph->FindNodeId(in_id);
        const Cluster& cluster_in = clusters[in_id].Get();
        if (compilation_candidates.find(in) != compilation_candidates.cend()) {
          bool devices_compatible;
          TF_RETURN_IF_ERROR(AreDevicesCompatible(
              cluster_to, cluster_in, global_jit_level, &devices_compatible));
          if (!devices_compatible) {
            found_split = true;
          }
        }
      }
      if (found_split) continue;

      // If contracting the edge would create a cycle, bail out.
      // However, just because we can't merge the clusters now does not mean
      // we won't be able to merge them in the future.
      // e.g., if we have edges 1->2, 2->3 and 1->3, we cannot contract edge
      // 1->3. But if we first contract 1->2 then we can later contract 1->3.
      if (!cycles.ContractEdge(from, to)) continue;

      // Merge the clusters. ContractEdge uses 'from' as the number of the
      // merged node, so make sure 'from' is the chosen representative.
      cluster_from->devices.insert(cluster_to.devices.begin(),
                                   cluster_to.devices.end());
      if (!cluster_to.resource_op_device.empty()) {
        cluster_from->resource_op_device = cluster_to.resource_op_device;
      }
      cluster_from->has_xla_compile_attr |= cluster_to.has_xla_compile_attr;
      clusters[from].Merge(&clusters[to]);

      worklist.push_back(&clusters[from]);
      break;
    }
  }

  // Count the number of non-trivial elements in each cluster.
  std::vector<int> effective_cluster_sizes(graph->num_node_ids());
  for (const Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;
    // Identity nodes will be removed if the node gets marked for compilation.
    // Therefore we don't want to count them towards the effective cluster size.
    if (n->def().op() != "Identity") {
      effective_cluster_sizes[cluster]++;
    }
  }

  // Names for each cluster.
  std::unordered_map<int, string> cluster_names;

  if (flags->tf_xla_clustering_debug) {
    DumpGraphToFile("before_mark_for_compilation", **options.graph,
                    options.flib_def);
  }

  absl::flat_hash_map<int, std::pair<bool, string>>
      should_compile_cluster_cache;

  // Mark clusters for compilation that:
  // * are placed on a device that requires compilation (an XlaDevice),
  // * are explicitly marked for compilation (_XlaCompile=true), or
  // * have more than flags->tf_xla_min_cluster_size elements (applicable only
  //   if compilation is enabled, otherwise there will be no such candidates).
  const int min_cluster_size = flags->tf_xla_min_cluster_size;
  for (Node* n : compilation_candidates) {
    const Cluster& cluster = clusters[n->id()].Get();
    bool should_compile;
    string device;
    TF_RETURN_IF_ERROR(ShouldCompileCluster(&should_compile_cluster_cache,
                                            global_jit_level, cluster,
                                            &should_compile, &device));
    if (!should_compile) {
      continue;
    }

    int cluster_repr = cluster.representative;

    // Compile if the user marked this node _XlaCompile=true
    bool compile_attr = false;
    bool marked_for_compilation = false;
    if (GetNodeAttr(n->attrs(), kXlaCompileAttr, &compile_attr).ok()) {
      marked_for_compilation = compile_attr;
    } else if (options.flib_def->GetAttr(*n, kXlaCompileAttr, &compile_attr)
                   .ok()) {
      marked_for_compilation = compile_attr;
    }

    // Compile if this is a cluster of >= min_cluster_size compilable operators.
    // Also, always compile if it contains at least one op that is marked for
    // compilation that is not an Identity op.
    if (effective_cluster_sizes[cluster_repr] >= min_cluster_size ||
        (effective_cluster_sizes[cluster_repr] > 0 && marked_for_compilation)) {
      string& name = cluster_names[cluster_repr];

      if (name.empty()) {
        name = absl::StrCat("cluster_", cluster_sequence_num++);
      }
      n->AddAttr(kXlaClusterAttr, name);
      n->AddAttr(kXlaAlreadyClustered, true);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  if (flags->tf_xla_clustering_debug) {
    DumpGraphToFile("mark_for_compilation", **options.graph, options.flib_def);

    // We also dump out an annoated version of the TF graph where the nodes
    // names are prefixed with the cluster names.  This can help visualizing the
    // clustering decisions on TensorBoard.
    Graph new_graph((*options.graph)->op_registry());
    CopyGraph(**options.graph, &new_graph);

    for (Node* n : new_graph.nodes()) {
      if (absl::optional<absl::string_view> cluster_name =
              GetXlaClusterForNode(*n)) {
        n->set_name(absl::StrCat(*cluster_name, "/", n->name()));
      } else if (n->type_string() == "VarHandleOp") {
        n->set_name(absl::StrCat("varhandle/", n->name()));
      } else {
        // There is room for improvement here.  In particular, it may help to
        // split these unclustered nodes into classes where every node in a
        // specific class has edges to and from the same set of clusters.
        n->set_name(absl::StrCat("unclustered/", n->name()));
      }
    }

    DumpGraphToFile("mark_for_compilation_annotated", new_graph,
                    options.flib_def);
  }

  VLogClusteringSummary(*graph);

  return Status::OK();
}

}  // namespace tensorflow
