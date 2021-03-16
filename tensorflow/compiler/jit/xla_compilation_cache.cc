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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include <numeric>

#include "tensorflow/compiler/mlir/mlir_bridge_rollout_policy.h"
#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

constexpr int64 XlaCompilationCache::kDefaultCompilationThreshold;

XlaCompilationCache::XlaCompilationCache(xla::LocalClient* client,
                                         DeviceType device_type)
    : client_(client), device_type_(std::move(device_type)) {}

XlaCompilationCache::~XlaCompilationCache() {
  // Ensure any use of our programs have completed by waiting for all stream
  // executors to complete.
  for (auto* executor : client_->backend().stream_executors()) {
    bool ok = executor->SynchronizeAllActivity();
    if (!ok) {
      LOG(ERROR) << "Error synchronizing activity while waiting for all "
                    "programs to complete";
    }
  }
  // TODO(b/110813685): Think about the program ownership model. Programs are
  // currently owned by the compilation cache which means we must wait for
  // program completion in the destructor. There are multiple compilation caches
  // around, which complicates things a little. Perhaps having programs be
  // shared_ptrs (an invasive change) would make the model easier to reason
  // about?
}

string XlaCompilationCache::DebugString() const {
  return "XLA JIT compilation cache";
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string XlaCompilationCache::Signature::HumanString() const {
  string result = name;
  for (const auto& a : arg_shapes) {
    absl::StrAppend(&result, ",", DataTypeString(a.first));
    absl::StrAppend(&result, " [", absl::StrJoin(a.second, ","), "]");
  }

  for (const auto& v : arg_values) {
    absl::StrAppend(&result, "; ", v.DebugString());
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_shapes != other.arg_shapes) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (int i = 0, end = arg_values.size(); i < end; ++i) {
    if (arg_values[i].dtype() != other.arg_values[i].dtype() ||
        arg_values[i].shape() != other.arg_values[i].shape() ||
        arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
      return false;
    }
  }
  return true;
}

uint64 XlaCompilationCache::Signature::Hash::operator()(
    const XlaCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.arg_shapes) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
  }
  for (const auto& arg : signature.arg_values) {
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
  }
  return h;
}

xla::StatusOr<XlaCompilationCache::Signature>
XlaCompilationCache::BuildSignature(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));

  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
      case XlaCompiler::Argument::kConstantResource:
        signature.arg_values.push_back(arg.constant_value);
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.arg_shapes.emplace_back(arg.type,
                                          arg.DimensionSizesAsInlinedVector());
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }
  return std::move(signature);
}

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0, end = result.xla_input_shapes.size(); i < end; ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(options.device_ordinal != -1
                                       ? options.device_ordinal
                                       : client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator);
  build_options.set_alias_passthrough_params(options.alias_passthrough_params);
  build_options.mutable_debug_options()->set_xla_detailed_logging(
      options.detailed_logging);
  TF_ASSIGN_OR_RETURN(
      auto executables,
      client_->Compile(*result.computation, argument_layouts, build_options));
  TF_RET_CHECK(executables.size() == 1);
  *executable = std::move(executables[0]);
  return Status::OK();
}

Status XlaCompilationCache::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const XlaCompiler::CompileOptions& compile_options,
    CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  absl::optional<int64> compile_threshold;
  if (compile_mode == CompileMode::kLazy) {
    compile_threshold = kDefaultCompilationThreshold;
  }
  auto compile_fn = [&](XlaCompiler* compiler,
                        XlaCompiler::CompilationResult* result) {
    return compiler->CompileFunction(compile_options, function, args, result);
  };
  return CompileImpl(options, function, args, compile_fn,
                     /*compile_threshold=*/compile_threshold,
                     out_compilation_result, out_executable);
}

static bool ShouldBeMegamorphic(int64 compile_count, int64 execution_count) {
  const int64 kCompileThreshold = 10;
  const int64 kMinExecutionsPerCompile = 50;

  // This heuristic is trying to capture the following property: have we sunk a
  // certain minimum amount of compile time into the cluster that didn't quite
  // "pay off"?
  return compile_count > kCompileThreshold &&
         execution_count < kMinExecutionsPerCompile * compile_count;
}

// Creates a simple graph using the specified op as the only op apart from the
// arg and retval nodes.
static xla::StatusOr<std::unique_ptr<Graph>> CreateGraph(
    const NodeDef& node_def, absl::Span<const XlaCompiler::Argument> args,
    absl::Span<const DataType> result_types) {
  // TODO(b/74182462): We implement this by creating a new dummy Graph including
  // _Arg nodes, and let CompileGraph walk it. This could be optimized.
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Status status;
  // First create the actual node we care about computing.
  Node* main_node = graph->AddNode(node_def, &status);
  TF_RETURN_IF_ERROR(status);

  // Create dummy _Arg nodes. Link these to `node` and also via a control
  // dependency edge to the _SOURCE node.
  for (int64 i = 0, end = args.size(); i < end; ++i) {
    Node* node;
    string arg_name = absl::StrCat("_arg", i);
    Status status =
        NodeBuilder(arg_name, FunctionLibraryDefinition::kArgOp)
            .ControlInput(graph->source_node())
            .Attr("T", args[i].kind == XlaCompiler::Argument::kResource
                           ? DT_RESOURCE
                           : args[i].type)
            .Attr("index", i)
            .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(node, 0, main_node, i);
  }

  // Similarly with return values, create dummy _Retval nodes fed by `node`.
  for (int64 i = 0, end = result_types.size(); i < end; ++i) {
    Node* node;
    string retval_name = absl::StrCat("_retval", i);
    Status status = NodeBuilder(retval_name, FunctionLibraryDefinition::kRetOp)
                        .Input(main_node, i)
                        .Attr("T", result_types[i])
                        .Attr("index", i)
                        .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
  }
  FixupSourceAndSinkEdges(graph.get());
  return graph;
}

Status XlaCompilationCache::CompileSingleOp(
    const XlaCompiler::Options& options,
    absl::Span<const XlaCompiler::Argument> args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  // Remove the "_class" attribute from the attribute set used to create the
  // compilation cache key. This attribute is information for the colocator
  // and causes false uniqueness between nodes.
  name.mutable_attr()->erase("_class");
  auto compile_op = [&](XlaCompiler* compiler,
                        XlaCompiler::CompilationResult* result) {
    std::vector<DataType> result_dtypes(ctx->num_outputs());
    for (int i = 0, end = result_dtypes.size(); i < end; ++i) {
      result_dtypes[i] = ctx->expected_output_dtype(i);
    }

    const NodeDef& node_def = ctx->op_kernel().def();
    TF_ASSIGN_OR_RETURN(auto graph, CreateGraph(node_def, args, result_dtypes));

    const ConfigProto* config = ctx->function_library()->config_proto();
    // TODO(b/171039585): Support tf.VarIsInitializedOp using MLIR.
    bool use_mlir = config &&
                    GetMlirBridgeRolloutPolicy(
                        *graph, /*function_library=*/nullptr,
                        *config, /*uses_uninitialized_resource_args=*/
                        AnyUninitializedResourceArg(args)) ==
                        MlirBridgeRolloutPolicy::kEnabledByUser &&
                    node_def.op() != "VarIsInitializedOp";
    if (!use_mlir) {
      return compiler->CompileGraph(compile_options, node_def.name(),
                                    std::move(graph), args, result);
    }

    VLOG(1) << "Using MLIR bridge";
    GraphDebugInfo debug_info;
    std::vector<std::string> control_rets;
    if (result_dtypes.empty()) {
      control_rets.push_back(node_def.name());
    }
    return CompileGraphToXlaHlo(
        *graph, mlir::SpanToArrayRef<XlaCompiler::Argument>(args), control_rets,
        options.device_type.type_string(), compile_options.use_tuple_arg,
        *options.flib_def, debug_info, options.shape_representation_fn, result);
  };
  return CompileImpl(options, name, args, compile_op,
                     /*compile_threshold=*/absl::nullopt,
                     out_compilation_result, out_executable);
}

namespace {
// Print something that users can search for to definitively ascertain that XLA
// was used for their TF model.
//
// Prints only once to avoid spamming LOG(INFO).
void LogOnceXlaCompiledFirstCluster() {
  static absl::once_flag log_once;
  absl::call_once(log_once, [] {
    LOG(INFO) << "Compiled cluster using XLA!  This line is logged at most "
                 "once for the lifetime of the process.";
  });
}
}  // namespace

Status XlaCompilationCache::CompileImpl(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const std::function<Status(XlaCompiler* compiler,
                               XlaCompiler::CompilationResult*)>& compile_fn,
    absl::optional<int64> compile_threshold,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  if (FailOnXlaCompilation()) {
    return errors::Internal("XLA compilation disabled");
  }

  DCHECK_NE(out_executable, nullptr);
  VLOG(2) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << args.size();
    for (int i = 0, end = args.size(); i < end; i++) {
      VLOG(3) << i << ": " << args[i].HumanString();
    }
  }

  TF_ASSIGN_OR_RETURN(Signature signature, BuildSignature(function, args));
  VLOG(2) << "Signature: " << signature.HumanString();

  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      e.reset(new Entry);
    }
    entry = e.get();
  }

  // We always compile a cluster the very first time it is executed.  This is an
  // optimistic guess that pays off for statically shaped TensorFlow graphs
  // (since they get the benefit of XLA right away without waiting for warmup)
  // and doesn't hurt much for dynamically shaped TensorFlow graphs (we "pay" at
  // most one cluster-compilation's worth of compile time).
  bool is_first_execution;

  // We avoid compiling clusters that have "gone megamorphic" i.e. have an
  // excessive amount of shape dynamism.
  bool is_megamorphic;

  {
    mutex_lock lock(cluster_compile_stats_mu_);
    auto it =
        cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{})
            .first;
    is_first_execution = it->second.execution_count++ == 0;

    // The is_megamorphic bit is "sticky".  We assume clusters that have been
    // observed to be megamorphic once stay megamorphic forever.
    if (!it->second.is_megamorphic &&
        ShouldBeMegamorphic(/*compile_count=*/it->second.compile_count,
                            /*execution_count=*/it->second.execution_count)) {
      VLOG(1) << "Marking " << function.name()
              << " as megamorphic, compile_count=" << it->second.compile_count
              << " execution_count=" << it->second.execution_count;
      it->second.is_megamorphic = true;
    }

    is_megamorphic = it->second.is_megamorphic;
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  int64 current_request_count = ++entry->request_count;
  VLOG(2) << "Compilation cache entry hit: " << entry->compiled
          << " signature: " << signature.HumanString() << " with request count "
          << current_request_count << " and compile threshold "
          << compile_threshold.value_or(0);
  if (!entry->compiled) {
    XLA_SCOPED_LOGGING_TIMER("Compilation of XLA executable");
    const bool should_compile = [&] {
      if (!compile_threshold.has_value()) {
        // Lazy compilation is disabled.
        return true;
      }

      if (is_megamorphic) {
        BroadcastOptimizationRemark(XlaOptimizationRemark::MEGAMORPHIC_FUNCTION,
                                    function.name())
            .IgnoreError();
        VLOG(3) << "Not compiling cluster " << function.name()
                << " because it is megamorphic.";
        return false;
      }

      if (is_first_execution) {
        return true;
      }

      bool reached_compile_threshold =
          current_request_count >= *compile_threshold;
      if (!reached_compile_threshold) {
        VLOG(3)
            << "Not compiling cluster " << function.name()
            << " because it has not reached compile threshold; threshold is "
            << *compile_threshold << " execution count "
            << current_request_count << ".";
      }
      return reached_compile_threshold;
    }();

    if (!should_compile) {
      VLOG(2) << "Not compiling for signature: " << signature.HumanString();
      *out_compilation_result = nullptr;
      *out_executable = nullptr;
      return Status::OK();
    }

    tensorflow::Env* env = tensorflow::Env::Default();
    const uint64 compile_start_us = env->NowMicros();
    // Do the actual JIT compilation without holding the lock (it can take
    // a long time.)

    XlaCompiler compiler(options);
    entry->compiled = true;

    entry->compilation_status =
        compile_fn(&compiler, &entry->compilation_result);
    TF_RETURN_IF_ERROR(entry->compilation_status);
    CHECK_EQ(entry->executable.get(), nullptr);
    entry->compilation_status =
        BuildExecutable(options, entry->compilation_result, &entry->executable);

    const uint64 compile_end_us = env->NowMicros();
    const uint64 compile_time_us = compile_end_us - compile_start_us;
    metrics::UpdateXlaCompilationTime(compile_time_us);
    {
      mutex_lock lock(cluster_compile_stats_mu_);
      auto it = cluster_compile_stats_.find(function.name());
      it->second.compile_count++;
      it->second.cumulative_compile_time_us += compile_time_us;
      LogOnceXlaCompiledFirstCluster();
      VLOG(1) << "compiled " << function.name() << " "
              << it->second.compile_count
              << " times, compile time: " << compile_time_us
              << " us, cumulative: " << it->second.cumulative_compile_time_us
              << " us ("
              << tensorflow::strings::HumanReadableElapsedTime(compile_time_us /
                                                               1.0e6)
              << " / "
              << tensorflow::strings::HumanReadableElapsedTime(
                     it->second.cumulative_compile_time_us / 1.0e6)
              << ")";

      XlaJitCompilationActivity jit_compilation_activity;
      jit_compilation_activity.set_cluster_name(function.name());
      jit_compilation_activity.set_compile_count(it->second.compile_count);
      jit_compilation_activity.set_compile_time_us(compile_time_us);
      jit_compilation_activity.set_cumulative_compile_time_us(
          it->second.cumulative_compile_time_us);

      TF_RETURN_IF_ERROR(
          BroadcastXlaActivity(std::move(jit_compilation_activity)));
    }
  }
  TF_RETURN_IF_ERROR(entry->compilation_status);
  *out_compilation_result = &entry->compilation_result;
  *out_executable = entry->executable.get();
  return Status::OK();
}

}  // namespace tensorflow
