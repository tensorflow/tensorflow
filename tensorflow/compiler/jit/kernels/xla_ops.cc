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

#include "tensorflow/compiler/jit/kernels/xla_ops.h"

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/stream_executor_util.h"

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

namespace tensorflow {

namespace {


// A closure describing how to run a compiled version of a TensorFlow function.
//
// It may seem unusual to stick the resource variable snapshots in this class.
// This is necessary: we need to use the snapshots observed by the compiler as
// the initial values for the resource variables (and cannot snapshot them again
// during execution) because otherwise we risk observing a different snapshot
// with shapes different from what we compiled for.
class XlaExecutableClosure {
 public:
  explicit XlaExecutableClosure(
      xla::LocalClient* client, xla::LocalExecutable* executable,
      const XlaCompiler::CompilationResult* compilation_result,
      ResourceVarsSnapshot resource_var_snapshots, int num_constant_args)
      : client_(client),
        executable_(executable),
        compilation_result_(compilation_result),
        resource_var_snapshots_(std::move(resource_var_snapshots)),
        num_constant_args_(num_constant_args) {}

  XlaExecutableClosure(XlaExecutableClosure&&) = default;
  XlaExecutableClosure& operator=(XlaExecutableClosure&&) = default;

  xla::LocalClient* client() const { return client_; }
  xla::LocalExecutable* executable() const { return executable_; }
  const XlaCompiler::CompilationResult* compilation_result() const {
    return compilation_result_;
  }
  const ResourceVarsSnapshot& resource_var_snapshots() const {
    return resource_var_snapshots_;
  }
  int num_constant_args() const { return num_constant_args_; }

 private:
  xla::LocalClient* client_;
  xla::LocalExecutable* executable_;
  const XlaCompiler::CompilationResult* compilation_result_;
  ResourceVarsSnapshot resource_var_snapshots_;
  int num_constant_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosure);
};

// This maintains a mapping from a globally unique ID to XlaExecutableClosure
// instances.
class XlaExecutableClosureStore {
 public:
  XlaExecutableClosureStore() : key_counter_(0) {}

  using KeyT = string;

  KeyT Produce(XlaExecutableClosure result) {
    mutex_lock l(mutex_);
    KeyT key = absl::StrCat(key_counter_++);
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  XlaExecutableClosure Consume(const KeyT& key) {
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    XlaExecutableClosure value = std::move(it->second);
    closures_.erase(it);
    return value;
  }

  static XlaExecutableClosureStore* Global() {
    static XlaExecutableClosureStore* instance = new XlaExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  int64 key_counter_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<KeyT, XlaExecutableClosure> closures_
      TF_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosureStore);
};

}  // namespace

XlaLocalLaunchBase::XlaLocalLaunchBase(OpKernelConstruction* ctx,
                                       const std::vector<int>& constants,
                                       const std::vector<int>& resources,
                                       const NameAttrList& function,
                                       bool has_ref_vars)
    : OpKernel(ctx),
      constants_(constants),
      resources_(resources),
      function_(function),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      has_ref_vars_(has_ref_vars) {}

static Status CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool has_ref_vars,
    const XlaPlatformInfo& platform_info,
    absl::Span<const Tensor* const> inputs,
    absl::Span<VariableInfo const> variable_infos,
    absl::Span<const int> constants, bool lazy, bool may_alias_resource_update,
    xla::LocalClient** client,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return errors::Internal("No resource manager.");
  }

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaCompilationCache>(
      rm->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache) {
        return BuildXlaCompilationCache(ctx->device(), platform_info, cache);
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  *client = static_cast<xla::LocalClient*>(cache->client());

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  XlaCompiler::Options options = GenerateCompilerOptions(
      *cache, *ctx->function_library(), ctx->device(),
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr,
      platform_info, has_ref_vars, &tf_allocator_adapter);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update = !has_ref_vars &&
                                          may_alias_resource_update;

  xla::StatusOr<std::vector<XlaCompiler::Argument>> args =
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constants, inputs, variable_infos,
          static_cast<Device*>(ctx->device()));
  TF_RETURN_IF_ERROR(args.status());
  return cache->Compile(options, function, *args, compile_options,
                        lazy ? XlaCompilationCache::CompileMode::kLazy
                             : XlaCompilationCache::CompileMode::kStrict,
                        compilation_result, executable);
}

void XlaLocalLaunchBase::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaLocalLaunchOpBase::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  xla::LocalClient* client;
  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;

  std::vector<VariableInfo> variable_infos;
  {
    OP_REQUIRES_OK(
        ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                        inputs, resources_, &variable_infos));
    OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variable_infos)));
    Status s = CompileToLocalExecutable(
        ctx, function_, /*has_ref_vars=*/has_ref_vars_, platform_info_, inputs,
        variable_infos, constants_, /*lazy=*/false,
        /*may_alias_resource_update=*/true, &client, &compilation_result,
        &executable);
    OP_REQUIRES_OK(ctx, s);
  }

  std::map<int, const Tensor*> resource_var_ptrs;
  for (int i = 0; i < resources_.size(); i++) {
    resource_var_ptrs[resources_[i]] = variable_infos[i].var()->tensor();
  }

  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  se::DeviceMemoryAllocator* allocator = GetAllocator(
      &tf_allocator_adapter, ctx->device(),
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr,
      platform_info_);
  int device_ordinal = stream ? stream->parent()->device_ordinal()
                              : client->default_device_ordinal();
  XlaComputationLaunchContext launch_context(
      client, allocator, device_ordinal,
      /*allocate_xla_tensors=*/platform_info_.is_on_xla_device(),
      platform_info_.UseMultipleStreams());
  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  xla::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
      launch_context.PopulateInputs(ctx, compilation_result, resource_var_ptrs,
                                    /*missing_ctx_input_prefix=*/0,
                                    input_output_alias);
  OP_REQUIRES_OK(ctx, execution_inputs.status());

  // Execute the computation.
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  xla::StatusOr<xla::ExecutionOutput> execution_output;
  if (!stream || platform_info_.platform_id() == se::host::kHostPlatformId) {
    execution_output =
        executable->Run(std::move(*execution_inputs), run_options);
  } else {
    execution_output =
        executable->RunAsync(std::move(*execution_inputs), run_options);
  }
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";
  OP_REQUIRES_OK(
      ctx, launch_context.PopulateOutputs(
               ctx, compilation_result, execution_output->ConsumeResult(),
               /*missing_ctx_input_prefix=*/0, absl::MakeSpan(variable_infos),
               input_output_alias, resource_var_ptrs));

  VLOG(1) << "Done";
}

namespace {
// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
std::vector<int> ConstantsVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
}

std::vector<int> ResourcesVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));

  DataTypeVector arg_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Targs", &arg_types));

  int num_resources;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Nresources", &num_resources));

  std::vector<int> resources(num_resources);
  std::iota(resources.begin(), resources.end(),
            constant_types.size() + arg_types.size());
  return resources;
}

NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr("function", &func));
  return *func;
}

bool MustCompileAttr(OpKernelConstruction* ctx) {
  bool must_compile;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr("must_compile", &must_compile));
  return must_compile;
}

bool HasRefVars(OpKernelConstruction* ctx) {
  bool has_ref_vars;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr(kXlaHasReferenceVarsAttr, &has_ref_vars));
  return has_ref_vars;
}

}  // namespace

XlaLocalLaunchOp::XlaLocalLaunchOp(OpKernelConstruction* ctx)
    : XlaLocalLaunchBase(ctx, ConstantsVector(ctx), ResourcesVector(ctx),
                         FunctionAttr(ctx), /*has_ref_vars=*/true) {}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

XlaCompileOp::XlaCompileOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      constants_(ConstantsVector(ctx)),
      resources_(ResourcesVector(ctx)),
      function_(FunctionAttr(ctx)),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      must_compile_(MustCompileAttr(ctx)),
      has_ref_vars_(HasRefVars(ctx)) {}

void XlaCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaCompileOp " << def().name()
          << (must_compile_ ? "(must-compile)" : "");
  xla::LocalClient* client;
  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;
  ResourceVarsSnapshot variables;

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  bool cannot_compile_cluster;
  {
    mutex_lock guard(cannot_compile_cluster_mu_);
    cannot_compile_cluster = cannot_compile_cluster_;
  }

  if (GetXlaOpsCommonFlags().tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    executable = nullptr;
  } else {
    std::vector<VariableInfo> variable_infos;
    OP_REQUIRES_OK(
        ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                        inputs, resources_, &variable_infos));
    OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variable_infos)));

    // Do not alias resource updates as locking variables in XlaCompile and
    // unlocking them in XlaRun may lead to deadlocks.
    Status status = CompileToLocalExecutable(
        ctx, function_, has_ref_vars_, platform_info_, inputs, variable_infos,
        constants_,
        /*lazy=*/!must_compile_,
        /*may_alias_resource_update=*/false, &client, &kernel, &executable);
    OP_REQUIRES_OK(ctx, SnapshotResourceVariables(ctx, resources_,
                                                  variable_infos, &variables));
    if (must_compile_ || status.code() != error::UNIMPLEMENTED) {
      OP_REQUIRES_OK(ctx, status);
    }

    if (status.code() == error::UNIMPLEMENTED) {
      LOG(WARNING) << "Compilation failed:" << status.ToString()
                   << ".  Falling back to TF function call.";

      BroadcastOptimizationRemark(
          XlaOptimizationRemark::UNIMPLEMENTED_OPERATION, status.ToString())
          .IgnoreError();
      executable = nullptr;
      mutex_lock guard(cannot_compile_cluster_mu_);
      cannot_compile_cluster_ = true;
    }
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  if (!executable) {
    DCHECK(!must_compile_);
    Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));

    Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
    compilation_successful.scalar<bool>()() = false;
    ctx->set_output(0, Tensor(cpu_allocator, DT_STRING, TensorShape({})));
    ctx->set_output(1, compilation_successful);
    return;
  }

  // Each execution of an XlaCompile op creates a new XlaExecutableClosure, even
  // if it didn't have to compile the cluster because of a compilation-cache
  // hit.  This is because we at least need new snapshots of the resource
  // variables.
  XlaExecutableClosureStore::KeyT key =
      XlaExecutableClosureStore::Global()->Produce(XlaExecutableClosure(
          client, executable, kernel, std::move(variables), constants_.size()));

  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  compilation_key.flat<tstring>()(0) = key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

XlaRunOp::XlaRunOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), platform_info_(XlaPlatformInfoFromDevice(ctx->device())) {}

void XlaRunOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaRunOp " << def().name();
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);
  const XlaExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);

  XlaExecutableClosure closure =
      XlaExecutableClosureStore::Global()->Consume(key);

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  se::DeviceMemoryAllocator* allocator = GetAllocator(
      &tf_allocator_adapter, ctx->device(),
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr,
      platform_info_);
  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
  int device_ordinal = stream ? stream->parent()->device_ordinal()
                              : closure.client()->default_device_ordinal();
  XlaComputationLaunchContext launch_context(
      closure.client(), allocator, device_ordinal,
      /*allocate_xla_tensors=*/platform_info_.is_on_xla_device(),
      /*use_multiple_streams=*/platform_info_.UseMultipleStreams());

  // We're missing the must-be-constant inputs, tell `PopulateInputs`
  // about this.  We don't actually need these inputs because they've
  // already been baked into the compiled kernel.
  const xla::HloInputOutputAliasConfig& input_output_alias =
      closure.executable()->executable()->module().input_output_alias_config();
  xla::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs;
  std::map<int, const Tensor*> snapshot_ptrs;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] {
          return absl::StrCat(
              "Populate Inputs (",
              closure.compilation_result()->xla_input_shapes.size(), ")");
        },
        tensorflow::profiler::TraceMeLevel::kInfo);

    for (auto& p : closure.resource_var_snapshots()) {
      snapshot_ptrs.emplace(p.first,
                            p.second.has_value() ? &p.second.value() : nullptr);
    }
    execution_inputs = launch_context.PopulateInputs(
        ctx, closure.compilation_result(), snapshot_ptrs,
        /*missing_ctx_input_prefix=*/closure.num_constant_args(),
        input_output_alias);
    OP_REQUIRES_OK(ctx, execution_inputs.status());
  }

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  xla::StatusOr<xla::ExecutionOutput> execution_output;
  if (!stream || platform_info_.platform_id() == se::host::kHostPlatformId) {
    execution_output =
        closure.executable()->Run(std::move(*execution_inputs), run_options);
  } else {
    execution_output = closure.executable()->RunAsync(
        std::move(*execution_inputs), run_options);
  }
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time in computation: " << elapsed << "us";


  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] {
        return absl::StrCat("Populate Outputs (", ctx->num_outputs(), ")");
      },
      tensorflow::profiler::TraceMeLevel::kInfo);

  xla::StatusOr<std::vector<VariableInfo>> variable_infos = GatherVariableInfo(
      ctx, *closure.compilation_result(), closure.num_constant_args());
  OP_REQUIRES_OK(ctx, variable_infos.status());
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*variable_infos)));
  OP_REQUIRES_OK(
      ctx,
      launch_context.PopulateOutputs(
          ctx, closure.compilation_result(), execution_output->ConsumeResult(),
          /*missing_ctx_input_prefix=*/closure.num_constant_args(),
          absl::MakeSpan(*variable_infos), input_output_alias, snapshot_ptrs));
}

XlaMergeOp::XlaMergeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaMergeOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaMergeOp " << def().name();
  int i = 0;
  if (ctx->has_input(i) || ctx->has_input(++i)) {
    ctx->set_output(0, ctx->input(i));
  }
}

REGISTER_KERNEL_BUILDER(Name("XlaLaunch").Device(DEVICE_CPU), XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("XlaLaunch")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("resources"),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("_XlaCompile").Device(DEVICE_CPU), XlaCompileOp);
REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_CPU), XlaRunOp);
REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_GPU).HostMemory("key"),
                        XlaRunOp);

REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_CPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_GPU), XlaMergeOp);

}  // namespace tensorflow
