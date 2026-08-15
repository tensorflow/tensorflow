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

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_compile_util.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/thread_annotations.h"

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
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

auto* xla_launch_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/xla_launch_counter",
    "The number of times a XlaLaunch is called.", "device");

// A closure describing how to run a compiled version of a TensorFlow function.
//
// It may seem unusual to stick the resource variable snapshots in this class.
// This is necessary: we need to use the snapshots observed by the compiler as
// the initial values for the resource variables (and cannot snapshot them again
// during execution) because otherwise we risk observing a different snapshot
// with shapes different from what we compiled for.
template <typename ExecutableType, typename ClientType>
class ExecutableClosure {
 public:
  explicit ExecutableClosure(
      ClientType* client, ExecutableType* executable,
      const XlaCompiler::CompilationResult* compilation_result,
      ResourceVarsSnapshot resource_var_snapshots, int num_constant_args)
      : client_(client),
        executable_(executable),
        compilation_result_(compilation_result),
        resource_var_snapshots_(std::move(resource_var_snapshots)),
        num_constant_args_(num_constant_args) {}

  ExecutableClosure(ExecutableClosure&&) = default;
  ExecutableClosure& operator=(ExecutableClosure&&) = default;

  ClientType* client() const { return client_; }
  ExecutableType* executable() const { return executable_; }
  const XlaCompiler::CompilationResult* compilation_result() const {
    return compilation_result_;
  }
  const ResourceVarsSnapshot& resource_var_snapshots() const {
    return resource_var_snapshots_;
  }
  int num_constant_args() const { return num_constant_args_; }

 private:
  ClientType* client_;
  ExecutableType* executable_;
  const XlaCompiler::CompilationResult* compilation_result_;
  ResourceVarsSnapshot resource_var_snapshots_;
  int num_constant_args_;

  ExecutableClosure(const ExecutableClosure&) = delete;
  void operator=(const ExecutableClosure&) = delete;
};

// This maintains a mapping from a globally unique ID to ExecutableClosure
// instances.
template <typename ExecutableType, typename ClientType>
class ExecutableClosureStore {
 public:
  ExecutableClosureStore() : key_counter_(0) {}

  using KeyT = std::string;

  KeyT Produce(ExecutableClosure<ExecutableType, ClientType> result) {
    mutex_lock l(mutex_);
    KeyT key = absl::StrCat(key_counter_++);
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  ExecutableClosure<ExecutableType, ClientType> Consume(const KeyT& key) {
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    ExecutableClosure<ExecutableType, ClientType> value = std::move(it->second);
    closures_.erase(it);
    return value;
  }

  static ExecutableClosureStore* Global() {
    static ExecutableClosureStore* instance = new ExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  int64_t key_counter_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<KeyT, ExecutableClosure<ExecutableType, ClientType>>
      closures_ TF_GUARDED_BY(mutex_);

  ExecutableClosureStore(const ExecutableClosureStore&) = delete;
  void operator=(const ExecutableClosureStore&) = delete;
};

using PjRtExecutableClosure =
    ExecutableClosure<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtExecutableClosureStore =
    ExecutableClosureStore<xla::PjRtLoadedExecutable, xla::PjRtClient>;

absl::StatusOr<
    std::pair<std::vector<XlaCompiler::Argument>, ResourceVarsSnapshot>>
GetXlaCompilerArgsAndSnapshotVariables(
    absl::Span<const int> variable_indices,
    absl::Span<const int> must_be_constant_idxs,
    absl::Span<const Tensor* const> inputs, OpKernelContext* ctx) {
  std::pair<std::vector<XlaCompiler::Argument>, ResourceVarsSnapshot> result;

  std::vector<VariableInfo> variable_infos;
  TF_RETURN_IF_ERROR(
      GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(), inputs,
                                 variable_indices, &variable_infos));
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));

  TF_RETURN_IF_ERROR(SnapshotResourceVariables(ctx, variable_indices,
                                               variable_infos, &result.second));

  TF_ASSIGN_OR_RETURN(result.first,
                      XlaComputationLaunchContext::BuildXlaCompilerArguments(
                          must_be_constant_idxs, inputs, variable_infos,
                          static_cast<Device*>(ctx->device())));
  return result;
}

absl::Status GetUpdatedVariables(
    const OpKernelContext* ctx, absl::Span<const Tensor* const> inputs,
    absl::Span<const int> variable_indices,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<VariableInfo>* variable_infos) {
  std::set<int> variables_updated;
  for (const auto& resource_update : compilation_result.resource_updates) {
    if (resource_update.modified) {
      variables_updated.insert(resource_update.input_index);
    }
  }
  return GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                    inputs, variable_indices,
                                    &variables_updated, variable_infos);
}

// Get-or-create thread pool for a given collective.
static thread::ThreadPool* GetOrCreateThreadPoolForCollective(
    const XlaCompilationResult::CollectiveInfo& collective_info) {
  static absl::Mutex m(absl::kConstInit);
  static auto& thread_pool_cache ABSL_GUARDED_BY(m) =
      *new absl::node_hash_map<XlaCompilationResult::CollectiveInfo,
                               thread::ThreadPool>();
  absl::MutexLock l(m);
  auto it = thread_pool_cache.find(collective_info);
  if (it == thread_pool_cache.end()) {
    // Create & cache thread pool.
    auto inserted_it = thread_pool_cache.emplace(
        std::piecewise_construct, std::forward_as_tuple(collective_info),
        std::forward_as_tuple(Env::Default(), "xla_collective_thread_pool",
                              collective_info.group_size));
    return &inserted_it.first->second;
  }
  return &it->second;
}

void RunInThreadPoolIfCollectivesPresent(
    const XlaCompiler::CompilationResult& compilation_result,
    std::function<void()> execution_fn) {
  // If we are using collectives, we need to run in a separate threadpool.
  if (compilation_result.collective_info.has_value()) {
    GetOrCreateThreadPoolForCollective(*compilation_result.collective_info)
        ->Schedule(execution_fn);
  } else {
    // Otherwise, just run normally: we merely "pretend" to be asynchronous.
    execution_fn();
  }
}

}  // namespace

XlaLocalLaunchBase::XlaLocalLaunchBase(OpKernelConstruction* ctx,
                                       const std::vector<int>& constants,
                                       const std::vector<int>& resources,
                                       const NameAttrList& function,
                                       bool has_ref_vars)
    : AsyncOpKernel(ctx),
      constants_(constants),
      resources_(resources),
      function_(function),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      has_ref_vars_(has_ref_vars) {}

void XlaLocalLaunchBase::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  VLOG(1) << "XlaLocalLaunchOpBase::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  xla_launch_counter->GetCell(platform_info_.device_type().type_string())
      ->IncrementBy(1);

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<XlaCompiler::Argument> xla_compiler_args;
  const XlaCompiler::CompilationResult* compilation_result;

  xla::PjRtClient* pjrt_client;                // Not owned.
  xla::PjRtLoadedExecutable* pjrt_executable;  // Not owned.

  // Note that here we assume the shape of the variables don't change between
  // compilation and execution. The locks on the variables are released before
  // compilation so that we can achieve parallel compilation of different batch
  // sizes during warm-up.
  {
    // Creating a scope so that the locks on the variables are released when
    // variable_infos goes out of scope.
    std::vector<VariableInfo> variable_infos;
    std::set<int> variables_updated;
    // Here we only need to reader-lock the variables, so we pass an empty
    // variables_updated set here.
    absl::Status status = GetVariableInfosFromInputs(
        ctx->resource_manager(), ctx->device(), inputs, resources_,
        &variables_updated, &variable_infos);
    OP_REQUIRES_OK_ASYNC(ctx, status, done);
    status = LockVariables(absl::MakeSpan(variable_infos));
    OP_REQUIRES_OK_ASYNC(ctx, status, done);
    auto status_or_xla_compiler_args =
        XlaComputationLaunchContext::BuildXlaCompilerArguments(
            constants_, inputs, variable_infos,
            static_cast<Device*>(ctx->device()));
    OP_REQUIRES_OK_ASYNC(ctx, status_or_xla_compiler_args.status(), done);
    xla_compiler_args = std::move(status_or_xla_compiler_args.value());
  }

  VLOG(2) << "Compiling using PJRT";
  absl::Status status = CompileToPjRtLoadedExecutable(
      *ctx, platform_info_, function_, xla_compiler_args,
      DeviceCompileMode::kStrict, has_ref_vars_,
      /*may_alias_resource_update=*/true, &compilation_result, &pjrt_client,
      &pjrt_executable);
  OP_REQUIRES_OK_ASYNC(ctx, status, done);

  VLOG(2) << "Compiled using PJRT: " << status;
  VLOG(2) << "pjrt_executable != nullptr: " << (pjrt_executable != nullptr);
  VLOG(2) << "compilation_result != nullptr: "
          << (compilation_result != nullptr);
  VLOG(2) << "Executing using PJRT.";

  auto run_pjrt_cluster = [ctx, pjrt_client, pjrt_executable,
                           compilation_result, done, inputs,
                           resources = resources_]() {
    // Separate scope so that VariableInfo locks are released before done() is
    // called.
    {
      std::vector<VariableInfo> variable_infos;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          GetUpdatedVariables(ctx, inputs, resources, *compilation_result,
                              &variable_infos),
          done);
      OP_REQUIRES_OK_ASYNC(ctx, LockVariables(absl::MakeSpan(variable_infos)),
                           done);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          RunPjRtExecutable(inputs, variable_infos, *compilation_result,
                            pjrt_client, pjrt_executable, ctx),
          done);
    }
    VLOG(2) << "Done executing with PJRT.";
    done();
  };

  RunInThreadPoolIfCollectivesPresent(*compilation_result, run_pjrt_cluster);
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

std::vector<int> VectorAttr(OpKernelConstruction* ctx,
                            absl::string_view attr_name) {
  std::vector<int> vec;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr(attr_name, &vec));
  return vec;
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

class XlaLaunchV2Op : public XlaLocalLaunchBase {
 public:
  explicit XlaLaunchV2Op(OpKernelConstruction* ctx)
      : XlaLocalLaunchBase(ctx, VectorAttr(ctx, "constants"),
                           VectorAttr(ctx, "resources"), FunctionAttr(ctx),
                           /*has_ref_vars=*/true) {}
};

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
  const XlaCompiler::CompilationResult* kernel = nullptr;
  xla::PjRtClient* pjrt_client = nullptr;
  xla::PjRtLoadedExecutable* pjrt_executable = nullptr;
  ResourceVarsSnapshot variables_snapshot;

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  bool cannot_compile_cluster;
  {
    mutex_lock guard(cannot_compile_cluster_mu_);
    cannot_compile_cluster = cannot_compile_cluster_;
  }
  DeviceCompileMode compile_mode = [&] {
    if (must_compile_) {
      return DeviceCompileMode::kStrict;
    }
    return GetXlaOpsCommonFlags()->tf_xla_async_compilation
               ? DeviceCompileMode::kAsync
               : DeviceCompileMode::kLazy;
  }();

  if (GetXlaOpsCommonFlags()->tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    pjrt_executable = nullptr;
  } else {
    auto args_and_variables_snapshot = GetXlaCompilerArgsAndSnapshotVariables(
        resources_, constants_, inputs, ctx);
    OP_REQUIRES_OK(ctx, args_and_variables_snapshot.status());
    const std::vector<XlaCompiler::Argument>& args =
        args_and_variables_snapshot->first;
    variables_snapshot = std::move(args_and_variables_snapshot->second);

    // Do not alias resource updates as locking variables in XlaCompile and
    // unlocking them in XlaRun may lead to deadlocks.
    VLOG(2) << "Using PJRT for compilation. Function name: "
            << function_.name();
    absl::Status status = CompileToPjRtLoadedExecutable(
        *ctx, platform_info_, function_, args, compile_mode, has_ref_vars_,
        /*may_alias_resource_update=*/false, &kernel, &pjrt_client,
        &pjrt_executable);
    if (compile_mode != DeviceCompileMode::kLazy ||
        status.code() != error::UNIMPLEMENTED) {
      OP_REQUIRES_OK(ctx, status);
    }

    if (status.code() == error::UNIMPLEMENTED) {
      LOG(WARNING) << "Compilation failed:" << status
                   << ".  Falling back to TF function call.";

      BroadcastOptimizationRemark(
          XlaOptimizationRemark::UNIMPLEMENTED_OPERATION, status.ToString())
          .IgnoreError();
      pjrt_executable = nullptr;
      mutex_lock guard(cannot_compile_cluster_mu_);
      cannot_compile_cluster_ = true;
    }
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  // Async compilation returns nullptr executable without an error.
  if (!pjrt_executable) {
    DCHECK(!must_compile_);
    Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
    Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
    compilation_successful.scalar<bool>()() = false;
    ctx->set_output(0, compilation_key);
    ctx->set_output(1, compilation_successful);
    return;
  }

  // Each execution of an XlaCompile op creates a new ExecutableClosure, even
  // if it didn't have to compile the cluster because of a compilation-cache
  // hit.  This is because we at least need new snapshots of the resource
  // variables.
  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  PjRtExecutableClosureStore::KeyT key =
      PjRtExecutableClosureStore::Global()->Produce(PjRtExecutableClosure(
          pjrt_client, pjrt_executable, kernel, std::move(variables_snapshot),
          constants_.size()));
  compilation_key.flat<tstring>()(0) = key;
  VLOG(2) << "Compiled with PJRT. compilation_key: " << key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

XlaRunOp::XlaRunOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaRunOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaRunOp " << def().name();
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);

  const PjRtExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);
  PjRtExecutableClosure closure =
      PjRtExecutableClosureStore::Global()->Consume(key);

  // Fetch inputs from the OpKernelContext. Inputs are the same as the ones
  // for XlaCompile, except that the must-be-constant inputs that appear in
  // the beginning are stripped off and the closure key is appended as the
  // last input. So the inputs look like: input tensors, resource variables,
  // closure key tensor.
  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  absl::flat_hash_map<int, const Tensor*> variable_snapshots;
  for (const auto& [variable_index, variable_tensor] :
       closure.resource_var_snapshots()) {
    variable_snapshots.emplace(variable_index, variable_tensor.has_value()
                                                   ? &variable_tensor.value()
                                                   : nullptr);
  }

  absl::StatusOr<std::vector<VariableInfo>> updated_variables =
      GatherVariableInfo(ctx, *closure.compilation_result(),
                         closure.num_constant_args());
  OP_REQUIRES_OK(ctx, updated_variables.status());
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*updated_variables)));
  OP_REQUIRES_OK(
      ctx,
      RunPjRtExecutable(closure.num_constant_args(), inputs, variable_snapshots,
                        *updated_variables, *closure.compilation_result(),
                        closure.client(), closure.executable(), ctx));
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

REGISTER_KERNEL_BUILDER(Name("XlaLaunchV2").Device(DEVICE_CPU), XlaLaunchV2Op);

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

REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_CPU), XlaRunOp);
REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_GPU).HostMemory("key"),
                        XlaRunOp);
REGISTER_KERNEL_BUILDER(
    Name("_XlaRun").Device(DEVICE_DEFAULT).HostMemory("key"), XlaRunOp);

REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_CPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_GPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_DEFAULT), XlaMergeOp);

}  // namespace tensorflow
