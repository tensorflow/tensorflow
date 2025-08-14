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
#include <memory>
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
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_compile_util.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_host_recv_device_context.h"
#include "tensorflow/compiler/jit/xla_host_send_device_context.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/lib/traceme.h"

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
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
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

  using KeyT = string;

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

using XlaExecutableClosure =
    ExecutableClosure<xla::LocalExecutable, xla::LocalClient>;
using XlaExecutableClosureStore =
    ExecutableClosureStore<xla::LocalExecutable, xla::LocalClient>;
using PjRtExecutableClosure =
    ExecutableClosure<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtExecutableClosureStore =
    ExecutableClosureStore<xla::PjRtLoadedExecutable, xla::PjRtClient>;

se::Stream* GetStream(OpKernelContext* ctx) {
  return ctx->op_device_context() ? ctx->op_device_context()->stream()
                                  : nullptr;
}

XlaComputationLaunchContext GetLaunchContext(
    const XlaPlatformInfo& platform_info, OpKernelContext* ctx,
    xla::LocalClient* client, se::DeviceMemoryAllocator* allocator) {
  se::Stream* stream = GetStream(ctx);
  int device_ordinal = stream ? stream->parent()->device_ordinal()
                              : client->default_device_ordinal();
  XlaComputationLaunchContext launch_context(
      client, allocator, device_ordinal,
      /*allocate_xla_tensors=*/platform_info.is_on_xla_device(),
      /*use_multiple_streams=*/platform_info.UseMultipleStreams());
  return launch_context;
}

absl::Status GetTaskName(const absl::string_view device_name,
                         std::string* task_name) {
  string ignored;
  if (!DeviceNameUtils::SplitDeviceName(device_name, task_name, &ignored)) {
    return errors::InvalidArgument("Unable to parse device name: ",
                                   device_name);
  }

  return absl::OkStatus();
}

// Provide SendDeviceMemoryFunction for XLA host callbacks.  This callback
// handles transferring from device to host.
xla::SendDeviceMemoryFunction GetSendDeviceMemoryFunction(
    OpKernelContext* ctx, const std::string& program_key) {
  return
      [ctx, program_key](
          int64_t channel_id, se::Stream* stream, const xla::Shape& shape,
          const se::DeviceMemoryBase& device_memory_base,
          const absl::flat_hash_map<std::string, std::string>& frontend_attrs)
          -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
        auto iter = frontend_attrs.find("_xla_host_transfer_rendezvous");

        // Generate the Rendezvous key.
        const std::string& rendezvous_key_base =
            absl::StrCat(program_key, iter->second);

        const std::string& src_device = ctx->device()->name();
        std::string task_prefix;
        TF_RETURN_IF_ERROR(GetTaskName(src_device, &task_prefix));
        const std::string dst_device =
            absl::StrCat(task_prefix, "/device:CPU:0");
        const std::string& rendezvous_key =
            Rendezvous::CreateKey(src_device, /*src_incarnation=*/1, dst_device,
                                  rendezvous_key_base, FrameAndIter(0, 0));
        VLOG(2) << "Rendezvous Key for receiving at host: " << rendezvous_key;

        RendezvousInterface::ParsedKey parsed_key;
        TF_RETURN_IF_ERROR(Rendezvous::ParseKey(rendezvous_key, &parsed_key));

        TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
        tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
            tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
                std::move(event));

        Rendezvous::Args args;
        // Rendezvous::Args owns the device context pointer.
        args.device_context = new XlaHostRecvDeviceContext(
            stream, device_memory_base, shape, done_event);

        Tensor host_tensor;
        TF_RETURN_IF_ERROR(
            ctx->rendezvous()->Send(parsed_key, args, host_tensor, false));

        return std::move(done_event);
      };
}

// Provide RecvDeviceMemoryFunction for XLA host callbacks.  This callback
// handles transferring from host to device.
xla::RecvDeviceMemoryFunction GetRecvDeviceMemoryFunction(
    OpKernelContext* ctx, const std::string& program_key) {
  return
      [ctx, program_key](
          int64_t channel_id, se::Stream* stream, const xla::Shape& shape,
          se::DeviceMemoryBase* device_memory_base,
          const absl::flat_hash_map<std::string, std::string>& frontend_attrs)
          -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
        auto iter = frontend_attrs.find("_xla_host_transfer_rendezvous");

        // Generate the Rendezvous key.
        const std::string& rendezvous_key_base =
            absl::StrCat(program_key, iter->second);

        const std::string& dst_device = ctx->device()->name();
        std::string task_prefix;
        TF_RETURN_IF_ERROR(GetTaskName(dst_device, &task_prefix));
        const std::string src_device =
            absl::StrCat(task_prefix, "/device:CPU:0");
        const std::string& rendezvous_key =
            Rendezvous::CreateKey(src_device, /*src_incarnation=*/1, dst_device,
                                  rendezvous_key_base, FrameAndIter(0, 0));
        VLOG(2) << "Rendezvous Key for sending from host: " << rendezvous_key;

        RendezvousInterface::ParsedKey parsed_key;
        TF_RETURN_IF_ERROR(Rendezvous::ParseKey(rendezvous_key, &parsed_key));

        TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
        tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
            tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
                std::move(event));

        Rendezvous::Args args;
        // Rendezvous::Args owns the device context pointer.
        args.device_context = new XlaHostSendDeviceContext(
            stream, device_memory_base, shape, done_event);

        Tensor device_tensor;
        bool is_dead;
        TF_RETURN_IF_ERROR(ctx->rendezvous()->Recv(
            parsed_key, args, &device_tensor, /*is_dead=*/&is_dead));

        return std::move(done_event);
      };
}

absl::StatusOr<xla::ExecutionOutput> RunExecutable(
    const XlaPlatformInfo& platform_info,
    const XlaComputationLaunchContext& launch_context,
    std::vector<xla::ExecutionInput> execution_inputs,
    xla::ExecutableRunOptions run_options, xla::LocalExecutable* executable,
    OpKernelContext* ctx, se::DeviceMemoryAllocator* allocator) {
  VLOG(2) << "Executing Xla Computation.";
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  se::Stream* stream = GetStream(ctx);
  run_options.set_stream(GetStream(ctx));
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());

  absl::StatusOr<xla::ExecutionOutput> execution_output;
  bool run_synchronous =
      !stream || platform_info.platform_id() == se::host::kHostPlatformId;
  if (run_synchronous) {
    execution_output =
        executable->Run(std::move(execution_inputs), run_options);
  } else {
    execution_output =
        executable->RunAsync(std::move(execution_inputs), run_options);
  }

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time for Xla Executable Run: " << elapsed << "us";
  return execution_output;
}

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

absl::Status CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool has_ref_vars,
    const XlaPlatformInfo& platform_info,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool may_alias_resource_update,
    xla::LocalClient** client,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return absl::InternalError("No resource manager.");
  }

  TF_ASSIGN_OR_RETURN(DeviceType compilation_device_type,
                      GetCompilationDeviceType(platform_info.device_type()));

  XlaDeviceCompiler* xla_device_compiler;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaDeviceCompiler>(
      rm->default_container(), "xla_device_compiler", &xla_device_compiler,
      [&](XlaDeviceCompiler** xla_device_compiler) {
        return BuildXlaDeviceCompiler(ctx->device(), ctx->function_library(),
                                      platform_info, compilation_device_type,
                                      xla_device_compiler);
      }));
  DeviceCompilationProfiler* profiler;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<DeviceCompilationProfiler>(
      rm->default_container(), "device_compilation_profiler", &profiler,
      [](DeviceCompilationProfiler** profiler) {
        *profiler = new DeviceCompilationProfiler();
        return absl::OkStatus();
      }));
  // Hold the reference to the XLA device compiler and profiler during
  // evaluation. (We could probably free them sooner because the ResourceMgr
  // will retain references, but this is more obviously correct.)
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  *client = static_cast<xla::LocalClient*>(xla_device_compiler->client());

  XlaCompiler::Options options = GenerateCompilerOptions(
      *xla_device_compiler, *ctx->function_library(), ctx->device(),
      GetStream(ctx), platform_info, has_ref_vars);

  XlaCompiler::CompileOptions compile_options =
      GenerateCompileOptions(has_ref_vars, may_alias_resource_update);

  return xla_device_compiler->CompileIfNeeded(
      options, function, args, compile_options, compile_mode, profiler,
      compilation_result, executable);
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

  xla::LocalClient* client;          // Not owned.
  xla::LocalExecutable* executable;  // Not owned.

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

  bool use_pjrt = GetXlaOpsCommonFlags()
                      ->tf_xla_use_device_api.IsEnabledInXlaLaunchForDevice(
                          platform_info_.device_type());
  if (use_pjrt) {
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
    return;
  }

  absl::Status status = CompileToLocalExecutable(
      ctx, function_, /*has_ref_vars=*/has_ref_vars_, platform_info_,
      xla_compiler_args, DeviceCompileMode::kStrict,
      /*may_alias_resource_update=*/true, &client, &compilation_result,
      &executable);
  OP_REQUIRES_OK_ASYNC(ctx, status, done);

  // Continuation of the execution, may be run in a different thread.
  auto run_xla_cluster = [ctx, client, executable, compilation_result, done,
                          inputs, resources = resources_]() {
    // Separate scope so that VariableInfo locks are released before done is
    // called.
    {
      auto platform_info = XlaPlatformInfoFromDevice(ctx->device());
      std::vector<VariableInfo> variable_infos;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          GetUpdatedVariables(ctx, inputs, resources, *compilation_result,
                              &variable_infos),
          done);
      OP_REQUIRES_OK_ASYNC(ctx, LockVariables(absl::MakeSpan(variable_infos)),
                           done);
      absl::flat_hash_map<int, const Tensor*> resource_var_ptrs;
      for (int i = 0; i < resources.size(); i++) {
        resource_var_ptrs[resources[i]] = variable_infos[i].var()->tensor();
      }

      std::shared_ptr<se::DeviceMemoryAllocator> allocator =
          GetAllocator(ctx->device(), GetStream(ctx), platform_info);
      XlaComputationLaunchContext launch_context =
          GetLaunchContext(platform_info, ctx, client, allocator.get());

      const xla::HloInputOutputAliasConfig& input_output_alias =
          executable->executable()->module().input_output_alias_config();
      absl::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
          launch_context.PopulateInputs(
              ctx, compilation_result, resource_var_ptrs,
              /*missing_ctx_input_prefix=*/0, input_output_alias);
      OP_REQUIRES_OK_ASYNC(ctx, execution_inputs.status(), done);

      xla::gpu::GpuExecutableRunOptions gpu_options;
      xla::DeviceAssignment device_assignment;
      xla::ExecutableRunOptions run_options;
      if (compilation_result->collective_info.has_value()) {
        OP_REQUIRES_OK_ASYNC(ctx,
                             ResolveDeviceAssignment(
                                 ctx, *compilation_result->collective_info,
                                 run_options, device_assignment, gpu_options),
                             done);
      }

      // Hardcode run id to always be zero: TF distributed strategy
      // differentiates between subsequent runs using dependency edges. This
      // is safe, as only TF dist-strat can produce distributed ops, and we
      // can rely on TF dist-strat invariants.
      xla::RunId run_id(0);
      run_options.set_run_id(run_id);

      absl::StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
          platform_info, launch_context, std::move(*execution_inputs),
          run_options, executable, ctx, allocator.get());
      OP_REQUIRES_ASYNC(ctx, execution_output.ok(), execution_output.status(),
                        done);

      OP_REQUIRES_OK_ASYNC(
          ctx,
          launch_context.PopulateOutputs(
              ctx, compilation_result, execution_output->ConsumeResult(),
              /*missing_ctx_input_prefix=*/0, absl::MakeSpan(variable_infos),
              input_output_alias, resource_var_ptrs),
          done);
      VLOG(1) << "Done";
    }
    done();
  };

  RunInThreadPoolIfCollectivesPresent(*compilation_result, run_xla_cluster);
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
  xla::LocalClient* client = nullptr;
  xla::LocalExecutable* executable = nullptr;
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

  bool use_pjrt =
      GetXlaOpsCommonFlags()
          ->tf_xla_use_device_api.IsEnabledInXlaCompileAndRunForDevice(
              platform_info_.device_type());

  if (GetXlaOpsCommonFlags()->tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    executable = nullptr;
  } else {
    auto args_and_variables_snapshot = GetXlaCompilerArgsAndSnapshotVariables(
        resources_, constants_, inputs, ctx);
    OP_REQUIRES_OK(ctx, args_and_variables_snapshot.status());
    const std::vector<XlaCompiler::Argument>& args =
        args_and_variables_snapshot->first;
    variables_snapshot = std::move(args_and_variables_snapshot->second);

    // Do not alias resource updates as locking variables in XlaCompile and
    // unlocking them in XlaRun may lead to deadlocks.
    absl::Status status;
    if (use_pjrt) {
      VLOG(2) << "Using PJRT for compilation. Function name: "
              << function_.name();
      status = CompileToPjRtLoadedExecutable(
          *ctx, platform_info_, function_, args, compile_mode, has_ref_vars_,
          /*may_alias_resource_update=*/false, &kernel, &pjrt_client,
          &pjrt_executable);
    } else {
      status = CompileToLocalExecutable(
          ctx, function_, has_ref_vars_, platform_info_, args, compile_mode,
          /*may_alias_resource_update=*/false, &client, &kernel, &executable);
    }
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
      executable = nullptr;
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
  if (!executable && !pjrt_executable) {
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
  if (use_pjrt) {
    PjRtExecutableClosureStore::KeyT key =
        PjRtExecutableClosureStore::Global()->Produce(PjRtExecutableClosure(
            pjrt_client, pjrt_executable, kernel, std::move(variables_snapshot),
            constants_.size()));
    compilation_key.flat<tstring>()(0) = key;
    VLOG(2) << "Compiled with PJRT. compilation_key: " << key;
  } else {
    XlaExecutableClosureStore::KeyT key =
        XlaExecutableClosureStore::Global()->Produce(XlaExecutableClosure(
            client, executable, kernel, std::move(variables_snapshot),
            constants_.size()));
    compilation_key.flat<tstring>()(0) = key;
    VLOG(2) << "Compiled with XLA. compilation_key: " << key;
  }

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

  bool use_pjrt =
      GetXlaOpsCommonFlags()
          ->tf_xla_use_device_api.IsEnabledInXlaCompileAndRunForDevice(
              platform_info_.device_type());

  if (use_pjrt) {
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

    {
      absl::StatusOr<std::vector<VariableInfo>> updated_variables =
          GatherVariableInfo(ctx, *closure.compilation_result(),
                             closure.num_constant_args());
      OP_REQUIRES_OK(ctx, updated_variables.status());
      OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*updated_variables)));
      OP_REQUIRES_OK(
          ctx, RunPjRtExecutable(closure.num_constant_args(), inputs,
                                 variable_snapshots, *updated_variables,
                                 *closure.compilation_result(),
                                 closure.client(), closure.executable(), ctx));
    }

    OP_REQUIRES_OK(ctx, absl::OkStatus());
    return;
  }

  const XlaExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);

  XlaExecutableClosure closure =
      XlaExecutableClosureStore::Global()->Consume(key);
  std::shared_ptr<se::DeviceMemoryAllocator> allocator =
      GetAllocator(ctx->device(), GetStream(ctx), platform_info_);
  XlaComputationLaunchContext launch_context =
      GetLaunchContext(platform_info_, ctx, closure.client(), allocator.get());

  // We're missing the must-be-constant inputs, tell `PopulateInputs`
  // about this.  We don't actually need these inputs because they've
  // already been baked into the compiled kernel.
  const xla::HloInputOutputAliasConfig& input_output_alias =
      closure.executable()->executable()->module().input_output_alias_config();
  absl::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs;
  absl::flat_hash_map<int, const Tensor*> snapshot_ptrs;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] {
          return absl::StrCat(
              "Populate Inputs (",
              closure.compilation_result()->xla_input_shapes.size(), ")");
        },
        tsl::profiler::TraceMeLevel::kInfo);

    for (const auto& [variable_index, variable_tensor] :
         closure.resource_var_snapshots()) {
      snapshot_ptrs.emplace(variable_index, variable_tensor.has_value()
                                                ? &variable_tensor.value()
                                                : nullptr);
    }
    execution_inputs = launch_context.PopulateInputs(
        ctx, closure.compilation_result(), snapshot_ptrs,
        /*missing_ctx_input_prefix=*/closure.num_constant_args(),
        input_output_alias);
    OP_REQUIRES_OK(ctx, execution_inputs.status());
  }

  xla::ExecutableRunOptions run_options;

  // Host callbacks used for HLO send/recv.
  xla::SendDeviceMemoryFunction send_function =
      GetSendDeviceMemoryFunction(ctx, key);
  run_options.set_send_device_memory_function(&send_function);
  xla::RecvDeviceMemoryFunction recv_function =
      GetRecvDeviceMemoryFunction(ctx, key);
  run_options.set_recv_device_memory_function(&recv_function);

  absl::StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
      platform_info_, launch_context, std::move(*execution_inputs), run_options,
      closure.executable(), ctx, allocator.get());
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  tsl::profiler::TraceMe hlo_module_activity(
      [&] {
        return absl::StrCat("Populate Outputs (", ctx->num_outputs(), ")");
      },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::StatusOr<std::vector<VariableInfo>> variable_infos = GatherVariableInfo(
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
