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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_H_

#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compilation_cache.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/device_executable_persistor.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/tf_graph_to_hlo_compiler.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/local_client.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Compiles/lowers a given Tensorflow graph/function/cluster into a compiled XLA
// compilation (HLO) using the XlaCompiler and compiles the resulting
// XlaCompilationResult into an `ExecutableType` (eg. xla::LocalExecutable) by
// calling `ClientType` (eg. xla::LocalClient).
//
// Caches the compiled XlaCompilationResult and Executable using a
// DeviceCompilationCache. Compilation is done only when there's a cache miss.
//
// Uses the DeviceExecutablePersistor class for persistence and tries to load a
// serialized executable from disk upon a request for compilation. If the
// appropriate executable isn't found on disk, compiles the given Tensorflow
// function/graph/cluster into an XlaCompilationResult (HLO) and
// `ExecutableType` and tries saving/persisting the compiled HLO and executable
// to disk.
//
// Since XLA computations must have static shapes, DeviceCompiler generates a
// new XLA computation for each new set of input shapes.
// TODO(b/255826209): De-templatize once we've moved to Device API completely.
template <typename ExecutableType, typename ClientType>
class DeviceCompiler : public ResourceBase {
 public:
  DeviceCompiler(
      std::unique_ptr<DeviceExecutablePersistor<ExecutableType, ClientType>>
          persistor,
      std::unique_ptr<DeviceCompilerClient<ExecutableType, ClientType>>
          compiler_client);
  ~DeviceCompiler() override;

  enum class CompileScope {
    kOp,
    kFunction,
  };

  // Compiles a function into a XlaCompiler::CompilationResult that can be used
  // to execute an XLA Computation. Compilation results are cached. Compilation
  // is skipped if there is a cache hit. `function` is the name of a Tensorflow
  // function to compile. `args` is a description of the arguments to the
  // computation.
  //
  // `compile_mode` controls the behavior of the compilation cache on a cache
  // miss.  If `compile_mode` is `kLazy` then, based on some profitability
  // heuristics, the compilation cache may decide not to compile the cluster at
  // this time.  In this case it returns null into both `out_compilation_result`
  // and `out_executable`.  If `compile_mode` is `kStrict` then the compilation
  // cache always attempts the compilation on a cache miss. If compilation mode
  // is 'kAsync' compilation of the cluster happens in the background while the
  // fallback path executes.
  //
  // The result of compilation is written to `*out_compilation_result`, which
  // must be non-null. If `out_executable` is non-null, also builds an
  // `ExecutableType` and sets `out_executable` to point to it. The
  // resulting executable pointer may be null if the computation has no
  // non-constant outputs.
  Status CompileIfNeeded(
      const XlaCompiler::Options& options, const NameAttrList& function,
      const std::vector<XlaCompiler::Argument>& args,
      const XlaCompiler::CompileOptions& compile_options,
      DeviceCompileMode compile_mode, DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      ExecutableType** out_executable);

  // As above, but for a single op.
  Status CompileSingleOpIfNeeded(
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const XlaCompiler::CompileOptions& compile_options, OpKernelContext* ctx,
      DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      ExecutableType** out_executable);

  ClientType* client() const { return compiler_client_->client(); }
  const DeviceType& device_type() const { return persistor_->device_type(); }
  DeviceCompilationCache<ExecutableType>* cache() { return cache_.get(); }
  DeviceExecutablePersistor<ExecutableType, ClientType>* persistor() {
    return persistor_.get();
  }
  DeviceCompilerClient<ExecutableType, ClientType>* compiler_client() {
    return compiler_client_.get();
  }

  string DebugString() const override;

 private:
  // Common implementation of Compile and CompileSingleOp. The `OpKernelContext`
  // parameter is always null for the former.
  Status CompileImpl(
      const XlaCompiler::CompileOptions& compile_options,
      const XlaCompiler::Options& options, const NameAttrList& function,
      const std::vector<XlaCompiler::Argument>& args, CompileScope scope,
      DeviceCompileMode compile_mode, OpKernelContext* ctx,
      DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      ExecutableType** out_executable);

  StatusOr<typename DeviceCompilationCache<ExecutableType>::Value>
  CompileStrict(
      const DeviceCompilationClusterSignature& sig,
      const XlaCompiler::CompileOptions& compile_options,
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const NameAttrList& function,
      typename DeviceCompilationCache<ExecutableType>::Value cache_value,
      CompileScope scope, OpKernelContext* ctx,
      DeviceCompilationProfiler* profiler, mutex* mu)
      TF_EXCLUSIVE_LOCKS_REQUIRED(*mu);

  Status CompileAsynchronous(const DeviceCompilationClusterSignature& sig,
                             const XlaCompiler::CompileOptions& compile_options,
                             const XlaCompiler::Options& options,
                             const std::vector<XlaCompiler::Argument>& args,
                             const NameAttrList& function, CompileScope scope,
                             OpKernelContext* ctx,
                             DeviceCompilationProfiler* profiler);

  std::unique_ptr<DeviceExecutablePersistor<ExecutableType, ClientType>>
      persistor_;
  std::unique_ptr<DeviceCompilerClient<ExecutableType, ClientType>>
      compiler_client_;
  std::unique_ptr<DeviceCompilationCache<ExecutableType>> cache_;

  // Pool of threads for asynchronous compilations.
  std::unique_ptr<thread::ThreadPool> async_compiler_threads_;

  mutex cluster_mutexes_mu_;
  absl::flat_hash_map<DeviceCompilationClusterSignature, std::unique_ptr<mutex>,
                      DeviceCompilationClusterSignature::Hash>
      cluster_mutexes_ TF_GUARDED_BY(cluster_mutexes_mu_);

  DeviceCompiler(const DeviceCompiler&) = delete;
  void operator=(const DeviceCompiler&) = delete;
};

namespace device_compiler_internal {
// Print something that users can search for to definitively ascertain that XLA
// was used for their TF model.
// Prints only once to avoid spamming LOG(INFO).
inline void LogOnceXlaCompiledFirstCluster() {
  static absl::once_flag log_once;
  absl::call_once(log_once, [] {
    LOG(INFO) << "Compiled cluster using XLA!  This line is logged at most "
                 "once for the lifetime of the process.";
  });
}

template <typename ExecutableType>
inline Status EligibleToPersist(DeviceCompileState compile_state,
                                const ExecutableType* executable) {
  if (compile_state != DeviceCompileState::kCompiled) {
    return errors::FailedPrecondition(
        "Cache entry to serialize is not compiled.");
  }
  if (executable == nullptr) {
    return errors::FailedPrecondition(
        "LocalExecutable not found for cache entry to serialize.");
  }
  return absl::OkStatus();
}
}  // namespace device_compiler_internal

template <typename ExecutableType, typename ClientType>
DeviceCompiler<ExecutableType, ClientType>::DeviceCompiler(
    std::unique_ptr<DeviceExecutablePersistor<ExecutableType, ClientType>>
        persistor,
    std::unique_ptr<DeviceCompilerClient<ExecutableType, ClientType>>
        compiler_client)
    : persistor_(std::move(persistor)),
      compiler_client_(std::move(compiler_client)) {
  cache_ = std::make_unique<DeviceCompilationCache<ExecutableType>>();
  async_compiler_threads_ = std::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "async_compiler_threads",
      kNumAsyncDeviceCompilerThreads);
}

template <typename ExecutableType, typename ClientType>
DeviceCompiler<ExecutableType, ClientType>::~DeviceCompiler() {
  // Since programs are owned by the cache, ensure any use of our programs have
  // completed by waiting for all stream executors to complete.
  compiler_client_->WaitForProgramsToFinish();
  // Wait for all outstanding compilations to finish.
  // Resetting the pointer explicitly in the top level destructor.
  // Without this, the pointer would be reset when the AsyncCompilationState
  // is destructed, which is dependent on the order of the members in the
  // DeviceCompiler class, which is error prone if the order changes.
  async_compiler_threads_.reset();
  // TODO(b/110813685): Think about the program ownership model. Programs are
  // currently owned by the compilation cache which means we must wait for
  // program completion in the destructor. There are multiple compilation caches
  // around, which complicates things a little. Perhaps having programs be
  // shared_ptrs (an invasive change) would make the model easier to reason
  // about?
}

template <typename ExecutableType, typename ClientType>
string DeviceCompiler<ExecutableType, ClientType>::DebugString() const {
  return "DeviceCompiler";
}

template <typename ExecutableType, typename ClientType>
Status DeviceCompiler<ExecutableType, ClientType>::CompileIfNeeded(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    const XlaCompiler::CompileOptions& compile_options,
    DeviceCompileMode compile_mode, DeviceCompilationProfiler* profiler,
    const XlaCompiler::CompilationResult** out_compilation_result,
    ExecutableType** out_executable) {
  return CompileImpl(compile_options, options, function, args,
                     CompileScope::kFunction, compile_mode, /*ctx=*/nullptr,
                     profiler, out_compilation_result, out_executable);
}

template <typename ExecutableType, typename ClientType>
Status DeviceCompiler<ExecutableType, ClientType>::CompileSingleOpIfNeeded(
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args,
    const XlaCompiler::CompileOptions& compile_options, OpKernelContext* ctx,
    DeviceCompilationProfiler* profiler,
    const XlaCompiler::CompilationResult** out_compilation_result,
    ExecutableType** out_executable) {
  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  // Remove the "_class" attribute from the attribute set used to create the
  // compilation cache key. This attribute is information for the colocator
  // and causes false uniqueness between nodes.
  name.mutable_attr()->erase("_class");
  return CompileImpl(compile_options, options, name, args, CompileScope::kOp,
                     DeviceCompileMode::kStrict, ctx, profiler,
                     out_compilation_result, out_executable);
}

template <typename ExecutableType, typename ClientType>
StatusOr<typename DeviceCompilationCache<ExecutableType>::Value>
DeviceCompiler<ExecutableType, ClientType>::CompileStrict(
    const DeviceCompilationClusterSignature& sig,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args,
    const NameAttrList& function,
    typename DeviceCompilationCache<ExecutableType>::Value cache_value,
    CompileScope scope, OpKernelContext* ctx,
    DeviceCompilationProfiler* profiler, mutex* mu) {
  tensorflow::Env* env = tensorflow::Env::Default();
  const uint64 compile_start_us = env->NowMicros();

  TfGraphToHloCompiler compiler(options);
  cache_value.compile_state = DeviceCompileState::kCompiled;

  std::unique_ptr<ExecutableType> out_executable;
  auto out_compilation_result =
      std::make_unique<XlaCompiler::CompilationResult>();

  if (scope == CompileScope::kOp) {
    cache_value.compilation_status = compiler.CompileSingleOp(
        compile_options, ctx, args, out_compilation_result.get());
  } else {
    CHECK(scope == CompileScope::kFunction);  // Crash OK
    cache_value.compilation_status = compiler.Compile(
        compile_options, function, args, out_compilation_result.get());
  }
  TF_RETURN_IF_ERROR(cache_value.compilation_status);
  TF_RET_CHECK(cache_value.executable == nullptr);
  TF_RET_CHECK(out_compilation_result->computation != nullptr);

  auto loaded_executable = persistor_->TryToLoadExecutable(
      DeviceCompilationClusterSignature::Hash()(sig), sig.HumanString(),
      options, *out_compilation_result, compiler_client_.get());

  if (loaded_executable.has_value()) {
    cache_value.compilation_status = loaded_executable->status();
    if (loaded_executable->ok()) {
      out_executable = *std::move(*loaded_executable);
      metrics::UpdatePersistentCacheLoadCount();
    }
  } else {
    auto built_executable =
        compiler_client_->BuildExecutable(options, *out_compilation_result);
    TF_RETURN_IF_ERROR(built_executable.status());
    out_executable = *std::move(built_executable);

    TF_RETURN_IF_ERROR(
        device_compiler_internal::EligibleToPersist<ExecutableType>(
            cache_value.compile_state, out_executable.get()));
    TF_RETURN_IF_ERROR(persistor_->TryToPersistExecutable(
        DeviceCompilationClusterSignature::Hash()(sig), sig.HumanString(),
        options, *out_compilation_result, *out_executable,
        compiler_client_.get()));
  }

  cache_value.compilation_result = out_compilation_result.get();
  cache_value.executable = out_executable.get();
  cache_->Store(sig, cache_value.compile_state, cache_value.compilation_status,
                std::move(out_compilation_result), std::move(out_executable));

  const uint64 compile_end_us = env->NowMicros();
  const uint64 compile_time_us = compile_end_us - compile_start_us;

  device_compiler_internal::LogOnceXlaCompiledFirstCluster();
  TF_RETURN_IF_ERROR(profiler->RegisterCompilation(
      function, compile_time_us, loaded_executable.has_value()));
  return cache_value;
}

template <typename ExecutableType, typename ClientType>
Status DeviceCompiler<ExecutableType, ClientType>::CompileAsynchronous(
    const DeviceCompilationClusterSignature& signature,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args,
    const NameAttrList& function, CompileScope scope, OpKernelContext* ctx,
    DeviceCompilationProfiler* profiler) {
  // Explicitly capture all required data by value for async compilation.
  // Update compilation state in cache.
  cache_->Store(signature, DeviceCompileState::kCompiling, std::nullopt,
                std::nullopt, std::nullopt);
  profiler->IncrementOngoingAsyncCompilations();
  // Don't move the above code into the thread function as it synchronously
  // updates the async compilation state!

  // When the ThreadPool for the compilation cache is destroyed, it waits for
  // compilations to have finished. This means that both 'entry' and 'this' will
  // be alive for the duration of the compilation.
  // !!Pay attention when additional variables must be captured by this lambda!!
  // All values are captured by value. Make sure that all pointer values (like
  // entry) do not get freed until the lambda has finished.
  const std::string& function_name = function.name();
  async_compiler_threads_->Schedule([=] {
    VLOG(2) << "Starting asynchronous compilation of cluster " << function_name
            << '.';
    // We don't need to lock mu, but do it anyway to satisfy thread safety
    // analysis.
    mutex mu;
    mutex_lock lock(mu);
    auto cache_value = typename DeviceCompilationCache<ExecutableType>::Value();
    auto s = CompileStrict(signature, compile_options, options, args, function,
                           cache_value, scope, ctx, profiler, &mu);
    VLOG(2) << "Finished asynchronous compililation of cluster "
            << function_name << '.';
    profiler->DecrementOngoingAsyncCompilations();
    // Update compilation status in cache.
    if (!s.ok()) {
      cache_->Store(signature, std::nullopt, s.status(), std::nullopt,
                    std::nullopt);
    }
  });
  return absl::OkStatus();
}

template <typename ExecutableType, typename ClientType>
Status DeviceCompiler<ExecutableType, ClientType>::CompileImpl(
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args, CompileScope scope,
    DeviceCompileMode compile_mode, OpKernelContext* ctx,
    DeviceCompilationProfiler* profiler,
    const XlaCompiler::CompilationResult** out_compilation_result,
    ExecutableType** out_executable) {
  DCHECK_NE(out_executable, nullptr);
  VLOG(2) << "DeviceCompiler::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << args.size();
    for (int i = 0, end = args.size(); i < end; i++) {
      VLOG(3) << i << ": " << args[i].HumanString();
    }
  }
  TF_ASSIGN_OR_RETURN(auto signature,
                      DeviceCompilationClusterSignature::Build(function, args));

  // The outer lock protects the existence of the mutex in the map.
  mutex* cluster_mutex;
  {
    mutex_lock lock(cluster_mutexes_mu_);
    auto it =
        cluster_mutexes_.emplace(signature, std::make_unique<mutex>()).first;
    cluster_mutex = it->second.get();
  }

  profiler->RegisterExecution(function);

  string human_signature;
  if (VLOG_IS_ON(2)) {
    human_signature = VLOG_IS_ON(3) ? signature.HumanString() : function.name();
    VLOG(2) << "DeviceCompilationClusterSignature: " << human_signature;
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock cluster_compile_lock(*cluster_mutex);
  auto cache_value = cache_->LookupOrCreate(signature);

  int64_t current_request_count = cache_value.request_count;
  VLOG(2) << "Compilation cache entry hit: "
          << static_cast<int>(cache_value.compile_state)
          << " signature: " << human_signature << " with request count "
          << current_request_count;

  DeviceCompileState state = cache_value.compile_state;
  *out_compilation_result = nullptr;
  *out_executable = nullptr;

  // Check if the requested entry is uncompiled and return an error if
  // compilation is disabled. This will raise an error for kLazy even if we have
  // not yet hit the compilation threshold and no compilation happens this
  // round. This is to avoid non-determanism of when compilation is disallowed,
  // for example by changing the threshold.
  if (state == DeviceCompileState::kUncompiled && FailOnXlaCompilation()) {
    VLOG(1) << "XLA compilation disabled: " << function.name() << "\n"
            << absl::StrJoin(
                   args, "\n",
                   [](std::string* out, const XlaCompiler::Argument& arg) {
                     absl::StrAppend(out, " arg: ", arg.HumanString());
                   });

    return errors::Internal("XLA compilation disabled");
  }

  if (state == DeviceCompileState::kUncompiled) {
    XLA_SCOPED_LOGGING_TIMER("Compilation of XLA executable");
    if (options.stream_id > 0) {
      VLOG(2) << "Not compiling for stream group " << options.stream_id;
      return absl::OkStatus();
    }
    if (!profiler->ShouldCompileCluster(function, compile_mode,
                                        current_request_count)) {
      VLOG(2) << "Not compiling for signature: " << human_signature;
      return absl::OkStatus();
    } else if (compile_mode == DeviceCompileMode::kAsync) {
      VLOG(2) << "Queueing asynchronous compilation for signature: "
              << human_signature;
      TF_RETURN_IF_ERROR(CompileAsynchronous(signature, compile_options,
                                             options, args, function, scope,
                                             ctx, profiler));
      return absl::OkStatus();
    } else {
      VLOG(2) << "Instantly compiling for signature: " << human_signature;
      TF_ASSIGN_OR_RETURN(
          cache_value,
          CompileStrict(signature, compile_options, options, args, function,
                        cache_value, scope, ctx, profiler, cluster_mutex));
    }
  } else if (state == DeviceCompileState::kCompiling) {
    VLOG(2) << "Ongoing asynchronous compilation for signature: "
            << human_signature;
    return absl::OkStatus();
  } else if (state == DeviceCompileState::kCompiled) {
    VLOG(2) << "Already Compiled for signature: " << human_signature;
  }

  TF_RETURN_IF_ERROR(cache_value.compilation_status);
  *out_compilation_result = cache_value.compilation_result;
  *out_executable = cache_value.executable;
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILER_H_
