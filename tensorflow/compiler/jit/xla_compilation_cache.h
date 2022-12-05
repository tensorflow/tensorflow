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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/device_executable_persistor.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {

// The XlaCompilationCache class caches the results of the XlaCompiler class,
// which converts a Tensorflow graph into a compiled XLA compilation.
//
// Since XLA computations must have static shapes, the cache generates a new
// XLA computation for each new set of input shapes.
//
// Currently no cache eviction policy is implemented and the cache grows without
// bound.
class XlaCompilationCache : public ResourceBase {
 public:
  XlaCompilationCache(
      std::unique_ptr<
          DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>>
          persistor,
      std::unique_ptr<
          DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient>>
          compiler_client);
  ~XlaCompilationCache() override;

  enum class CompileState { kUncompiled, kCompiling, kCompiled };

  enum class CompileScope {
    kOp,
    kFunction,
  };

  // Compiles a function into a XlaCompiler::CompilationResult that can be used
  // to execute an XLA Computation. Compilation results are cached.
  // `function` is the name of a Tensorflow function to compile.
  // `args` is a description of the arguments to the computation.
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
  // xla::LocalExecutable and sets `out_executable` to point to it. The
  // resulting executable pointer may be null if the computation has no
  // non-constant outputs.
  Status Compile(const XlaCompiler::Options& options,
                 const NameAttrList& function,
                 const std::vector<XlaCompiler::Argument>& args,
                 const XlaCompiler::CompileOptions& compile_options,
                 DeviceCompileMode compile_mode,
                 DeviceCompilationProfiler* profiler,
                 const XlaCompiler::CompilationResult** out_compilation_result,
                 xla::LocalExecutable** out_executable);

  // As above, but for a single op.
  Status CompileSingleOp(
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const XlaCompiler::CompileOptions& compile_options, OpKernelContext* ctx,
      DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      xla::LocalExecutable** out_executable);

  struct CompilationResultAndExecutable {
    const XlaCompiler::CompilationResult* compilation_result;
    xla::LocalExecutable* executable;
  };

  // Returns CompilationResultAndExecutable with non-null compilation_result and
  // executable if the signature is already compiled.
  // If the signature has not been compiled yet, this function returns a
  // CompilationResultAndExecutable instance with only nullptrs in it.
  // Non-ok status means something other than the 2 circumstances above
  // happened.
  StatusOr<CompilationResultAndExecutable>
  GetCompilationResultIfAlreadyCompiled(
      const NameAttrList& function,
      absl::Span<const XlaCompiler::Argument> args);

  xla::LocalClient* client() const { return compiler_client_->client(); }
  const DeviceType& device_type() const { return persistor_->device_type(); }

  string DebugString() const override;

  // Describes the types, shapes and any compile-time constant arguments
  // to a kernel. Key that uniquely identifies a compilation output.
  struct Signature {
    string name;

    // List of args (either as a TensorTypeAndShape or as a Tensor value)
    // for compile-time constant arguments to the compilation, ordered by
    // argument number. Tensors must be in host memory.
    using TensorTypeAndShape =
        std::pair<DataType, absl::InlinedVector<int64_t, 4>>;
    absl::InlinedVector<std::variant<Tensor, TensorTypeAndShape>, 8> args;

    bool operator==(const Signature& other) const;

    struct Hash {
      uint64 operator()(const Signature& signature) const;
    };

    // Returns a human-readable description of the signature.
    string HumanString() const;
  };

  // Builds the signature for a compilation.
  static StatusOr<Signature> BuildSignature(
      const NameAttrList& function,
      absl::Span<const XlaCompiler::Argument> args);

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
      xla::LocalExecutable** out_executable);

  // The value associated with a cache entry.
  struct Entry {
    mutex mu;

    // The current compilation state for this entry.
    CompileState compile_state = CompileState::kUncompiled;

    // The number of times a compilation with this signature has been requested.
    int64_t request_count = 0;

    // Did compilation succeed?
    Status compilation_status TF_GUARDED_BY(mu);

    // Output of the XlaCompiler.
    XlaCompiler::CompilationResult compilation_result TF_GUARDED_BY(mu);

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<xla::LocalExecutable> executable TF_GUARDED_BY(mu);
  };

  Status CompileStrict(const Signature& sig,
                       const XlaCompiler::CompileOptions& compile_options,
                       const XlaCompiler::Options& options,
                       const std::vector<XlaCompiler::Argument>& args,
                       const NameAttrList& function, CompileScope scope,
                       OpKernelContext* ctx,
                       DeviceCompilationProfiler* profiler, Entry* entry)
      TF_EXCLUSIVE_LOCKS_REQUIRED(entry->mu);
  Status CompileAsynchronous(const Signature& sig,
                             const XlaCompiler::CompileOptions& compile_options,
                             const XlaCompiler::Options& options,
                             const std::vector<XlaCompiler::Argument>& args,
                             const NameAttrList& function, CompileScope scope,
                             OpKernelContext* ctx,
                             DeviceCompilationProfiler* profiler, Entry* entry);

  mutex compile_cache_mu_;
  absl::flat_hash_map<Signature, std::unique_ptr<Entry>, Signature::Hash> cache_
      TF_GUARDED_BY(compile_cache_mu_);

  std::unique_ptr<
      DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>>
      persistor_;
  std::unique_ptr<DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient>>
      compiler_client_;
  // Pool of threads for asynchronous compilations.
  std::unique_ptr<thread::ThreadPool> async_compiler_threads_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompilationCache);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
