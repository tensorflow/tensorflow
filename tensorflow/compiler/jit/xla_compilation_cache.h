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
#include "tensorflow/compiler/jit/device_compilation_cache.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
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
// TODO(b/255826209): Rename to DeviceCompiler and update comments to reflect
// functionality.
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
  // xla::LocalExecutable and sets `out_executable` to point to it. The
  // resulting executable pointer may be null if the computation has no
  // non-constant outputs.
  Status CompileIfNeeded(
      const XlaCompiler::Options& options, const NameAttrList& function,
      const std::vector<XlaCompiler::Argument>& args,
      const XlaCompiler::CompileOptions& compile_options,
      DeviceCompileMode compile_mode, DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      xla::LocalExecutable** out_executable);

  // As above, but for a single op.
  Status CompileSingleOpIfNeeded(
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const XlaCompiler::CompileOptions& compile_options, OpKernelContext* ctx,
      DeviceCompilationProfiler* profiler,
      const XlaCompiler::CompilationResult** out_compilation_result,
      xla::LocalExecutable** out_executable);

  xla::LocalClient* client() const { return compiler_client_->client(); }
  const DeviceType& device_type() const { return persistor_->device_type(); }
  DeviceCompilationCache<xla::LocalExecutable>* cache() { return cache_.get(); }

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
      xla::LocalExecutable** out_executable);

  StatusOr<DeviceCompilationCache<xla::LocalExecutable>::Value> CompileStrict(
      const DeviceCompilationClusterSignature& sig,
      const XlaCompiler::CompileOptions& compile_options,
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const NameAttrList& function,
      DeviceCompilationCache<xla::LocalExecutable>::Value cache_value,
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

  std::unique_ptr<
      DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>>
      persistor_;
  std::unique_ptr<DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient>>
      compiler_client_;
  std::unique_ptr<DeviceCompilationCache<xla::LocalExecutable>> cache_;

  // Pool of threads for asynchronous compilations.
  std::unique_ptr<thread::ThreadPool> async_compiler_threads_;

  mutex cluster_mutexes_mu_;
  absl::flat_hash_map<DeviceCompilationClusterSignature, std::unique_ptr<mutex>,
                      DeviceCompilationClusterSignature::Hash>
      cluster_mutexes_ TF_GUARDED_BY(cluster_mutexes_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompilationCache);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
