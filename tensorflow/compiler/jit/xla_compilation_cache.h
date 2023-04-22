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

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

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
  XlaCompilationCache(xla::LocalClient* client, DeviceType device_type);
  ~XlaCompilationCache() override;

  enum class CompileMode {
    kLazy,
    kStrict,
    kAsync,
  };

  enum class CompileState {
    kUncompiled,
    kCompiling,
    kCompiled,
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
                 CompileMode compile_mode,
                 const XlaCompiler::CompilationResult** out_compilation_result,
                 xla::LocalExecutable** out_executable);

  // As above, but calls XlaCompiler::CompileSingleOp instead of
  // XlaCompiler::CompileFunction. If MLIR bridge is enabled through ConfigProto
  // in OpKernelContext, then uses MLIR bridge for compilation instead of
  // XlaCompiler, if possible.
  Status CompileSingleOp(
      const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args, OpKernelContext* ctx,
      const XlaCompiler::CompileOptions& compile_options,
      const XlaCompiler::CompilationResult** out_compilation_result,
      xla::LocalExecutable** out_executable);

  xla::LocalClient* client() const { return client_; }
  const DeviceType& device_type() const { return device_type_; }

  string DebugString() const override;

  // Describes the types, shapes and any compile-time constant arguments
  // to a kernel. Key that uniquely identifies a compilation output.
  struct Signature {
    string name;

    // List of Tensor types & shapes for compile-time constant arguments to the
    // compilation, ordered by argument number.
    absl::InlinedVector<std::pair<DataType, absl::InlinedVector<int64, 4>>, 4>
        arg_shapes;

    // List of Tensor values for compile-time constant arguments to the
    // compilation, ordered by argument number. Tensors must be in host memory.
    absl::InlinedVector<Tensor, 4> arg_values;

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
  // Common implementation of Compile and CompileSingleOp.
  Status CompileImpl(
      const XlaCompiler::Options& options, const NameAttrList& function,
      const std::vector<XlaCompiler::Argument>& args,
      const std::function<Status(XlaCompiler* compiler,
                                 const std::vector<XlaCompiler::Argument>& args,
                                 XlaCompiler::CompilationResult*)>& compile_fn,
      CompileMode compile_mode,
      const XlaCompiler::CompilationResult** out_compilation_result,
      xla::LocalExecutable** out_executable);

  // Takes `result` which has been compiled from a Tensorflow subgraph to a
  // XLA computation already, and generates an XLA LocalExecutable `executable`.
  Status BuildExecutable(const XlaCompiler::Options& options,
                         const XlaCompiler::CompilationResult& result,
                         std::unique_ptr<xla::LocalExecutable>* executable);

  xla::LocalClient* const client_;
  const DeviceType device_type_;

  // The value associated with a cache entry.
  struct Entry {
    mutex mu;

    // The current compilation state for this entry.
    CompileState compile_state = CompileState::kUncompiled;

    // The number of times a compilation with this signature has been requested.
    int64 request_count = 0;

    // Did compilation succeed?
    Status compilation_status TF_GUARDED_BY(mu);

    // Output of the XlaCompiler.
    XlaCompiler::CompilationResult compilation_result TF_GUARDED_BY(mu);

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<xla::LocalExecutable> executable TF_GUARDED_BY(mu);
  };

  Status CompileStrict(
      Entry* entry, const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const string& function_name,
      const std::function<Status(XlaCompiler* compiler,
                                 const std::vector<XlaCompiler::Argument>& args,
                                 XlaCompiler::CompilationResult*)>& compile_fn)
      TF_EXCLUSIVE_LOCKS_REQUIRED(entry->mu);
  Status CompileAsynchronous(
      Entry* entry, const XlaCompiler::Options& options,
      const std::vector<XlaCompiler::Argument>& args,
      const string& function_name,
      const std::function<Status(XlaCompiler* compiler,
                                 const std::vector<XlaCompiler::Argument>& args,
                                 XlaCompiler::CompilationResult*)>& compile_fn);

  mutex compile_cache_mu_;
  absl::flat_hash_map<Signature, std::unique_ptr<Entry>, Signature::Hash> cache_
      TF_GUARDED_BY(compile_cache_mu_);

  struct ClusterCompileStats {
    // Number of times the cluster has been (re-)compiled.
    int64 compile_count = 0;

    // The number of times this cluster has been executed.
    int64 execution_count = 0;

    // Cumulative time spent compiling the cluster.
    int64 cumulative_compile_time_us = 0;

    // True if we have decided that this cluster is too dynamic (i.e. its shapes
    // change too frequently) to profitably JIT compile.  Once a cluster is
    // tagged megamorphic, it stays megamorphic forever.
    bool is_megamorphic = false;
  };

  mutex cluster_compile_stats_mu_;

  // Maps cluster names to compilation statistics for said cluster.
  absl::flat_hash_map<string, ClusterCompileStats> cluster_compile_stats_
      TF_GUARDED_BY(cluster_compile_stats_mu_);

  struct AsyncCompilationState {
    mutex async_compilation_state_mu;

    // Number of threads for asynchronous compilations.
    static constexpr int64 kNumCompilerThreads = 10;

    // Maximum number of ongoing compilations.
    static constexpr int64 kMaxNumOngoingCompilations = kNumCompilerThreads;

    // Number of ongoing compilations.
    int64 num_ongoing_compilations TF_GUARDED_BY(async_compilation_state_mu) =
        0;

    // Pool of threads for asynchronous compilations.
    std::unique_ptr<thread::ThreadPool> compiler_threads;

    AsyncCompilationState() {
      compiler_threads = absl::make_unique<tensorflow::thread::ThreadPool>(
          tensorflow::Env::Default(), "async_compiler_threads",
          kNumCompilerThreads);
    }

  } async_compilation_state_;

  // The number of times a lazy compilation must be requested for a specific
  // signature before  we attempt to compile it.
  static constexpr int64 kDefaultCompilationThreshold = 2;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompilationCache);
};

// Creates a single-node graph using the specified node_def as the only op apart
// from the arg and retval nodes.
StatusOr<std::unique_ptr<Graph>> CreateGraph(
    const NodeDef& node_def, absl::Span<const XlaCompiler::Argument> args,
    absl::Span<const DataType> result_types);

// Use XlaCompiler to compile a single op into HLO.
Status XlaSingleOpToHlo(XlaCompiler* compiler,
                        const XlaCompiler::Options& options,
                        const std::vector<XlaCompiler::Argument>& args,
                        OpKernelContext* ctx,
                        const XlaCompiler::CompileOptions& compile_options,
                        XlaCompiler::CompilationResult* compilation_result);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
