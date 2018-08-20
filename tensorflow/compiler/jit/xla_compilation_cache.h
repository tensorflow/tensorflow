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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Struct that represents a possibly-absent Tensor.
struct OptionalTensor {
  string name;           // A descriptive name
  bool present = false;  // Is the tensor present?
  Tensor value;          // If present, what is the Tensor's value?
};

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

  // Compiles a function into a XlaCompiler::CompilationResult that can be used
  // to execute an XLA Computation. Compilation results are cached.
  // `function` is the name of a Tensorflow function to compile.
  // `constant_args` is a map of tensorflow argument number to its constant
  //  value.
  // `variable_args` is a snapshot of the current values of the
  // resource variable arguments to `function`; uninitialized variables are
  // represented by an absent OptionalTensor.
  // The result of compilation is written to `*compilation_result`, which must
  // be non-null. If `executable` is non-null, also builds an
  // xla::LocalExecutable and sets `executable` to point to it. The resulting
  // executable pointer may be null if the computation has no non-constant
  // outputs.
  Status Compile(const XlaCompiler::Options& options,
                 const NameAttrList& function,
                 const std::map<int, Tensor>& constant_args,
                 const std::map<int, OptionalTensor>& variable_args,
                 OpKernelContext* ctx,
                 const XlaCompiler::CompilationResult** compilation_result,
                 xla::LocalExecutable** executable,
                 const XlaCompiler::CompileOptions* compile_options);

  // As above, but calls XlaCompiler::CompileSingleOp instead of
  // XlaCompiler::CompileFunction.
  Status CompileSingleOp(
      const XlaCompiler::Options& options,
      const std::map<int, Tensor>& constant_args,
      const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
      const XlaCompiler::CompilationResult** compilation_result,
      xla::LocalExecutable** executable,
      const XlaCompiler::CompileOptions* compile_options);

  xla::LocalClient* client() const { return client_; }
  const DeviceType& device_type() const { return device_type_; }

  string DebugString() override;

 private:
  // Common implementation of Compile and CompileSingleOp.
  Status CompileImpl(const XlaCompiler::Options& options,
                     const NameAttrList& function,
                     const std::map<int, Tensor>& constant_args,
                     const std::map<int, OptionalTensor>& variable_args,
                     OpKernelContext* ctx,
                     const XlaCompiler::CompilationResult** compilation_result,
                     xla::LocalExecutable** executable,
                     const XlaCompiler::CompileOptions* compile_options,
                     bool compile_single_op);

  // Takes `result` which has been compiled from a Tensorflow subgraph to a
  // XLA computation already, and generates an XLA LocalExecutable `executable`.
  Status BuildExecutable(const XlaCompiler::Options& options,
                         const XlaCompiler::CompilationResult& result,
                         std::unique_ptr<xla::LocalExecutable>* executable);

  xla::LocalClient* const client_;
  const DeviceType device_type_;

  // Describes the types, shapes and any compile-time constant arguments
  // to a kernel. Key that uniquely identifies a compilation output.
  struct Signature {
    string name;

    std::vector<std::pair<DataType, TensorShape>> arg_types;

    // List of Tensor values for compile-time constant arguments to the
    // compilation, ordered by argument number. Tensors must be in host memory.
    std::vector<Tensor> arg_values;

    bool operator==(const Signature& other) const;

    struct Hash {
      uint64 operator()(const Signature& signature) const;
    };
  };
  static string SignatureDebugString(const Signature& sig);

  // Builds the signature for a compilation.
  Status BuildSignature(const NameAttrList& function,
                        const std::map<int, Tensor>& constant_args,
                        const std::map<int, OptionalTensor>& variable_args,
                        OpKernelContext* ctx, Signature* signature);

  // The value associated with a cache entry.
  struct Entry {
    mutex mu;

    // Have we tried compiling this entry?
    bool compiled = false;

    // Did compilation succeed?
    Status compilation_status GUARDED_BY(mu);

    // Output of the XlaCompiler.
    XlaCompiler::CompilationResult compilation_result GUARDED_BY(mu);

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<xla::LocalExecutable> executable GUARDED_BY(mu);
  };

  mutex compile_cache_mu_;
  gtl::FlatMap<Signature, std::unique_ptr<Entry>, Signature::Hash> cache_
      GUARDED_BY(compile_cache_mu_);

  struct CompileStats {
    // Number of times the cluster has been (re-)compiled.
    int64 compile_count = 0;

    // Cumulative time spent compiling the cluster.
    int64 cumulative_compile_time_us = 0;
  };
  mutex compile_stats_mu_;

  // Maps cluster names to compilation statistics for said cluster.
  gtl::FlatMap<string, CompileStats> compile_stats_
      GUARDED_BY(compile_stats_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompilationCache);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
