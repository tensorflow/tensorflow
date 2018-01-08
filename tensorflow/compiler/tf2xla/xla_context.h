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

// This file defines the contexts used during XLA compilation.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class XlaOpKernelContext;

// The XlaContext is the data structure that holds the state of an XLA
// compilation, that is accessible from OpKernelContexts when compiling a
// subgraph of Ops using XLA.
class XlaContext : public ResourceBase {
 public:
  // Retrieves the XlaContext of the current compilation.
  static XlaContext& Get(const OpKernelContext* ctx);
  static XlaContext& Get(const XlaOpKernelContext* ctx);

  // Creates a new XlaContext.
  XlaContext(XlaCompiler* compiler, xla::ComputationBuilder* builder,
             bool allow_cpu_custom_calls, bool resolve_compile_time_constants);

  // Virtual method defined by ResourceBase.
  string DebugString() override;

  XlaCompiler* compiler() const { return compiler_; }

  // Returns the ComputationBuilder that Ops use for compiling new
  // expressions.
  xla::ComputationBuilder* builder();

  bool allow_cpu_custom_calls() const { return allow_cpu_custom_calls_; }

  const std::vector<XlaExpression>& args() const { return args_; }
  void set_args(std::vector<XlaExpression> args);

  const std::vector<XlaExpression>& retvals() { return retvals_; }

  // This is called by the Retval Op to associate a computed value
  // with a specific return value of the subgraph.
  void AddRetval(int retval_index, DataType type,
                 const xla::ComputationDataHandle& handle);

  // As for Retval, but for return values that are compile-time constants.
  Status AddConstRetval(int retval_index, DataType dtype,
                        const xla::Literal& literal);

  // Creates a resource with resource `kind` and initial type `type` and
  // value `handle`. `name` is a descriptive name for use in error messages.
  // Fails if the resource already exists.
  Status CreateResource(XlaResource::Kind kind, int arg_num, string name,
                        DataType type, const xla::ComputationDataHandle& handle,
                        XlaResource** resource);

  const std::vector<std::unique_ptr<XlaResource>>& resources() {
    return resources_;
  }

  // Get an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateMax(const DataType type);

  // Get an XLA lambda to compute Min. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateMin(const DataType type);

  // Get an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateAdd(const DataType type);

  // Get an XLA lambda to compute Mul. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateMul(const DataType type);

  // The name of the XlaContext resource during symbolic graph execution.
  static const char kXlaContextResourceName[];

 private:
  XlaCompiler* const compiler_;

  // The ComputationBuilder used to construct the subgraph's compiled
  // representation.
  xla::ComputationBuilder* builder_;

  // Allow ops to emit CustomCall operations for CPU.
  const bool allow_cpu_custom_calls_;

  // If true, constant return values are returned as Tensors instead of
  // run-time computation outputs.
  const bool resolve_compile_time_constants_;

  // Arguments to the Tensorflow graph, indexed by _Arg index.
  // Includes both compile-time constant arguments and runtime parameters.
  std::vector<XlaExpression> args_;

  // Return values of the Tensorflow graph, indexed by _Retval index.
  std::vector<XlaExpression> retvals_;

  // Holds ownership of resources. The resources are not ordered.
  std::vector<std::unique_ptr<XlaResource>> resources_;

  // Cache of prebuilt computations indexed by their type.
  using ComputationMap = std::map<DataType, xla::Computation>;

  // Finds the value for the given type in out map if it already
  // exists or makes a new value with create function and keeps it the
  // map. The returned value != nullptr and is owned by the map.
  const xla::Computation* LookupOrCreate(
      DataType type, ComputationMap* out,
      const std::function<xla::Computation()>& create);

  // Cached computation to compute Max of two elements, specialized by type.
  ComputationMap max_func_;

  // Cached computation to compute Min of two elements, specialized by type.
  ComputationMap min_func_;

  // Cached computation to compute Sum of two elements, specialized by type.
  ComputationMap add_func_;

  // Cached computation to compute Mul of two elements, specialized by type.
  ComputationMap mul_func_;

  // Cached computation to compute Sigmoid of an element, specialized by type.
  ComputationMap sigmoid_func_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
