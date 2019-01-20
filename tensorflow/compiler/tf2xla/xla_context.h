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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
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

  // Creates a new XlaContext. See the documentation on the class data fields
  // for descriptions of the arguments.
  XlaContext(XlaCompiler* compiler, xla::XlaBuilder* builder);

  // Virtual method defined by ResourceBase.
  string DebugString() const override;

  XlaCompiler* compiler() const { return compiler_; }

  // Returns the XlaBuilder that Ops use for compiling new expressions.
  xla::XlaBuilder* builder() { return builder_; }

  const std::vector<XlaExpression>& args() const { return args_; }
  void set_args(std::vector<XlaExpression> args);

  const std::vector<XlaExpression>& retvals() { return retvals_; }

  // Sets a return value.
  // Since we do not always know in advance how many return values there are,
  // grows the return values vector to size index+1 if it is smaller.
  void SetRetval(int index, const XlaExpression& expression);

  // Adds 'resource' to the set of resources owned by the context.
  XlaResource* AddResource(std::unique_ptr<XlaResource> resource);

  const std::vector<std::unique_ptr<XlaResource>>& resources() {
    return resources_;
  }

  // Get an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMax(const DataType type);

  // Get an XLA lambda to compute Min. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMin(const DataType type);

  // Get an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateAdd(const DataType type);

  // Get an XLA lambda to compute Mul. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMul(const DataType type);

  // The name of the XlaContext resource during symbolic graph execution.
  static const char kXlaContextResourceName[];

 private:
  XlaCompiler* const compiler_;

  // The XlaBuilder used to construct the subgraph's compiled representation.
  xla::XlaBuilder* builder_;

  // Arguments to the Tensorflow graph, indexed by _Arg index.
  // Includes both compile-time constant arguments and runtime parameters.
  std::vector<XlaExpression> args_;

  // Return values of the Tensorflow graph, indexed by _Retval index.
  std::vector<XlaExpression> retvals_;

  // Holds ownership of resources. The resources are not ordered.
  std::vector<std::unique_ptr<XlaResource>> resources_;

  // Cache of prebuilt computations indexed by their type.
  using ComputationMap = std::map<DataType, xla::XlaComputation>;

  // Finds the value for the given type in out map if it already
  // exists or makes a new value with create function and keeps it the
  // map. The returned value != nullptr and is owned by the map.
  const xla::XlaComputation* LookupOrCreate(
      DataType type, ComputationMap* out,
      const std::function<xla::XlaComputation()>& create);

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
