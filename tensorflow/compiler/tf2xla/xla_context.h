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

// This file defines the contexts used to represent XLA JIT computatations.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// A XlaExpression wraps an XLA computation. Each Tensor sent
// along an edge during XLA JIT compilation represents a
// XlaExpression, and the shape of the Tensor matches the shape of
// the subcomputation in the ComputationDataHandle. Each
// expression is either a constant, an unbound parameter, or a
// function of previously-compiled expressions.
class XlaExpression {
 public:
  XlaExpression();

  // handle() stores the XLA handle of the computation that the
  // expression represents.
  void set_handle(const xla::ComputationDataHandle& h);
  const xla::ComputationDataHandle& handle() const;

  void set_constant_value(Tensor value);
  bool has_constant_value() const { return has_constant_value_; }
  const Tensor& constant_value() const { return constant_value_; }

 private:
  friend class XlaContext;

  // The XLA handle of the expression's computation.
  xla::ComputationDataHandle handle_;

  // If this expression is a constant with a known value, 'constant_value' is a
  // host-memory Tensor containing the value. Used to avoid invoking XLA for
  // expressions that are trivially constant.
  bool has_constant_value_;
  Tensor constant_value_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExpression);
};

// The XlaContext is the data structure accessible from
// OpKernelContexts when evaluating a subgraph of Ops for JIT
// compilation by XLA. When an Op is executed during JIT
// compilation the input Tensors to the Op store handles to
// subcomputations compiled by earlier Ops in the subgraph. The Op can
// retrieve these subcomputations by calling either
// GetExpressionFromTensor, which returns the XlaExpression holding
// the subcomputation; or EvaluateAsConstant which returns an XLA
// literal of the result of the subcomputation or an error status if
// the subcomputation depends on unbound parameters. The Op may then
// use the ComputationBuilder available from XlaContext::builder()
// to compile one or more functions of the inputs into
// ComputationDataHandles. The handles can be stored as new
// expressions corresponding to the outputs of the Op by calling
// CreateOutputTensorFromComputation or
// CreateConstantOutputTensor. The *only* correct way to allocate an
// output tensor is using one of the preceding two methods, since they
// ensure there is a valid XlaExpression backing the output
// tensor. No Op should ever call allocate_output or allocate_temp
// directly on the OpKernelContext. It is permissible to pass a tensor
// from an Op input to an output (e.g. call ctx->set_output with a
// tensor passed as an input). As an example, the softmax Op produces
// output from input as follows:
//
//    XlaContext& tc = XlaContext::Get(context);
//    xla::ComputationBuilder& b = tc.builder();
//    xla::ComputationDataHandle logits =
//        tc.GetComputationFromTensor(logits_in));
//    ... The softmax computation uses the builder b to compute a
//        xla::ComputationDataHandle softmax holding the desired output.
//    ...
//    OP_REQUIRES_OK(context, tc.CreateOutputTensorFromComputation(
//                                context, 0, logits_in.shape().dim_sizes(),
//                                softmax));
//
class XlaContext : public ResourceBase {
 public:
  // If a retval can be evaluated at JIT time it is returned as a
  // Literal in a ConstRetVal struct as part of the ComputationResult.
  // TODO(misard) reconcile this with the duplicate data structure in
  // the XlaCompilationCache class.
  struct ConstRetVal {
    // The index of the RetVal corresponding to this constant literal.
    int index;
    // If status is not OK, value's data is undefined.
    Status status;
    // The value of the RetVal evaluated at JIT compilation
    // time. value.shape() always gives the correct shape of the
    // RetVal. If !status.ok() then value's data is undefined, otherwise the
    // Tensor buffer is allocated in CPU memory.
    Tensor value;
  };


  // Virtual method defined by ResourceBase.
  string DebugString() override;

  // Retrieve the XlaContext corresponding to a step's JIT compilation.
  static XlaContext& Get(const OpKernelContext* ctx);
  static XlaContext& Get(const XlaOpKernelContext* ctx) {
    return Get(ctx->op_kernel_context());
  }

  // Create a new XlaContext.
  XlaContext(XlaCompiler* compiler, xla::Client* client,
             const string& computation_name, bool allow_cpu_custom_calls,
             bool resolve_compile_time_constants);

  // Builds XLA computations for each of the arguments.
  // Should only be called once to initialize the arguments. Not thread-safe.
  Status BuildArguments(std::vector<XlaCompiler::Argument> arguments,
                        bool use_tuple_arg) TF_MUST_USE_RESULT;

  // Returns the results of the symbolic computation that have accumulated in
  // the XlaContext. After CollectResults() is called, the context is left in
  // an invalid state and must not be reused.
  // Sets `requires_runtime_context` if the emitted computation requires a
  // runtime context argument. `compile_time_constants` describes any non
  // data-dependent results of the computation. `num_nonconst_ouputs` is set to
  // the number of outputs of the `computation`.
  Status CollectResults(xla::Computation* computation,
                        bool* requires_runtime_context,
                        std::vector<ConstRetVal>* compile_time_constants,
                        int* num_nonconst_outputs);

  // This is called by the Retval Op to associate a computed value
  // with a specific return value of the subgraph.
  void AddRetval(int retval_index, const xla::ComputationDataHandle& handle);

  // As for Retval, but for return values that are compile-time constants.
  Status AddConstRetval(int retval_index, DataType dtype,
                        const xla::Literal& literal);

  // Mark the computation as having side effects (i.e., Send operators).
  void AddSideEffects();

  // Retrieves the ComputationDataHandle from an input Tensor to an Op. This
  // computation was constructed by an Op that executed previously and
  // created the output Tensor using CreateOutputTensorFromComputation
  // or CreateConstantOutputTensor.
  static const xla::ComputationDataHandle& GetComputationFromTensor(
      const Tensor& tensor);

  XlaCompiler* compiler() const { return compiler_; }

  // Returns the ComputationBuilder that Ops use for compiling new
  // expressions.
  xla::ComputationBuilder& builder();

  const std::vector<XlaCompiler::Argument>& args() const { return args_; }
  xla::ComputationDataHandle parameter(int num) { return parameters_[num]; }

  // Get the runtime context parameter, adding one if it does not already exist.
  // Dies if not compiling a local executable.
  const xla::ComputationDataHandle& GetOrCreateRuntimeContextParameter();

  bool allow_cpu_custom_calls() const { return allow_cpu_custom_calls_; }

  // Get an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateMax(const DataType type);

  // Get an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateAdd(const DataType type);

  // Get an XLA lambda to compute Sigmoid. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateSigmoid(const DataType type);

  // The name of the XlaContext resource during symbolic graph execution.
  static const char kXlaContextResourceName[];

 private:
  friend class XlaOpKernelContext;

  // This method is used to retrieve an expression that was allocated by
  // a previous Op.
  static const XlaExpression* CastExpressionFromTensor(const Tensor& tensor);

  // This method is used to retrieve an uninitialized expression from a
  // newly-allocated tensor.
  static XlaExpression* CastExpressionFromUninitializedTensor(Tensor* tensor);

  // Retrieves the expression from an input Tensor to an Op. This
  // expression was constructed by an Op that executed previously and
  // created the output Tensor using CreateOutputTensorFromComputation
  // or CreateConstantOutputTensor.
  static const XlaExpression* GetExpressionFromTensor(const Tensor& tensor);

  XlaCompiler* const compiler_;

  mutable mutex mu_;

  // The ComputationBuilder used to construct the subgraph's compiled
  // representation.
  xla::ComputationBuilder xla_builder_ GUARDED_BY(mu_);

  // Number of XLA Parameters, not counting the context parameter, if any.
  int num_parameters_;

  // Arguments to the JIT compilation, both compile-time constant arguments and
  // runtime parameters.
  std::vector<XlaCompiler::Argument> args_;
  bool use_tuple_arg_ = false;

  // Runtime parameters to the XLA computation. Does not include
  // compile-time constant arguments.
  std::vector<xla::ComputationDataHandle> parameters_;

  // Allow ops to emit CustomCall operations for CPU.
  const bool allow_cpu_custom_calls_;

  // If true, constant return values are returned as Tensors instead of
  // run-time computation outptus.
  const bool resolve_compile_time_constants_;

  // When 'has_context_parameter_' is true, this is the computation handle
  // for an additional final parameter to the computation, through which will be
  // passed a XlaLocalRuntimeContext* at runtime. Created on demand by
  // GetOrCreateRuntimeContextParameter().
  bool has_context_parameter_ GUARDED_BY(mu_) = false;
  xla::ComputationDataHandle context_parameter_ GUARDED_BY(mu_);

  // The data-dependent return values of the computation.
  std::vector<std::pair<int, xla::ComputationDataHandle>> retval_
      GUARDED_BY(mu_);

  // The non-data-dependent return values of the computation.
  std::vector<ConstRetVal> compile_time_constant_ GUARDED_BY(mu_);

  // Does the computation have side effects, i.e., Send() calls?
  bool has_side_effects_ GUARDED_BY(mu_) = false;

  // Cache of prebuilt computations indexed by their type.
  using ComputationMap = std::map<DataType, xla::Computation>;

  // Finds the value for the given type in out map if it already
  // exists or makes a new value with create function and keeps it the
  // map. The returned value != nullptr and is owned by the map.
  const xla::Computation* LookupOrCreate(
      DataType type, ComputationMap* out,
      const std::function<xla::Computation()>& create) LOCKS_EXCLUDED(mu_);

  // Cached computation to compute Max of two elements, specialized by type.
  ComputationMap max_func_ GUARDED_BY(mu_);

  // Cached computation to compute Sum of two elements, specialized by type.
  ComputationMap add_func_ GUARDED_BY(mu_);

  // Cached computation to compute Sigmoid of an element, specialized by type.
  ComputationMap sigmoid_func_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
