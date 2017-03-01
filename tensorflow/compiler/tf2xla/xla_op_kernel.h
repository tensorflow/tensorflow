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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_OP_KERNEL_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_OP_KERNEL_H_

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class XlaOpKernelContext;

// Implementations of operators that generate XLA code should usually subclass
// XlaOpKernel and implement the Compile() method. Unlike a regular OpKernel,
// an XlaOpKernel produces and consumes symbolic values during compilation.
//
// See the comments in xla_context.h for more details.
class XlaOpKernel : public OpKernel {
 public:
  explicit XlaOpKernel(OpKernelConstruction* construction);

  // Subclasses should implement Compile(), much as standard OpKernels implement
  // Compute().
  virtual void Compile(XlaOpKernelContext* context) = 0;

 private:
  void Compute(OpKernelContext* context) final;
};

// The context passed to the Compile() method of XlaOpKernel. An
// XlaOpKernelContext is a variant of the standard OpKernel class, tailored for
// implementing operators that perform symbolic execution as part of the XLA
// compiler. The key difference is that XlaOpKernelContext produces and consumes
// data as XLA computations, rather than as standard Tensors.
//
// Under the hood, symbolic execution communicates using special Tensors that
// wrap XlaExpression objects, however this is an implementation detail that
// this class hides. The *only* correct way to allocate a Tensor during
// compilation is using the XlaOpKernelContext methods, since they ensure there
// is a valid XlaExpression backing the tensor. No Op should ever call
// allocate_output or allocate_temp directly on the underlying OpKernelContext.
class XlaOpKernelContext {
 public:
  explicit XlaOpKernelContext(OpKernelContext* context);

  // Returns the XLA ComputationBuilder containing the output of compilation.
  xla::ComputationBuilder* builder() const;

  // Inputs

  // Returns the number of inputs to the operator.
  int num_inputs() const { return context_->num_inputs(); }

  // Returns the type of input 'index'.
  DataType input_type(int index) { return context_->input(index).dtype(); }

  // Returns the shape of input 'index'.
  TensorShape InputShape(int index);

  // Returns input 'index' as a ComputationDataHandle. Unlike
  // OpKernelContext::Input returns a symbolic value rather than a concrete
  // Tensor.
  const xla::ComputationDataHandle& Input(int index);

  // Returns true if all inputs are the same shape, otherwise sets the
  // status to a non-OK value and returns false.
  // Usage: if (!context->ValidateInputsAreSameShape(this)) return;
  bool ValidateInputsAreSameShape(OpKernel* op) TF_MUST_USE_RESULT;

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list.
  Status InputList(StringPiece name,
                   std::vector<xla::ComputationDataHandle>* handles,
                   std::vector<TensorShape>* shapes);

  // Helper methods for constant inputs.

  // Evaluates input 'index' and stores it in '*constant_literal'. If the
  // expression cannot be evaluated, e.g., because it depends on unbound
  // parameters, returns a non-OK status.
  Status ConstantInput(int index, xla::Literal* constant_literal);

  // Evaluates input 'index', reshapes it to 'new_shape' if new_shape !=
  // InputShape(index), and stores it in '*constant_literal'. If the input
  // cannot be evaluated, e.g., because it depends on unbound parameters,
  // returns a non-Ok status. If InputShape(index).num_elements() !=
  // new_shape.num_elements(), returns an error status.
  Status ConstantInputReshaped(int index, gtl::ArraySlice<int64> new_shape,
                               xla::Literal* constant_literal);

  // Converts a constant 1D int32 or int64 tensor into a vector of int64s.
  Status ConstantInputAsIntVector(int index, std::vector<int64>* out);

  // Converts a constant 1D int32 or int64 tensor into a TensorShape.
  Status ConstantInputAsShape(int index, TensorShape* shape);

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list.
  Status ConstantInputList(StringPiece name,
                           std::vector<xla::Literal>* literals);

  // Outputs

  int num_outputs() const { return context_->num_outputs(); }
  DataType expected_output_dtype(int index) const {
    return context_->expected_output_dtype(index);
  }

  // Sets output 'index' to the ComputationDataHandle 'handle'.
  // All outputs should be set using SetOutput and SetConstantOutput, not
  // via the underlying OpKernelContext.
  void SetOutput(int index, const xla::ComputationDataHandle& handle);

  // Sets output 'index' to compile-time constant 'host_tensor', where
  // 'host_tensor' is a tensor in host memory. It is preferable to use
  // SetConstantOutput where possible.
  void SetConstantOutput(int index, const Tensor& host_tensor);

  // Status handling.
  void SetStatus(const Status& status) { context_->SetStatus(status); }
  Status status() { return context_->status(); }

  // Mark the op has having side effects (i.e., via Send).
  void SetOpHasSideEffects();

  // Variables

  // Reads the current value of the resouce variable referred to by input
  // 'index'.
  Status ReadVariableInput(int index, xla::ComputationDataHandle* value);

  // Sets output 'index' to be a reference to variable 'variable_id'. Used
  // to propagate resource variables through the compilation.
  void SetVariableOutput(int index, int variable_id);

  // Assigns the value `handle` to the variable referenced by input
  // `variable_index`. Marks the operator as having side effects.
  Status AssignVariable(int variable_index, DataType type,
                        const xla::ComputationDataHandle& handle);

  // Helper routines for the OP_REQUIRES macros
  void CtxFailure(Status s);
  void CtxFailureWithWarning(Status s);

  // If this kernel invocation is within a function execution,
  // call_frame() returns the call frame for the function call.
  FunctionCallFrame* call_frame() const { return context_->call_frame(); }

  FunctionLibraryRuntime* function_library() const {
    return context_->function_library();
  }

  const OpKernel& op_kernel() const { return context_->op_kernel(); }

  // Returns the underlying OpKernelContext. Use rarely.
  OpKernelContext* op_kernel_context() const { return context_; }

  // TODO(phawkins): find a better home for these helpers.

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

 private:
  OpKernelContext* const context_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_OP_KERNEL_H_
