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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
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

  XlaContext* xla_context() const;

  // Returns the XLA XlaBuilder containing the output of compilation.
  xla::XlaBuilder* builder() const;

  xla::ValueInference& value_inference();

  // Inputs

  // Returns the number of inputs to the operator.
  int num_inputs() const { return context_->num_inputs(); }

  // Returns the type of input `index`.
  DataType input_type(int index) const;

  // Returns the type of input `name`.
  DataType InputType(absl::string_view name);

  // Returns the type of input `index` as an xla::PrimitiveType. If the type
  // is not representable as an XLA type, sets an error status and returns
  // xla::PRIMITIVE_TYPE_INVALID.
  xla::PrimitiveType input_xla_type(int index);

  // Returns the type of input `name` as an xla::PrimitiveType. If the type
  // is not representable as an XLA type, sets an error status and returns
  // xla::PRIMITIVE_TYPE_INVALID.
  xla::PrimitiveType InputXlaType(absl::string_view name);

  // Returns the shape of input at `index` or input the given `name`. Note that
  // in case the shape of the input is not static, then the returned shape has
  // bounds as the dimension size instead of having unknown dimensions. Use
  // InputXlaShape instead that provides shapes with dynamism information.
  //
  ABSL_DEPRECATED(
      "Prefer InputXlaShape which handles dynamic shapes accurately.")
  TensorShape InputShape(int index);
  ABSL_DEPRECATED(
      "Prefer InputXlaShape which handles dynamic shapes accurately.")
  TensorShape InputShape(absl::string_view name);

  // Returns input `index` as a XlaOp. Unlike
  // OpKernelContext::Input returns a symbolic value rather than a concrete
  // Tensor.
  xla::XlaOp Input(int index);
  // Returns input `name` as a XlaOp.
  xla::XlaOp Input(absl::string_view name);

  // Returns the xla input shape for a given index.
  StatusOr<xla::Shape> InputXlaShape(int index);
  StatusOr<xla::Shape> InputXlaShape(absl::string_view name);

  // Returns true if all inputs are the same shape, otherwise sets the
  // status to a non-OK value and returns false.
  // Usage: if (!context->ValidateInputsAreSameShape(this)) return;
  bool ValidateInputsAreSameShape(OpKernel* op) TF_MUST_USE_RESULT;

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list.
  Status InputList(absl::string_view name, std::vector<xla::XlaOp>* handles,
                   std::vector<TensorShape>* shapes);
  // Evaluates input and returns their dynamism vector in a vector of
  // predicates.
  Status ResolveInputDynamismIntoPredVector(int index, std::vector<bool>* out);
  Status ResolveInputDynamismIntoPred(int index, bool* out);
  Status ResolveInputDynamismIntoPredVector(absl::string_view name,
                                            std::vector<bool>* out);
  Status ResolveInputDynamismIntoPred(absl::string_view name, bool* out);

  Status ResolveInputDynamism(int index, xla::Literal* dynamism_literal);
  Status ResolveInputDynamism(absl::string_view name,
                              xla::Literal* dynamism_literal);

  Status ResolveInputDynamismReshaped(int index,
                                      absl::Span<const int64_t> new_dims,
                                      xla::Literal* dynamism_literal);
  // Helper methods for constant inputs.

  // Evaluates input `index` and stores it in `*constant_literal`. If the
  // expression cannot be evaluated, e.g., because it depends on unbound
  // parameters, returns a non-OK status. This function can also be used to
  // infer constant input upper or lower bounds, by changing the `mode`
  // parameter.
  Status ConstantInput(
      int index, xla::Literal* constant_literal,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);
  Status ConstantInput(
      absl::string_view name, xla::Literal* constant_literal,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant scalar int32 or int64 tensor into an int64.
  Status ConstantInputAsIntScalar(
      int index, int64_t* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);
  Status ConstantInputAsIntScalar(
      absl::string_view name, int64_t* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  StatusOr<int64_t> ConstantInputAsIntScalar(
      absl::string_view name,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant scalar float32 or float64 tensor into a float64.
  Status ConstantInputAsFloatScalar(
      int index, double* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant 1D int32 or int64 tensor into a vector of int64s.
  Status ConstantInputAsIntVector(
      int index, std::vector<int64_t>* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);
  Status ConstantInputAsIntVector(
      absl::string_view name, std::vector<int64_t>* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Reshapes and converts a constant int32 or int64 tensor into a vector of
  // int64s.
  Status ConstantInputReshapedToIntVector(
      int index, std::vector<int64_t>* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);
  Status ConstantInputReshapedToIntVector(
      absl::string_view name, std::vector<int64_t>* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant int32 or int64 Tensor into an xla int64 Literal.
  Status ConstantInputAsInt64Literal(
      int index, xla::Literal* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);
  Status ConstantInputAsInt64Literal(
      absl::string_view name, xla::Literal* out,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant 1D int32 or int64 tensor into a TensorShape.
  Status ConstantInputAsShape(
      int index, TensorShape* shape,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Converts a constant 1D int32 or int64 tensor, or a scalar with value -1
  // into a PartialTensorShape.
  Status ConstantInputAsPartialShape(int index, PartialTensorShape* shape);

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list.
  Status ConstantInputList(
      absl::string_view name, std::vector<xla::Literal>* outputs,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Returns the Tensor representation of the constant input.
  StatusOr<Tensor> ConstantInputTensor(
      int index,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  // Returns an XlaExpression describing the value of 'index'.
  const XlaExpression& InputExpression(int index);
  const XlaExpression& InputExpression(absl::string_view name);

  // Outputs

  int num_outputs() const { return context_->num_outputs(); }
  DataType expected_output_dtype(int index) const {
    return context_->expected_output_dtype(index);
  }

  // Returns the type of output `index` as an xla::PrimitiveType. If the type
  // is not representable as an XLA type, sets an error status and returns
  // xla::PRIMITIVE_TYPE_INVALID.
  xla::PrimitiveType output_xla_type(int index);

  // Sets output `index` to the XlaOp `handle`.
  // All outputs should be set using SetOutput and SetConstantOutput, not
  // via the underlying OpKernelContext.
  void SetOutput(int index, const xla::XlaOp& handle);

  // Sets output `index` to compile-time constant `host_tensor`, where
  // `host_tensor` is a tensor in host memory. It is preferable to use
  // SetConstantOutput where possible.
  void SetConstantOutput(int index, const Tensor& host_tensor);

  // Returns an XlaExpression describing the value of 'index'.
  void SetOutputExpression(int index, const XlaExpression& expression);

  // Sets output `index` to the Tensor List `handle`.
  void SetTensorListOutput(int index, const xla::XlaOp& handle);

  // Status handling.
  void SetStatus(const Status& status) { context_->SetStatus(status); }
  Status status() { return context_->status(); }

  // Variables

  // Sets `*resource` to the resource associated with input `index`.
  Status GetResourceInput(int index, XlaResource** resource);

  // Sets output `index` to be a reference to resource `resource`.
  void SetResourceOutput(int index, XlaResource* resource);

  // Sets `*type` and `*shape` to the current type and shape of a variable's
  // value.
  Status GetVariableTypeAndShape(int index, DataType* type,
                                 TensorShape* shape) const;

  // When dynamic_dimension_is_minus_one is set, querying a dynamic dimension
  // returns "-1", this is useful when the underlying ops expect explicit
  // dynamic index like reshape.
  void set_dynamic_dimension_is_minus_one(bool value) {
    dynamic_dimension_is_minus_one_ = value;
  }

  bool dynamic_dimension_is_minus_one() const {
    return dynamic_dimension_is_minus_one_;
  }

  bool is_dynamic_dimension(int64_t dim_size) { return dim_size == -1; }

  // Reads the current value of the resource variable referred to by input
  // `index`. If `shape` is not nullptr, sets `*shape` to the shape of the
  // variable. Returns an error if the variable has not been initialized, or if
  // its type does not match `type`.
  Status ReadVariableInput(int index, DataType type, TensorShape* shape,
                           xla::XlaOp* value);
  // Reads the current value of the resource variable referred to by input
  // `name`.
  Status ReadVariableInput(absl::string_view name, DataType type,
                           TensorShape* shape, xla::XlaOp* value);

  // Assigns the value `handle` to the variable referenced by input
  // `input_index`. The variable must be of `type`. Returns an error if the
  // variable has been initialized with a different type or with a
  // different shape.
  Status AssignVariable(int input_index, DataType type, xla::XlaOp handle);
  // Assigns the value `handle` to the variable referenced by input `name`.
  Status AssignVariable(absl::string_view name, DataType type,
                        xla::XlaOp handle);

  // Helper routines for the OP_REQUIRES macros
  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  // If this kernel invocation is within a function execution,
  // call_frame() returns the call frame for the function call.
  CallFrameInterface* call_frame() const { return context_->call_frame(); }

  FunctionLibraryRuntime* function_library() const {
    return context_->function_library();
  }

  const OpKernel& op_kernel() const { return context_->op_kernel(); }

  // Returns the underlying OpKernelContext. Use rarely.
  OpKernelContext* op_kernel_context() const { return context_; }

  // Returns the XlaCompiler that is performing the compilation. Used for, e.g.,
  // While to compile nested computations.
  XlaCompiler* compiler() const;

  // TODO(phawkins): find a better home for these helpers.

  // Gets an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMax(const DataType type);

  // Gets an XLA lambda to compute Min. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMin(const DataType type);

  // Gets an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateAdd(const DataType type);

  // Gets an XLA lambda to compute LogAddExp. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateLogAddExp(const DataType type);

  // Gets an XLA lambda to compute Mul. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::XlaComputation* GetOrCreateMul(const DataType type);

  // Returns stack trace encoded as a string at a given module, or an empty
  // string if none found.
  std::string StackTrace() const;

 private:
  // Returns the tensor of input `name`.
  const Tensor& GetInputTensorByName(absl::string_view name);
  // Evaluates input `index`, reshapes it to `new_shape` if new_shape !=
  // InputShape(index), and stores it in `*constant_literal`. If the input
  // cannot be evaluated, e.g., because it depends on unbound parameters,
  // returns a non-Ok status. If InputShape(index).num_elements() !=
  // new_shape.num_elements(), returns an error status.
  Status ConstantInputReshaped(
      int index, absl::Span<const int64_t> new_dims,
      xla::Literal* constant_literal,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue);

  OpKernelContext* const context_;
  bool dynamic_dimension_is_minus_one_;
  xla::ValueInference value_inference_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_OP_KERNEL_H_
