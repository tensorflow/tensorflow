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

#include "tensorflow/compiler/tf2xla/xla_context.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

XlaExpression::XlaExpression() : has_constant_value_(false) {}

void XlaExpression::set_handle(const xla::ComputationDataHandle& h) {
  handle_ = h;
}
const xla::ComputationDataHandle& XlaExpression::handle() const {
  return handle_;
}

void XlaExpression::set_constant_value(Tensor value) {
  has_constant_value_ = true;
  constant_value_ = std::move(value);
}

const char XlaContext::kXlaContextResourceName[] = "_xla_context";

// Looks up the context associated with the current step. It is stored
// in a resource container managed by the device.
/* static */ XlaContext& XlaContext::Get(const OpKernelContext* ctx) {
  // When an Op kernel wants to use an XLA JIT context, the
  // per-step context is looked up in the resource manager. The
  // JIT will prepopulate the JITContext.
  XlaContext* context;
  TF_CHECK_OK(ctx->resource_manager()->Lookup(
      ctx->step_container()->name(), kXlaContextResourceName, &context));
  // The resource manager handed us a fresh reference to 'context', but retains
  // a reference itself so the context won't be freed. The resource manager will
  // outlive the JIT compilation.
  context->Unref();
  return *context;
}

Status XlaContext::BuildArguments(std::vector<XlaCompiler::Argument> args,
                                  bool use_tuple_arg) {
  args_ = std::move(args);
  use_tuple_arg_ = use_tuple_arg;

  // Compute the number of parameters, verify that they are sequential starting
  // from 0
  num_parameters_ = 0;
  for (const XlaCompiler::Argument& arg : args_) {
    if (arg.parameter < 0) continue;
    if (num_parameters_ != arg.parameter) {
      return errors::InvalidArgument(
          "Parameter numbers to JIT compilation are not consecutive starting "
          "from 0");
    }
    ++num_parameters_;

    if (arg.shape.num_elements() == 0) {
      return errors::InvalidArgument(
          "Non-constant argument must have a non-zero number of elements.");
    }
  }
  if (num_parameters_ == 0) return Status::OK();

  parameters_.resize(num_parameters_);

  std::vector<xla::Shape> parameter_shapes(num_parameters_);
  for (int i = 0; i < args_.size(); ++i) {
    const XlaCompiler::Argument& arg = args_[i];
    if (arg.parameter < 0) continue;
    // Computes the shapes of non-constant arguments.
    xla::PrimitiveType type;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(arg.type, &type));
    xla::ShapeUtil::PopulateShape(type, arg.shape.dim_sizes(),
                                  &parameter_shapes[arg.parameter]);
  }

  if (use_tuple_arg_ && num_parameters_ > 0) {
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(parameter_shapes);
    xla::ComputationDataHandle tuple =
        builder().Parameter(0, tuple_shape, "arg_tuple");
    for (int i = 0; i < args_.size(); ++i) {
      const XlaCompiler::Argument& arg = args_[i];
      if (arg.parameter < 0) continue;
      parameters_[arg.parameter] =
          builder().GetTupleElement(tuple, arg.parameter);
    }
  } else {
    for (int i = 0; i < args_.size(); ++i) {
      const XlaCompiler::Argument& arg = args_[i];
      if (arg.parameter < 0) continue;
      parameters_[arg.parameter] =
          builder().Parameter(arg.parameter, parameter_shapes[arg.parameter],
                              strings::StrCat("arg", i));
    }
  }
  return Status::OK();
}

Status XlaContext::CollectResults(
    xla::Computation* computation, bool* requires_runtime_context,
    std::vector<ConstRetVal>* compile_time_constants,
    int* num_nonconst_outputs) {
  mutex_lock l(mu_);

  xla::ComputationDataHandle handle;
  if (retval_.empty() && has_side_effects_) {
    // Build a empty tuple return value for computations that have side effects
    // but have no return values.
    handle = builder().Tuple({});
  } else if (retval_.size() == 1) {
    handle = retval_[0].second;

    // TODO(b/31775371): to workaround bug, add a no-op computation that is
    // guaranteed to be constructed after all of the formal parameters to the
    // computation.
    handle = builder().GetTupleElement(builder().Tuple({handle}), 0);

    // Ensure that the retval is returned even if another computation
    // was mistakenly placed on the ComputationBuilder.
    TF_CHECK_OK(builder().SetReturnValue(handle));
  } else if (retval_.size() > 1) {
    // There is at least one data-dependent expression: combine them
    // into a Tuple in index order before compiling.
    VLOG(1) << "Making the retval tuple.";
    std::sort(retval_.begin(), retval_.end(),
              [](const std::pair<int, xla::ComputationDataHandle>& a,
                 const std::pair<int, xla::ComputationDataHandle>& b) {
                return a.first < b.first;
              });
    std::vector<xla::ComputationDataHandle> elems;
    elems.reserve(retval_.size());
    for (const std::pair<int, xla::ComputationDataHandle>& r : retval_) {
      elems.push_back(r.second);
    }
    // Make a tuple from the vector of handles.
    handle = builder().Tuple(elems);
  }

  if (handle.handle() > 0) {
    // Builds the XLA computation.
    xla::StatusOr<xla::Computation> computation_status = builder().Build();
    if (!computation_status.ok()) {
      return computation_status.status();
    }
    *computation = computation_status.ConsumeValueOrDie();
  }

  // Make sure the compile time constants are in RetVal index order.
  std::sort(compile_time_constant_.begin(), compile_time_constant_.end(),
            [](const ConstRetVal& a, const ConstRetVal& b) {
              return a.index < b.index;
            });

  // Fill in the result details and return.
  *compile_time_constants = std::move(compile_time_constant_);
  *requires_runtime_context = has_context_parameter_;
  *num_nonconst_outputs = retval_.size();
  return Status::OK();
}

XlaContext::XlaContext(XlaCompiler* compiler, xla::Client* client,
                       const string& computation_name,
                       bool allow_cpu_custom_calls,
                       bool resolve_compile_time_constants)
    : compiler_(compiler),
      xla_builder_(client, computation_name),
      allow_cpu_custom_calls_(allow_cpu_custom_calls),
      resolve_compile_time_constants_(resolve_compile_time_constants) {}

const xla::ComputationDataHandle&
XlaContext::GetOrCreateRuntimeContextParameter() {
  mutex_lock lock(mu_);
  CHECK(allow_cpu_custom_calls_);
  CHECK(!use_tuple_arg_);
  if (has_context_parameter_) return context_parameter_;
  has_context_parameter_ = true;
  context_parameter_ = xla_builder_.Parameter(
      num_parameters_, xla::ShapeUtil::MakeOpaqueShape(), "tf_context");
  return context_parameter_;
}

string XlaContext::DebugString() { return "TLA JIT context"; }

// This is called by the Retval Op to associate a computed value
// with a specific return value of the subgraph.
void XlaContext::AddRetval(int retval_index,
                           const xla::ComputationDataHandle& handle) {
  VLOG(1) << "Added retval index " << retval_index << " to XLA computation";
  // Add the return value to the list being built up. The executor
  // is multi-threaded so this has to happen under the
  // lock.
  mutex_lock l(mu_);
  retval_.emplace_back(retval_index, handle);
}

Status XlaContext::AddConstRetval(int retval_index, DataType dtype,
                                  const xla::Literal& literal) {
  VLOG(1) << "Adding retval index " << retval_index
          << " with non-data-dependent tensor to XLA computation";
  if (resolve_compile_time_constants_) {
    ConstRetVal value;
    value.index = retval_index;
    TF_RETURN_IF_ERROR(LiteralToHostTensor(literal, dtype, &value.value));
    mutex_lock l(mu_);
    compile_time_constant_.push_back(std::move(value));
  } else {
    mutex_lock l(mu_);
    retval_.emplace_back(retval_index, xla_builder_.ConstantLiteral(literal));
  }
  return Status::OK();
}

void XlaContext::AddSideEffects() {
  mutex_lock lock(mu_);
  has_side_effects_ = true;
}

/* static */ const XlaExpression* XlaContext::CastExpressionFromTensor(
    const Tensor& tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor.tensor_data().data());
  CHECK_NE(expression->handle().handle(), 0);
  VLOG(1) << "Fetched T" << expression->handle().handle();
  return expression;
}

/* static */ XlaExpression* XlaContext::CastExpressionFromUninitializedTensor(
    Tensor* tensor) {
  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
  CHECK_EQ(expression->handle().handle(), 0);
  return const_cast<XlaExpression*>(expression);
}

/* static */ const XlaExpression* XlaContext::GetExpressionFromTensor(
    const Tensor& tensor) {
  return CastExpressionFromTensor(tensor);
}

/* static */ const xla::ComputationDataHandle&
XlaContext::GetComputationFromTensor(const Tensor& tensor) {
  return CastExpressionFromTensor(tensor)->handle();
}

xla::ComputationBuilder& XlaContext::builder() { return xla_builder_; }

const xla::Computation* XlaContext::GetOrCreateMax(const DataType type) {
  return LookupOrCreate(type, &max_func_, [this, type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Max() for " << type_string;
    xla::ComputationBuilder b(builder().client(), "max<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x = b.Parameter(0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y = b.Parameter(1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    b.Max(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::Computation* XlaContext::GetOrCreateAdd(const DataType type) {
  return LookupOrCreate(type, &add_func_, [this, type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Add() for " << type_string;
    xla::ComputationBuilder b(builder().client(), "add<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x = b.Parameter(0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y = b.Parameter(1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    b.Add(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::Computation* XlaContext::GetOrCreateSigmoid(const DataType type) {
  return LookupOrCreate(type, &sigmoid_func_, [this, type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Sigmoid() for " << type_string;
    xla::ComputationBuilder b(builder().client(),
                              "sigmoid<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x = b.Parameter(0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto one = b.ConstantLiteral(xla::LiteralUtil::One(xla_type));
    auto minus_one = b.Neg(one);
    b.Div(one, b.Add(b.Exp(b.Mul(x, minus_one)), one));
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::Computation* XlaContext::LookupOrCreate(
    DataType type, ComputationMap* out,
    const std::function<xla::Computation()>& create) {
  {
    mutex_lock l(mu_);
    const auto& entry = (*out)[type];
    if (!entry.IsNull()) {
      return &entry;
    }
  }
  auto new_entry = create();
  {
    mutex_lock l(mu_);
    // Somebody else might have made one concurrently.
    auto& entry = (*out)[type];
    if (entry.IsNull()) {
      entry = std::move(new_entry);
    }
    return &entry;
  }
}

}  // end namespace tensorflow
