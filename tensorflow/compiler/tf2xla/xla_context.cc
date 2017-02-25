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

namespace tensorflow {

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

void XlaContext::set_args(std::vector<Argument> args) {
  args_ = std::move(args);
}

XlaContext::XlaContext(XlaCompiler* compiler, xla::ComputationBuilder* builder,
                       bool allow_cpu_custom_calls,
                       bool resolve_compile_time_constants)
    : compiler_(compiler),
      builder_(builder),
      allow_cpu_custom_calls_(allow_cpu_custom_calls),
      resolve_compile_time_constants_(resolve_compile_time_constants) {}

const xla::ComputationDataHandle&
XlaContext::GetOrCreateRuntimeContextParameter() {
  CHECK(allow_cpu_custom_calls_);
  if (has_context_parameter_) return context_parameter_;
  has_context_parameter_ = true;

  // Allocate the next available parameter for the context parameter.
  int num_parameters = 0;
  for (const Argument& arg : args_) {
    if (!arg.value.is_constant) {
      ++num_parameters;
    }
  }
  context_parameter_ = builder_->Parameter(
      num_parameters, xla::ShapeUtil::MakeOpaqueShape(), "tf_context");
  return context_parameter_;
}

string XlaContext::DebugString() { return "TLA JIT context"; }

// This is called by the Retval Op to associate a computed value
// with a specific return value of the subgraph.
void XlaContext::AddRetval(int retval_index,
                           const xla::ComputationDataHandle& handle) {
  VLOG(1) << "Added retval index " << retval_index << " to XLA computation";
  // Add the return value to the list being built up.
  if (retvals_.size() <= retval_index) {
    retvals_.resize(retval_index + 1);
  }
  retvals_[retval_index].is_constant = false;
  retvals_[retval_index].handle = handle;
}

Status XlaContext::AddConstRetval(int retval_index, DataType dtype,
                                  const xla::Literal& literal) {
  VLOG(1) << "Adding retval index " << retval_index
          << " with non-data-dependent tensor to XLA computation";
  if (retvals_.size() <= retval_index) {
    retvals_.resize(retval_index + 1);
  }
  if (resolve_compile_time_constants_) {
    retvals_[retval_index].is_constant = true;
    TF_RETURN_IF_ERROR(LiteralToHostTensor(
        literal, dtype, &retvals_[retval_index].constant_value));
  } else {
    retvals_[retval_index].is_constant = false;
    retvals_[retval_index].handle = builder_->ConstantLiteral(literal);
  }
  return Status::OK();
}

void XlaContext::AddSideEffects() {
  has_side_effects_ = true;
}

xla::ComputationBuilder* XlaContext::builder() { return builder_; }

Status XlaContext::CreateVariable(int variable_id, string name, DataType type,
                                  const xla::ComputationDataHandle& handle) {
  auto result = variables_.emplace(variable_id, Variable());
  if (!result.second) {
    return errors::InvalidArgument("Duplicate ID ", variable_id,
                                   " for variable ", name);
  }
  Variable& var = result.first->second;
  var.name = std::move(name);
  var.type = type;
  var.initial_value = var.value = handle;
  return Status::OK();
}

Status XlaContext::AssignVariable(int variable_id, DataType type,
                                  const xla::ComputationDataHandle& handle) {
  auto it = variables_.find(variable_id);
  if (it == variables_.end()) {
    return errors::InvalidArgument("Unknown variable ID ", variable_id);
  }
  Variable& var = it->second;
  if (!((var.type == DT_INVALID && type != DT_INVALID) || (var.type == type))) {
    return errors::InvalidArgument(
        "Types of variables cannot change after initialization: old type was ",
        DataTypeString(var.type), ", new type is ", DataTypeString(type));
  }
  var.type = type;
  var.value = handle;
  return Status::OK();
}

Status XlaContext::ReadVariable(int variable_id,
                                xla::ComputationDataHandle* handle) {
  auto it = variables_.find(variable_id);
  if (it == variables_.end()) {
    return errors::InvalidArgument("Unknown variable ID ", variable_id);
  }
  *handle = it->second.value;
  if (handle->handle() == 0) {
    return errors::InvalidArgument("Read of uninitialized variable ",
                                   it->second.name);
  }
  return Status::OK();
}

const xla::Computation* XlaContext::GetOrCreateMax(const DataType type) {
  return LookupOrCreate(type, &max_func_, [this, type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Max() for " << type_string;
    xla::ComputationBuilder b(builder()->client(), "max<" + type_string + ">");
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
    xla::ComputationBuilder b(builder()->client(), "add<" + type_string + ">");
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
    xla::ComputationBuilder b(builder()->client(),
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
    const auto& entry = (*out)[type];
    if (!entry.IsNull()) {
      return &entry;
    }
  }
  auto new_entry = create();
  {
    // Somebody else might have made one concurrently.
    auto& entry = (*out)[type];
    if (entry.IsNull()) {
      entry = std::move(new_entry);
    }
    return &entry;
  }
}

}  // namespace tensorflow
