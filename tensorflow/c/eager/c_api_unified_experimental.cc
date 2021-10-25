/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api_unified_experimental.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;

namespace tensorflow {
namespace tracing {
typedef absl::flat_hash_map<std::string, tracing::FactoryFunction> FactoriesMap;

static FactoriesMap& GetFactories() {
  static FactoriesMap* factories = new FactoriesMap;
  return *factories;
}

static tracing::FactoryFunction default_factory;

void RegisterTracingEngineFactory(const string& name, FactoryFunction factory) {
  assert((!GetFactories().count(name)) ||
         (GetFactories()[name] == factory) &&
             "Duplicate tracing factory registration");
  GetFactories()[name] = factory;
}

Status SetDefaultTracingEngine(const char* name) {
  auto entry = GetFactories().find(name);
  if (entry != GetFactories().end()) {
    default_factory = GetFactories().find(name)->second;
    return Status::OK();
  }
  string msg = absl::StrCat(
      "No tracing engine factory has been registered with the key '", name,
      "' (available: ");
  // Ensure deterministic (sorted) order in the error message
  std::set<string> factories_sorted;
  for (const auto& factory : GetFactories())
    factories_sorted.insert(factory.first);
  const char* comma = "";
  for (const string& factory : factories_sorted) {
    msg += comma + factory;
    comma = ", ";
  }
  msg += ")";

  return errors::InvalidArgument(msg.c_str());
}

static TracingContext* CreateTracingExecutionContext(const char* fn_name,
                                                     TF_Status* s) {
  if (default_factory) {
    return default_factory(fn_name, s);
  }
  Set_TF_Status_from_Status(
      s, errors::FailedPrecondition("default_factory is nullptr"));
  return nullptr;
}

}  // end namespace tracing
}  // end namespace tensorflow

// =============================================================================
// Public C API entry points
//
// These are only the generic entry points for the C API. This file does not
// have any visibility into the graph/eager implementation and is only providing
// C bindings to the abstract classes defined in the
// c_api_unified_experimental_internal.h header.
//
// =============================================================================

using tensorflow::AbstractFunction;
using tensorflow::AbstractTensorHandle;
using tensorflow::DataType;
using tensorflow::dyn_cast;
using tensorflow::OutputList;
using tensorflow::Status;
using tensorflow::unwrap;
using tensorflow::wrap;
using tensorflow::tracing::CreateTracingExecutionContext;
using tensorflow::tracing::SetDefaultTracingEngine;
using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

void TF_SetTracingImplementation(const char* name, TF_Status* s) {
  Set_TF_Status_from_Status(s, SetDefaultTracingEngine(name));
}

// Creates a new TensorFlow function, it is an execution context attached to a
// given tracing context.
TF_ExecutionContext* TF_CreateFunction(const char* fn_name, TF_Status* s) {
  return wrap(CreateTracingExecutionContext(fn_name, s));
}

TF_AbstractFunction* TF_FinalizeFunction(TF_ExecutionContext* ctx,
                                         TF_OutputList* outputs, TF_Status* s) {
  AbstractFunction* func;
  TracingContext* tracing_ctx = dyn_cast<TracingContext>(unwrap(ctx));
  if (!tracing_ctx) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "Only TracingContext can be converted into a function."));
    return nullptr;
  }
  Set_TF_Status_from_Status(s, tracing_ctx->Finalize(unwrap(outputs), &func));
  TF_DeleteExecutionContext(ctx);
  return wrap(func);
}

TF_AbstractTensor* TF_AddFunctionParameter(TF_ExecutionContext* func,
                                           TF_DataType dtype, TF_Shape shape,
                                           TF_Status* s) {
  DCHECK_GE(shape.num_dims, -1);
  TracingTensorHandle* t;
  TracingContext* tracing_ctx = dyn_cast<TracingContext>(unwrap(func));
  if (!tracing_ctx) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "TF_AddFunctionParameter must be called on a TracingContext."));
    return nullptr;
  }
  tensorflow::PartialTensorShape partial_shape;
  if (shape.num_dims != -1) {
    DCHECK(shape.dim_sizes != nullptr);
    Status status = tensorflow::PartialTensorShape::MakePartialShape(
        reinterpret_cast<int64_t*>(shape.dim_sizes), shape.num_dims,
        &partial_shape);
    if (!status.ok()) {
      Set_TF_Status_from_Status(s, status);
      return nullptr;
    }
  }
  Set_TF_Status_from_Status(
      s, tracing_ctx->AddParameter(static_cast<DataType>(dtype), partial_shape,
                                   &t));
  return wrap(t);
}

void TF_DeleteExecutionContext(TF_ExecutionContext* c) { unwrap(c)->Release(); }

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* c) {
  return wrap((unwrap(c)->CreateOperation()));
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) { unwrap(op)->Release(); }

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) { unwrap(t)->Unref(); }

TF_OutputList* TF_NewOutputList() { return wrap(new OutputList); }
void TF_DeleteOutputList(TF_OutputList* o) { delete unwrap(o); }
void TF_OutputListSetNumOutputs(TF_OutputList* o, int num_outputs,
                                TF_Status* s) {
  unwrap(o)->expected_num_outputs = num_outputs;
  unwrap(o)->outputs.clear();
  unwrap(o)->outputs.resize(num_outputs);
}
int TF_OutputListNumOutputs(TF_OutputList* o) {
  return unwrap(o)->outputs.size();
}
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i) {
  return wrap(unwrap(o)->outputs[i]);
}
void TF_OutputListPushBack(TF_OutputList* o, TF_AbstractTensor* tensor,
                           TF_Status* s) {
  unwrap(o)->outputs.push_back(unwrap(tensor));
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
  Set_TF_Status_from_Status(s, unwrap(op)->Reset(op_type,
                                                 /*raw_device_name=*/nullptr));
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
  TracingOperation* tracing_op = dyn_cast<TracingOperation>(unwrap(op));
  if (!tracing_op) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "TF_AbstractOpSetOpName must be called on a TracingOperation."));
    return;
  }
  Set_TF_Status_from_Status(s, tracing_op->SetOpName(op_name));
}

void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s) {
  Status status =
      unwrap(op)->SetAttrType(attr_name, static_cast<DataType>(value));
  TF_SetStatus(s, static_cast<TF_Code>(status.code()),
               status.error_message().c_str());
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_Status* s) {
  for (int i = 0; i < num_inputs; i++) {
    Set_TF_Status_from_Status(s, unwrap(op)->AddInput(unwrap(inputs[i])));
    if (TF_GetCode(s) != TF_OK) {
      return;
    }
  }
  int num_outputs = unwrap(o)->expected_num_outputs;
  Set_TF_Status_from_Status(
      s, unwrap(op)->Execute(
             absl::MakeSpan(reinterpret_cast<AbstractTensorHandle**>(
                                unwrap(o)->outputs.data()),
                            unwrap(o)->outputs.size()),
             &num_outputs));
}

void TF_DeleteAbstractFunction(TF_AbstractFunction* func) {
  unwrap(func)->Unref();
}

void TF_ExecutionContextRegisterFunction(TF_ExecutionContext* ctx,
                                         TF_AbstractFunction* func,
                                         TF_Status* s) {
  Set_TF_Status_from_Status(s, unwrap(ctx)->RegisterFunction(unwrap(func)));
}
