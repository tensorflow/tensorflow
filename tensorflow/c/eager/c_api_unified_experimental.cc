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
#include "tensorflow/core/platform/types.h"

using tensorflow::string;
using tensorflow::internal::OutputList;
using tensorflow::internal::unwrap;

namespace tensorflow {
namespace internal {
typedef absl::flat_hash_map<std::string, FactoryFunction> FactoriesMap;

static FactoriesMap& GetFactories() {
  static FactoriesMap* factories = new FactoriesMap;
  return *factories;
}

static const char* default_factory = "<unset>";

void RegisterTracingEngineFactory(const string& name, FactoryFunction factory) {
  assert((!GetFactories().count(name)) ||
         (GetFactories()[name] == factory) &&
             "Duplicate tracing factory registration");
  GetFactories()[name] = factory;
}

void SetDefaultTracingEngine(const char* name) { default_factory = name; }

static ExecutionContext* CreateTracingExecutionContext(const char* fn_name,
                                                       TF_Status* s) {
  auto entry = GetFactories().find(default_factory);
  if (entry != GetFactories().end()) return entry->second(fn_name, s);
  string msg = absl::StrCat(
      "No tracing engine factory has been registered with the key '",
      default_factory, "' (available: ");
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

  TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
  return nullptr;
}

}  // end namespace internal
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

void TF_SetTracingImplementation(const char* name) {
  tensorflow::internal::SetDefaultTracingEngine(name);
}

// Creates a new TensorFlow function, it is an execution context attached to a
// given tracing context.
TF_ExecutionContext* TF_CreateFunction(const char* fn_name, TF_Status* s) {
  return wrap(tensorflow::internal::CreateTracingExecutionContext(fn_name, s));
}

TF_AbstractFunction* TF_FinalizeFunction(TF_ExecutionContext* ctx,
                                         TF_OutputList* outputs, TF_Status* s) {
  auto* func = wrap(unwrap(ctx)->Finalize(unwrap(outputs), s));
  TF_DeleteExecutionContext(ctx);
  return func;
}

TF_AbstractTensor* TF_AddFunctionParameter(TF_ExecutionContext* func,
                                           TF_DataType dtype, TF_Status* s) {
  return wrap(unwrap(func)->AddParameter(dtype, s));
}

void TF_DeleteExecutionContext(TF_ExecutionContext* c) { delete unwrap(c); }

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* c) {
  return wrap(unwrap(c)->CreateOperation());
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) { delete unwrap(op); }

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) { delete unwrap(t); }

TF_OutputList* TF_NewOutputList() { return wrap(new OutputList); }
void TF_DeleteOutputList(TF_OutputList* o) { delete unwrap(o); }
void TF_OutputListSetNumOutputs(TF_OutputList* o, int num_outputs,
                                TF_Status* s) {
  unwrap(o)->expected_num_outputs = num_outputs;
}
int TF_OutputListNumOutputs(TF_OutputList* o) {
  return unwrap(o)->outputs.size();
}
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i) {
  return wrap(unwrap(o)->outputs[i]);
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
  unwrap(op)->SetOpType(op_type, s);
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
  unwrap(op)->SetOpName(op_name, s);
}

void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s) {
  unwrap(op)->SetAttrType(attr_name, value, s);
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s) {
  unwrap(ctx)->ExecuteOperation(unwrap(op), num_inputs, &unwrap(*inputs),
                                unwrap(o), s);
}

void TF_DeleteAbstractFunction(TF_AbstractFunction* func) {
  delete unwrap(func);
}

void TF_ExecutionContextRegisterFunction(TF_ExecutionContext* ctx,
                                         TF_AbstractFunction* func,
                                         TF_Status* s) {
  unwrap(ctx)->RegisterFunction(unwrap(func), s);
}
