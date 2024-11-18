// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expruns or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TEST_UTIL_H_

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_convert_types_impl.h"

namespace litert {
namespace testing {

using ::litert::example::ExampleConvertTensor;
using ::litert::example::MakeExampleOpFinalizer;
using ::litert::example::MakeExampleTensorFinalizer;

// Wraps an allocated LiteRtOp.
class TestOpContext {
 public:
  using TensorArg = std::initializer_list<absl::string_view>;

  // Build an LiteRtOp and input/output tensors with only their names.
  TestOpContext(LiteRtOpCode code, TensorArg inputs, TensorArg outputs) {
    op_.op_code = code;
    auto input_it = inputs.begin();
    for (int i = 0; i < inputs.size(); ++i) {
      auto& input = tensors_.emplace_back();
      input.name.assign(input_it->begin(), input_it->end());
      input.user_arg_inds.push_back(i);
      input.users.push_back(&op_);
      op_.inputs.push_back(&input);
      input_it++;
    }
    auto output_it = outputs.begin();
    for (int i = 0; i < outputs.size(); ++i) {
      auto& output = tensors_.emplace_back();
      output.name.assign(output_it->begin(), output_it->end());
      output.user_arg_inds.push_back(i);
      output.users.push_back(&op_);
      op_.outputs.push_back(&output);
      output_it++;
    }
  }

  // Get the op.
  Op GetOp() { return Op(&op_); }

 private:
  LiteRtOpT op_;
  SmallVec<LiteRtTensorT> tensors_;
};

// FinalizingLegalization for example ops/tensors. Users won't have to define
// these this is simply to make testing easier.
using ExampleScopedFinalizingLegalization =
    ScopedFinalizingLegalization<example::ExampleOp, example::ExampleTensor>;

// Create an example FinalizingLegalization from the existing example
// legalization and above finalizers.
template <LiteRtOpCode OpCode = kLiteRtOpCodeTflCustom>
ExampleScopedFinalizingLegalization::Ptr
CreateExampleScopedFinalizingLegalization(
    example::ExampleGraphContext& graph_context,
    ExampleScopedFinalizingLegalization::SharedScope shared_scope) {
  return ExampleScopedFinalizingLegalization::Create(
      example::ExampleOpLegalization<OpCode>::Create(),
      example::MakeExampleTensorFinalizer(graph_context),
      example::MakeExampleOpFinalizer(graph_context), shared_scope);
}

// Wrapper around tensor map scope and tensor finalizer/converter. Used to setup
// scope manually.
class TestExampleScopeContext {
  using Legalisation = ExampleScopedFinalizingLegalization;
  using Scope = Legalisation::Scope;
  using SharedScope = Legalisation::SharedScope;
  using TensorKonverter = Legalisation::TensorKonverter;
  using TensorFinaliser = Legalisation::TensorFinaliser;

 public:
  TestExampleScopeContext(TensorKonverter tensor_converter,
                          TensorFinaliser tensor_finalizer)
      : tensor_converter_(tensor_converter),
        tensor_finalizer_(tensor_finalizer),
        shared_scope_(std::make_shared<Scope>()) {}

  // Get the current scope.
  SharedScope GetScope() { return shared_scope_; }

  // Convert, finalize and add resultant tensor to scope.
  LiteRtStatus Push(const Tensor& litert_tensor) {
    auto backend_tensor = tensor_converter_(litert_tensor);
    if (!backend_tensor) {
      return backend_tensor.Error().Status();
    }
    LITERT_RETURN_STATUS_IF_NOT_OK(tensor_finalizer_(*backend_tensor));
    return PushToScope(litert_tensor, *backend_tensor);
  }

 private:
  // Push converted tensor to scope.
  LiteRtStatus PushToScope(const Tensor& litert_tensor,
                           Legalisation::Tenser backend_tensor) {
    return Legalisation::DoPushToScope(*GetScope(), litert_tensor,
                                       backend_tensor);
  }

  TensorKonverter tensor_converter_;
  TensorFinaliser tensor_finalizer_;
  Legalisation::SharedScope shared_scope_;
};

// All the context needed for testing scope finalizing legalizations.
template <LiteRtOpCode OpCode = kLiteRtOpCodeTflCustom>
struct ConversionTestContext {
 private:
  using Legalisation = ExampleScopedFinalizingLegalization;
  using WrappedLegalization = example::ExampleOpLegalization<OpCode>;

 public:
  // Construct test context with graph context and default example
  // converters/finalizers.
  ConversionTestContext()
      : scope_context(ExampleConvertTensor,
                      MakeExampleTensorFinalizer(graph_context)),
        legalization(WrappedLegalization::Create(),
                     MakeExampleTensorFinalizer(graph_context),
                     MakeExampleOpFinalizer(graph_context),
                     scope_context.GetScope()) {}

  // Graph context referenced in conversion operations under test.
  example::ExampleGraphContext graph_context;

  // Scope map wrapper.
  TestExampleScopeContext scope_context;

  // The scoped finalizing legalization under test.
  Legalisation legalization;
};

}  // namespace testing
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TEST_UTIL_H_
