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

#include <memory>
#include <unordered_map>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/experimental/saved_model/core/tf_concrete_function_test_protos.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

class SavedConcreteFunctionLoadingTest : public ::testing::Test {
 public:
  SavedConcreteFunctionLoadingTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

class DummyCapture : public TensorHandleConvertible {
 public:
  DummyCapture(ImmediateExecutionContext* ctx, int8_t value)
      : TensorHandleConvertible(
            testing::CreateTensorHandle(ctx, DT_FLOAT, {2, 4}, value)) {}
};

FunctionDef FuncDefWithNumInputsOutputs(int num_inputs, int num_outputs) {
  FunctionDef func;
  OpDef* signature = func.mutable_signature();
  for (int i = 0; i < num_inputs; ++i) {
    signature->add_input_arg();
  }
  for (int i = 0; i < num_outputs; ++i) {
    signature->add_output_arg();
  }
  return func;
}

// A SavedConcreteFunction whose canonicalized input signature
// has less inputs than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooFewInputsInSavedConcreteFunction) {
  // `saved` has 1 input
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();

  // `func` has 2 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(2, 0);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose canonicalized input signature length +
// captures is less than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooFewInputsWithCapturesInSavedConcreteFunction) {
  // `saved` has 1 input, and 1 capture, for a total of 2 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  saved.add_bound_inputs(5);

  // `func` has 3 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 0);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                         context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose canonicalized input signature
// has more inputs than its corresponding FunctionDef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooManyInputsInSavedConcreteFunction) {
  // `saved` has 3 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();

  // `func` has 2 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(2, 0);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose canonicalized input signature
// has the same number of inputs than its corresponding FunctionDef, but has
// additional captures should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooManyInputsWithCaptureInSavedConcreteFunction) {
  // `saved` has 3 inputs, and 1 capture, for a total of 4 inputs.
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  saved.add_bound_inputs(5);

  // `func` has 3 inputs.
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 0);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                         context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose capture refers to an index not in the capture
// map should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, ImproperCaptureIndex) {
  // `saved` has 3 inputs, 1 capture, for a total of 4 inputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ThreeArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  // Capture is at index "10"
  saved.add_bound_inputs(10);

  // `func` has 4 inputs
  FunctionDef func = FuncDefWithNumInputsOutputs(4, 0);

  // `captures` only has a capture for index 5
  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                         context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose outputs are fewer than its corresponding
// functiondef should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest, TooFewOutputsInSavedConcreteFunction) {
  // `saved` has 0 inputs, 1 output
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ZeroArgInputSignature();
  *saved.mutable_output_signature() = testing::SingleReturnOutputSignature();

  // `func` has 0 inputs, 2 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(0, 2);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose outputs exceed its corresponding functiondef
// should cause an error.
TEST_F(SavedConcreteFunctionLoadingTest,
       TooManyOutputsInSavedConcreteFunction) {
  // `saved` has 1 input, 3 outputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ThreeReturnOutputSignature();

  // `func` has 1 input, 2 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(1, 2);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status =
      internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
}

// A SavedConcreteFunction whose (inputs + captures) = functiondef inputs,
// and whose outputs = functiondef outputs should successfully load.
TEST_F(SavedConcreteFunctionLoadingTest, SuccessfulLoad) {
  // `saved` has 1 input, 2 captures, 3 outputs
  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::SingleArgInputSignature();
  *saved.mutable_output_signature() = testing::ThreeReturnOutputSignature();
  saved.add_bound_inputs(2);
  saved.add_bound_inputs(5);

  // `func` has 3 inputs, 3 outputs
  FunctionDef func = FuncDefWithNumInputsOutputs(3, 3);

  std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>> captures;
  captures[2] = std::make_unique<DummyCapture>(context(), 1);
  captures[5] = std::make_unique<DummyCapture>(context(), 10);

  std::unique_ptr<TFConcreteFunction> result;
  absl::Status status = internal::LoadTFConcreteFunction(saved, &func, captures,
                                                         context(), &result);
  TF_EXPECT_OK(status) << status.message();
}

// A TFConcreteFunction should register functiondefs on creation, and
// remove them upon deletion.
TEST_F(SavedConcreteFunctionLoadingTest, RegistersAndRemovesFunctionDefs) {
  std::string func_name = "FooBarBazWombatFunction";

  SavedConcreteFunction saved;
  *saved.mutable_canonicalized_input_signature() =
      testing::ZeroArgInputSignature();
  *saved.mutable_output_signature() = testing::ZeroReturnOutputSignature();
  FunctionDef func = FuncDefWithNumInputsOutputs(0, 0);
  *func.mutable_signature()->mutable_name() = func_name;

  {
    std::unique_ptr<TFConcreteFunction> result;
    absl::Status status =
        internal::LoadTFConcreteFunction(saved, &func, {}, context(), &result);
    TF_EXPECT_OK(status) << status.message();
    // The function should be registered with context.
    EXPECT_TRUE(context()->FindFunctionByName(func_name));
  }

  // After `result's` destructor runs, the function should no longer be
  // registered with context.
  EXPECT_FALSE(context()->FindFunctionByName(func_name));
}

}  // namespace
}  // namespace tensorflow
