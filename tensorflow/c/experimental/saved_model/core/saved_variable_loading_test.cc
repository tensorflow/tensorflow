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
#include <vector>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

class SavedVariableLoadingTest : public ::testing::TestWithParam<
                                     std::tuple<DataType, std::vector<int64>>> {
 public:
  SavedVariableLoadingTest() {
    SessionOptions options;
    options.config.mutable_device_count()->insert({"CPU", 3});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    ctx_ = testing::CreateTestingEagerContext(device_mgr_.get());
  }

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// Sanity check that constructing a tensorflow::Variable from a SavedVariable
// 1. does not cause an error
// 2. preserves dtype and shape.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableSuccessful) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  TF_EXPECT_OK(internal::LoadSavedVariable(context(), saved_variable, &var));
  EXPECT_EQ(var->dtype(), dtype);
  EXPECT_EQ(var->shape(), shape);
}

// Verify that a device specified in the SavedVariable is kept.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableWithDevice) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  saved_variable.set_device("/job:localhost/replica:0/task:0/device:CPU:1"),
      shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  TF_ASSERT_OK(internal::LoadSavedVariable(context(), saved_variable, &var));
  EXPECT_EQ(down_cast<TensorHandle*>(var->handle())->resource_device()->name(),
            "/job:localhost/replica:0/task:0/device:CPU:1");
}

// Verify load failure if a non-existing device is specified.
TEST_P(SavedVariableLoadingTest, LoadSavedVariableWithInvalidDevice) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  TensorShape shape(std::get<1>(test_params));

  SavedVariable saved_variable;
  saved_variable.set_dtype(dtype);
  saved_variable.set_device("/job:localhost/replica:0/task:0/device:CPU:99"),
      shape.AsProto(saved_variable.mutable_shape());

  std::unique_ptr<Variable> var;
  ASSERT_NE(Status::OK(),
            internal::LoadSavedVariable(context(), saved_variable, &var));
}

// Assigning and reading values should yield
// consistent results.
TEST_P(SavedVariableLoadingTest, AssignAndReadVariableSuccesful) {
  auto& test_params = GetParam();
  DataType dtype = std::get<0>(test_params);
  std::vector<int64> shape_vector = std::get<1>(test_params);
  TensorShape shape(shape_vector);

  // Create the variable.
  Status status;
  std::unique_ptr<Variable> var;
  TF_EXPECT_OK(Variable::CreateUninitialized(context(), dtype, shape,
                                             absl::nullopt, nullptr, {}, &var));

  // Create a TensorHandle
  ImmediateTensorHandlePtr expected_handle =
      testing::CreateTensorHandle(context(), dtype, shape_vector, 42);
  AbstractTensorPtr expected_tensor(expected_handle->Resolve(&status));
  TF_EXPECT_OK(status) << status.error_message();

  // Assign the tensorhandle to the variable.
  TF_EXPECT_OK(var->Assign(expected_handle.get()));

  // Read back the value from the variable
  ImmediateTensorHandlePtr output_handle;
  TF_EXPECT_OK(var->ReadValue(&output_handle));
  AbstractTensorPtr output_tensor(output_handle->Resolve(&status));
  TF_EXPECT_OK(status) << status.error_message();

  // Check that output_tensor == expected_tensor
  EXPECT_EQ(output_tensor->Type(), expected_tensor->Type());
  EXPECT_EQ(output_tensor->NumElements(), expected_tensor->NumElements());
  testing::CheckBufferDataIsEqual(
      output_tensor->Type(), output_tensor->NumElements(),
      output_tensor->Data(), expected_tensor->Data());
}

// Test against combinations of SavedVariables of
// 1. Varying dtypes
// 2. Varying shapes
INSTANTIATE_TEST_SUITE_P(
    SavedVariableIntegerDtypesTest, SavedVariableLoadingTest,
    ::testing::Combine(
        ::testing::ValuesIn(testing::DataTypeSetToVector(kDataTypeIsInteger)),
        ::testing::ValuesIn(testing::InterestingShapes())));

INSTANTIATE_TEST_SUITE_P(
    SavedVariableFloatingDtypesTest, SavedVariableLoadingTest,
    ::testing::Combine(::testing::Values(DT_FLOAT, DT_DOUBLE),
                       ::testing::ValuesIn(testing::InterestingShapes())));

}  // namespace
}  // namespace tensorflow
