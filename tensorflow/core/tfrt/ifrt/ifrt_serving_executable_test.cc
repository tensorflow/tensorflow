/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable_test_util.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using tensorflow::ifrt_serving::test_utils::GetMlirModulePath;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;
using ::testing::Return;

struct VariableInputTestParam {
  std::vector<tensorflow::Tensor> in_tensors;
  std::vector<bool>
      is_variable;  // if is_variable[i] = true, then in_tensor[i] is a variable
                    // and can be preloaded as an ifrt array.
  std::vector<tensorflow::Tensor> expected_out_tensors;
};
using VariableInputTest = ::testing::TestWithParam<VariableInputTestParam>;

class IfrtServingExecutableTest : public ::testing::Test {
 protected:
  explicit IfrtServingExecutableTest() {
    helper_ = std::make_unique<test_utils::IfrtServingExecutableTestHelper>(
        &selector_);
  }

  tsl::test_util::MockServingDeviceSelector selector_;
  std::unique_ptr<test_utils::IfrtServingExecutableTestHelper> helper_;
};

TEST_F(IfrtServingExecutableTest, Basic) {
  int64_t program_id = 123456;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillOnce(Return(tsl::DeviceReservation(0, /*selector=*/nullptr)));
  auto executable =
      helper_->MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK(executable->Execute(absl::MakeSpan(inputs), {}).status());
  }
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(absl::MakeSpan(inputs), {}));
  const auto expected_out =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_F(IfrtServingExecutableTest, MultipleShapes) {
  int64_t program_id = 123456;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(6)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });
  auto executable =
      helper_->MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  auto x1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  const auto expected_out1 =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));
  std::vector<tensorflow::Tensor> inputs1{x1, y1};

  auto x2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({1, 4}));
  auto y2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({4, 1}));
  const auto expected_out2 =
      AsTensor<int32_t>({30}, tensorflow::TensorShape({1, 1}));

  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs1, outputs2;
  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK(executable->Execute(absl::MakeSpan(inputs1), {}).status());
  }
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK_AND_ASSIGN(outputs1,
                            executable->Execute(absl::MakeSpan(inputs1), {}));
    TF_ASSERT_OK_AND_ASSIGN(outputs2,
                            executable->Execute(absl::MakeSpan(inputs2), {}));
  }

  ASSERT_EQ(executable->num_executables(), 2);

  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  EXPECT_THAT(outputs2, ElementsAre(TensorEq(expected_out2)));
}

TEST_F(IfrtServingExecutableTest, ReturnFailOnUncompiledShapeAfterFrozen) {
  int64_t program_id = 123456;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(3)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });
  auto executable =
      helper_->MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  auto x1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  const auto expected_out1 =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));
  std::vector<tensorflow::Tensor> inputs1{x1, y1};
  std::vector<tensorflow::Tensor> outputs1;
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK(executable->Execute(absl::MakeSpan(inputs1), {}).status());
  }
  TF_ASSERT_OK_AND_ASSIGN(outputs1,
                          executable->Execute(absl::MakeSpan(inputs1), {}));

  // Freeze the model
  executable->Freeze();

  // After the freeze(), already compiled shape works ok, but uncompiled shape
  // shall return failure.
  outputs1.clear();
  TF_ASSERT_OK_AND_ASSIGN(outputs1,
                          executable->Execute(absl::MakeSpan(inputs1), {}));
  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  auto x2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({1, 4}));
  auto y2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({4, 1}));
  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs2;
  auto status = executable->Execute(absl::MakeSpan(inputs2), {});

  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(IfrtServingExecutableTest, Spmd) {
  int64_t program_id = 111111;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id))).Times(0);
  auto executable = helper_->MakeExecutable(
      program_id, GetMlirModulePath("spmd_executable.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                             tensorflow::TensorShape({4, 2}));
  auto y = AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                             tensorflow::TensorShape({4, 2}));

  auto z = AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                             tensorflow::TensorShape({4, 2}));

  const auto expected_out = AsTensor<int32_t>({33, 36, 39, 42, 45, 48, 51, 54},
                                              tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_F(IfrtServingExecutableTest, SpmdTwoReturns) {
  int64_t program_id = 111111;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id))).Times(0);
  auto executable = helper_->MakeExecutable(
      program_id, GetMlirModulePath("spmd_executable_two_returns.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                             tensorflow::TensorShape({4, 2}));
  auto y = AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                             tensorflow::TensorShape({4, 2}));

  auto z = AsTensor<int32_t>({21, 22, 23, 24, 25, 26, 27, 28},
                             tensorflow::TensorShape({4, 2}));

  const auto expected_out0 = AsTensor<int32_t>({33, 36, 39, 42, 45, 48, 51, 54},
                                               tensorflow::TensorShape({4, 2}));
  const auto expected_out1 = AsTensor<int32_t>({20, 20, 20, 20, 20, 20, 20, 20},
                                               tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y, z};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST_F(IfrtServingExecutableTest, SpmdXlaCallModuleShardy) {
  int64_t program_id = 111111;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id))).Times(0);
  auto executable = helper_->MakeExecutable(
      program_id,
      GetMlirModulePath("spmd_executable_xla_call_module_shardy.mlir"));

  auto x = AsTensor<int32_t>({11, 12, 13, 14, 15, 16, 17, 18},
                             tensorflow::TensorShape({4, 2}));
  auto y = AsTensor<int32_t>({8, 7, 6, 5, 4, 3, 2, 1},
                             tensorflow::TensorShape({4, 2}));

  const auto expected_out0 = AsTensor<int32_t>({3, 5, 7, 9, 11, 13, 15, 17},
                                               tensorflow::TensorShape({4, 2}));
  const auto expected_out1 = AsTensor<int32_t>({19, 19, 19, 19, 19, 19, 19, 19},
                                               tensorflow::TensorShape({4, 2}));

  std::vector<tensorflow::Tensor> inputs{x, y};

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST_F(IfrtServingExecutableTest, NoReturn) {
  int64_t program_id = 111111;
  EXPECT_CALL(selector_, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });
  auto executable = helper_->MakeExecutable(
      program_id, GetMlirModulePath("executable_no_return.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};
  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK(executable->Execute(absl::MakeSpan(inputs), {}).status());
  }

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable->Execute(absl::MakeSpan(inputs), {}));

  ASSERT_EQ(result.size(), 0);
}

TEST_P(VariableInputTest, InterleaveVariable) {
  tsl::test_util::MockServingDeviceSelector device_selector;
  test_utils::IfrtServingExecutableTestHelper helper(&device_selector);
  int64_t program_id = 111111;
  EXPECT_CALL(device_selector, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillRepeatedly(
          [](::testing::Unused) { return tsl::DeviceReservation(0, nullptr); });
  auto executable = helper.MakeExecutable(
      program_id, GetMlirModulePath("executable_long_inputs.mlir"));
  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry =
      helper.ifrt_restore_tensor_registry();

  std::vector<tensorflow::Tensor> inputs;
  std::vector<int> loaded_variable_indices;
  for (int i = 0; i < GetParam().in_tensors.size(); i++) {
    if (GetParam().is_variable[i]) {
      auto [input_tensor_promise, input_tensor_future] =
          tsl::Future<tensorflow::Tensor>::MakePromise();
      IfrtRestoreTensorRegistry::RestoredTensorInfo restore_tensor_info = {
          .dtype_and_shape{.dtype = GetParam().in_tensors[i].dtype(),
                           .shape = GetParam().in_tensors[i].shape()},
          .tensor_future = input_tensor_future};
      std::string variable_name = absl::StrCat("variable_", i);
      ASSERT_OK(ifrt_restore_tensor_registry->TryRegister(variable_name,
                                                          restore_tensor_info));
      loaded_variable_indices.push_back(i);
      input_tensor_promise.Set(GetParam().in_tensors[i]);
      // Use string tensor containing the key (name) in place of variable
      // tensor.
      tensorflow::Tensor key_tensor(tensorflow::DT_STRING, {});
      key_tensor.scalar<tsl::tstring>()() = variable_name;
      inputs.push_back(key_tensor);
    } else {
      inputs.push_back(GetParam().in_tensors[i]);
    }
  }

  ASSERT_EQ(inputs.size(), GetParam().is_variable.size());
  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper.num_cores(); i++) {
    TF_ASSERT_OK(executable
                     ->Execute(absl::MakeSpan(inputs),
                               absl::MakeSpan(loaded_variable_indices))
                     .status());
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable->Execute(absl::MakeSpan(inputs),
                          absl::MakeSpan(loaded_variable_indices)));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(GetParam().expected_out_tensors[0]),
                          TensorEq(GetParam().expected_out_tensors[1]),
                          TensorEq(GetParam().expected_out_tensors[2])));
}

INSTANTIATE_TEST_SUITE_P(
    VariableInputTests, VariableInputTest,
    ::testing::ValuesIn<VariableInputTestParam>(
        {
            // Basic case: all variables or all non-variables.
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, true, true, true, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, false, false, false, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            // Variable and non-variables are non-interleaved
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, false, false, true, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, true, false, false, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            // Variable and non-variables are interleaved
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {true, false, false, true, false},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
            {
                .in_tensors =
                    {
                        AsTensor<int32_t>({2, 2}, TensorShape({1, 2})),
                        AsTensor<int32_t>({3, 3}, TensorShape({2, 1})),
                        AsTensor<int32_t>({4, 4}, TensorShape({1, 2})),
                        AsTensor<int32_t>({5, 5}, TensorShape({2, 1})),
                        AsTensor<int32_t>({10, 10}, TensorShape({1, 2})),
                    },
                .is_variable = {false, true, true, false, true},
                .expected_out_tensors =
                    {
                        AsTensor<int32_t>({12}, TensorShape({1, 1})),
                        AsTensor<int32_t>({40}, TensorShape({1, 1})),
                        AsTensor<int32_t>({100}, TensorShape({1, 1})),
                    },
            },
        }));

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
