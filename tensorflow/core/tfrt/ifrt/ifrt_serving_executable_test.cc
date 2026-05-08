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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable_test_util.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using tensorflow::ifrt_serving::test_utils::GetMlirModulePath;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;
using ::testing::Return;

// Helper to set up a mock expectation for `ReserveDevice`.
// It returns device reservations in a round-robin fashion, cycling through
// available cores.
void SetUpMockDeviceReservation(
    tsl::test_util::MockServingDeviceSelector& selector, int64_t program_id,
    int num_cores) {
  auto device_index = std::make_shared<int>(0);
  int num_cores_const = std::max(1, num_cores);
  EXPECT_CALL(selector, ReserveDevice(absl::StrCat(program_id)))
      .WillRepeatedly([device_index, num_cores_const](::testing::Unused) {
        return tsl::DeviceReservation((*device_index)++ % num_cores_const,
                                      /*selector=*/nullptr);
      });
}

// Mock class for TpuH2DTransferExecutor.
// By default, `ScheduledH2DTransfers` pads input tensors to their static shapes
// based on `ifrt_shape` before calling the base class method.
class MockH2DTransferExecutor : public H2DTransferExecutor {
 public:
  explicit MockH2DTransferExecutor(xla::ifrt::Client& client)
      : H2DTransferExecutor(client) {
    ON_CALL(*this, ScheduledH2DTransfers(_, _))
        .WillByDefault([this](absl::Span<const InputHandle> handles,
                              tsl::thread::ThreadPool& thread_pool)
                           -> absl::StatusOr<
                               tsl::Future<std::vector<xla::ifrt::ArrayRef>>> {
          std::vector<InputHandle> new_handles;
          new_handles.reserve(handles.size());
          for (const auto& handle : handles) {
            new_handles.push_back(handle);
            tensorflow::TensorShape static_shape;
            if (handle.input_xla_shape != nullptr) {
              static_shape =
                  tensorflow::TensorShape(handle.input_xla_shape->dimensions());
            } else if (handle.ifrt_shape != nullptr) {
              static_shape = tensorflow::TensorShape(handle.ifrt_shape->dims());
            } else {
              static_shape = handle.tensor.shape();
            }
            if (handle.tensor.shape() != static_shape) {
              tensorflow::Tensor padded_tensor(handle.tensor.dtype(),
                                               static_shape);
              tensorflow::tensor::DeepCopy(handle.tensor, &padded_tensor);
              new_handles.back().tensor = padded_tensor;
            }
          }
          return H2DTransferExecutor::ScheduledH2DTransfers(new_handles,
                                                            thread_pool);
        });
    ON_CALL(*this, RunH2DTransfers()).WillByDefault(Return(absl::OkStatus()));
  }

  MOCK_METHOD(absl::StatusOr<tsl::Future<std::vector<xla::ifrt::ArrayRef>>>,
              ScheduledH2DTransfers,
              (absl::Span<const InputHandle> handles,
               tsl::thread::ThreadPool& thread_pool),
              (override));
  MOCK_METHOD(absl::Status, RunH2DTransfers, (), (override));
};

class MockH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  MockH2DTransferExecutorFactory() {
    ON_CALL(*this, CreateH2DTransferExecutor(_))
        .WillByDefault([&](xla::ifrt::Client& client) {
          return std::make_unique<MockH2DTransferExecutor>(client);
        });
  }

  MOCK_METHOD(absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>,
              CreateH2DTransferExecutor, (xla::ifrt::Client & ifrt_client),
              (override));
};

struct VariableInputTestParam {
  std::vector<tensorflow::Tensor> in_tensors;
  std::vector<bool>
      is_variable;  // if is_variable[i] = true, then in_tensor[i] is a variable
                    // and can be preloaded as an ifrt array.
  std::vector<tensorflow::Tensor> expected_out_tensors;
};

using VariableInputTest =
    ::testing::TestWithParam<std::tuple<VariableInputTestParam, bool>>;

class IfrtServingExecutableTest : public ::testing::TestWithParam<bool> {
 protected:
  explicit IfrtServingExecutableTest() {
    helper_ = std::make_unique<test_utils::IfrtServingExecutableTestHelper>(
        &selector_);
  }

  absl::StatusOr<std::vector<tensorflow::Tensor>> Execute(
      IfrtServingExecutable* executable,
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices = {}) {
    if (GetParam()) {
      TF_ASSIGN_OR_RETURN(
          auto future, executable->ExecuteAsync(inputs, variable_arg_indices));
      return future.Await();
    } else {
      return executable->Execute(inputs, variable_arg_indices);
    }
  }

  tsl::test_util::MockServingDeviceSelector selector_;
  std::unique_ptr<test_utils::IfrtServingExecutableTestHelper> helper_;
};

INSTANTIATE_TEST_SUITE_P(IfrtServingExecutableTests, IfrtServingExecutableTest,
                         ::testing::Bool());

TEST_P(IfrtServingExecutableTest, Basic) {
  int64_t program_id = 123456;
  SetUpMockDeviceReservation(selector_, program_id, helper_->num_cores());
  auto executable =
      helper_->MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};

  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));
  const auto expected_out =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_P(IfrtServingExecutableTest, MultipleShapes) {
  int64_t program_id = 123456;
  SetUpMockDeviceReservation(selector_, program_id, helper_->num_cores());
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
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(executable.get(), absl::MakeSpan(inputs1), {}));
  }
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        outputs1, Execute(executable.get(), absl::MakeSpan(inputs1), {}));
    TF_ASSERT_OK_AND_ASSIGN(
        outputs2, Execute(executable.get(), absl::MakeSpan(inputs2), {}));
  }

  ASSERT_EQ(executable->num_executables(), 2);

  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  EXPECT_THAT(outputs2, ElementsAre(TensorEq(expected_out2)));
}

TEST_P(IfrtServingExecutableTest, ReturnFailOnUncompiledShapeAfterFrozen) {
  int64_t program_id = 123456;
  SetUpMockDeviceReservation(selector_, program_id, helper_->num_cores());
  auto executable =
      helper_->MakeExecutable(program_id, GetMlirModulePath("executable.mlir"));

  auto x1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y1 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  const auto expected_out1 =
      AsTensor<int32_t>({14}, tensorflow::TensorShape({1, 1}));
  std::vector<tensorflow::Tensor> inputs1{x1, y1};
  std::vector<tensorflow::Tensor> outputs1;
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(executable.get(), absl::MakeSpan(inputs1), {}));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      outputs1, Execute(executable.get(), absl::MakeSpan(inputs1), {}));

  // Freeze the model
  executable->Freeze();

  // After the freeze(), already compiled shape works ok, but uncompiled shape
  // shall return failure.
  outputs1.clear();
  TF_ASSERT_OK_AND_ASSIGN(
      outputs1, Execute(executable.get(), absl::MakeSpan(inputs1), {}));
  EXPECT_THAT(outputs1, ElementsAre(TensorEq(expected_out1)));

  auto x2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({1, 4}));
  auto y2 = AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({4, 1}));
  std::vector<tensorflow::Tensor> inputs2{x2, y2};

  std::vector<tensorflow::Tensor> outputs2;
  auto status = Execute(executable.get(), absl::MakeSpan(inputs2), {});

  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_P(IfrtServingExecutableTest, Spmd) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result, ElementsAre(TensorEq(expected_out)));
}

TEST_P(IfrtServingExecutableTest, SpmdTwoReturns) {
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST_P(IfrtServingExecutableTest, SpmdXlaCallModuleShardy) {
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));

  EXPECT_THAT(result,
              ElementsAre(TensorEq(expected_out0), TensorEq(expected_out1)));
}

TEST_F(IfrtServingExecutableTest, EncodeLayout) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  const char* const kMlirModuleStr = R"(
    module {
      func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }
    }
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlirModuleStr, &context);
  ASSERT_TRUE(module);

  // Create shapes with layout
  xla::Shape shape0 = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {2, 2}, {1, 0});
  xla::Shape shape1 = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {2, 2}, {0, 1});  // Different layout
  std::vector<xla::Shape> input_shapes = {shape0, shape1};

  TF_ASSERT_OK(EncodeLayout(absl::MakeSpan(input_shapes), *module));

  auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(func);

  auto attr0 = func.getArgAttr(0, "mhlo.layout_mode");
  ASSERT_TRUE(attr0);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(attr0).getValue(), "{1,0}");

  auto attr1 = func.getArgAttr(1, "mhlo.layout_mode");
  ASSERT_TRUE(attr1);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(attr1).getValue(), "{0,1}");
}

TEST_P(IfrtServingExecutableTest, NoReturn) {
  int64_t program_id = 111111;
  SetUpMockDeviceReservation(selector_, program_id, helper_->num_cores());
  auto executable = helper_->MakeExecutable(
      program_id, GetMlirModulePath("executable_no_return.mlir"));

  auto x = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto y = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({3, 1}));
  std::vector<tensorflow::Tensor> inputs{x, y};
  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(executable.get(), absl::MakeSpan(inputs), {}));

  ASSERT_EQ(result.size(), 0);
}

TEST_P(IfrtServingExecutableTest, StaticShape) {
  absl::SetVLogLevel("tpu_h2d_transfer_executor", 2);
  int64_t program_id = 789012;
  SetUpMockDeviceReservation(selector_, program_id, helper_->num_cores());

  auto mock_h2d_factory =
      std::make_unique<NiceMock<MockH2DTransferExecutorFactory>>();
  helper_->SetH2DTransferExecutorFactory(std::move(mock_h2d_factory));

  auto executable = helper_->MakeExecutable(
      program_id, GetMlirModulePath("executable_static_shape.mlir"));

  // The MLIR module defines a MatMul operation where `%arg0` and `%arg1`
  // use `%arg2` as the static shape argument, as indicated by the
  // `tf._static_shape_arg_idx = 2` attribute.

  // Test case 1: Dynamic shapes are within the static shape bounds.
  // `input_x1` has a dynamic shape of {2, 3}.
  // `input_y1` has a dynamic shape of {2, 3}.
  auto input_x1 =
      AsTensor<int32_t>({1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3}));
  auto input_y1 =
      AsTensor<int32_t>({1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3}));
  // `shape_tensor` provides the static shape {4, 3}.
  auto shape_tensor = AsTensor<int64_t>({4, 3}, tensorflow::TensorShape({2}));
  std::vector<tensorflow::Tensor> inputs1{input_x1, input_y1, shape_tensor};

  // Iterate over all cores first for warmup execution. This ensures the
  // executable is compiled and cached.
  for (int i = 0; i < helper_->num_cores(); i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(executable.get(), absl::MakeSpan(inputs1), {}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto result1, Execute(executable.get(), absl::MakeSpan(inputs1), {}));

  // The computation is effectively `input_x1` * `input_y1`^T.
  // With dynamic shapes {2, 3} and {2, 3}, this results in a {2, 2} matrix
  // before padding.
  // Calculation:
  // input_x1 (2x3) * input_y1^T (3x2) -> 2x2
  //   1 2 3    *    1 4
  //   4 5 6         2 5
  //                 3 6
  // (1*1 + 2*2 + 3*3) = 1 + 4 + 9 = 14
  // (1*4 + 2*5 + 3*6) = 4 + 10 + 18 = 32
  // (4*1 + 5*2 + 6*3) = 4 + 10 + 18 = 32
  // (4*4 + 5*5 + 6*6) = 16 + 25 + 36 = 77
  // The resulting 2x2 matrix is:
  // [[14, 32],
  //  [32, 77]]
  // Since the executable was compiled with a static shape, the output is padded
  // to the static output shape of {4, 4}.
  ASSERT_EQ(result1.size(), 1);
  const auto& out_tensor1 = result1[0];
  EXPECT_EQ(out_tensor1.shape(), tensorflow::TensorShape({4, 4}));
  auto out_matrix1 = out_tensor1.matrix<int32_t>();
  EXPECT_EQ(out_matrix1(0, 0), 14);
  EXPECT_EQ(out_matrix1(0, 1), 32);
  EXPECT_EQ(out_matrix1(1, 0), 32);
  EXPECT_EQ(out_matrix1(1, 1), 77);

  executable->Freeze();

  // Test case 2: Different dynamic shape inputs, but the same static bounds
  // as provided by `shape_tensor`. This should result in a cache hit.
  // `input_x2`: 1x3. `input_y2`: 1x3.
  auto input_x2 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  auto input_y2 = AsTensor<int32_t>({1, 2, 3}, tensorflow::TensorShape({1, 3}));
  std::vector<tensorflow::Tensor> inputs2{input_x2, input_y2, shape_tensor};

  TF_ASSERT_OK_AND_ASSIGN(
      auto result2, Execute(executable.get(), absl::MakeSpan(inputs2), {}));

  // Calculation: `input_x2` (1x3) * `input_y2`^T (3x1) -> 1x1.
  //   1 2 3  *  1
  //             2
  //             3
  // Result: (1*1 + 2*2 + 3*3) = 14.
  // The output is padded to the static output shape of {4, 4}.
  ASSERT_EQ(result2.size(), 1);
  const auto& out_tensor2 = result2[0];
  EXPECT_EQ(out_tensor2.shape(), tensorflow::TensorShape({4, 4}));
  auto out_matrix2 = out_tensor2.matrix<int32_t>();
  EXPECT_EQ(out_matrix2(0, 0), 14);

  // Verify that only one executable was created, since both test cases use
  // the same `shape_tensor` for static shape.
  EXPECT_EQ(executable->num_executables(), 1);
}

TEST_P(VariableInputTest, InterleaveVariable) {
  const auto& param = std::get<0>(GetParam());
  bool use_async = std::get<1>(GetParam());

  tsl::test_util::MockServingDeviceSelector device_selector;
  test_utils::IfrtServingExecutableTestHelper helper(&device_selector);
  int64_t program_id = 111111;
  SetUpMockDeviceReservation(device_selector, program_id, helper.num_cores());
  auto executable = helper.MakeExecutable(
      program_id, GetMlirModulePath("executable_long_inputs.mlir"));
  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry =
      helper.ifrt_restore_tensor_registry();

  std::vector<tensorflow::Tensor> inputs;
  std::vector<int> loaded_variable_indices;
  for (int i = 0; i < param.in_tensors.size(); i++) {
    if (param.is_variable[i]) {
      auto [input_tensor_promise, input_tensor_future] =
          tsl::MakePromise<tensorflow::Tensor>();
      IfrtRestoreTensorRegistry::RestoredTensorInfo restore_tensor_info = {
          .dtype_and_shape = tsl::Future<DtypeAndShape>(
              DtypeAndShape{.dtype = param.in_tensors[i].dtype(),
                            .shape = param.in_tensors[i].shape()}),
          .tensor_future = input_tensor_future};
      std::string variable_name = absl::StrCat("variable_", i);
      ASSERT_OK(ifrt_restore_tensor_registry->TryRegister(variable_name,
                                                          restore_tensor_info));
      loaded_variable_indices.push_back(i);
      input_tensor_promise.Set(param.in_tensors[i]);
      // Use string tensor containing the key (name) in place of variable
      // tensor.
      tensorflow::Tensor key_tensor(tensorflow::DT_STRING, {});
      key_tensor.scalar<tsl::tstring>()() = variable_name;
      inputs.push_back(key_tensor);
    } else {
      inputs.push_back(param.in_tensors[i]);
    }
  }

  ASSERT_EQ(inputs.size(), param.is_variable.size());

  auto execute_fn = [&](absl::Span<const tensorflow::Tensor> inputs,
                        absl::Span<const int> variable_arg_indices)
      -> absl::StatusOr<std::vector<tensorflow::Tensor>> {
    if (use_async) {
      TF_ASSIGN_OR_RETURN(
          auto future, executable->ExecuteAsync(inputs, variable_arg_indices));
      return future.Await();
    } else {
      return executable->Execute(inputs, variable_arg_indices);
    }
  };

  // Iterate over all cores first for warmup execution.
  for (int i = 0; i < helper.num_cores(); i++) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, execute_fn(absl::MakeSpan(inputs),
                                absl::MakeSpan(loaded_variable_indices)));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          execute_fn(absl::MakeSpan(inputs),
                                     absl::MakeSpan(loaded_variable_indices)));

  EXPECT_THAT(result, ElementsAre(TensorEq(param.expected_out_tensors[0]),
                                  TensorEq(param.expected_out_tensors[1]),
                                  TensorEq(param.expected_out_tensors[2])));
}

INSTANTIATE_TEST_SUITE_P(
    VariableInputTests, VariableInputTest,
    ::testing::Combine(
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
            }),
        ::testing::Bool()));

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
