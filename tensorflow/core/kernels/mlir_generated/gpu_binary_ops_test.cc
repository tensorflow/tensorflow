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

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests are parametrized with the kernel name, the input data type and the
// output data type.
struct BinaryTestParam {
  std::string op_name;
  DataType input_type;
  DataType output_type;
  BinaryTestParam(const std::string& name, DataType input, DataType output)
      : op_name(name), input_type(input), output_type(output) {}
};

// To add additional tests for other kernels, search for PLACEHOLDER in this
// file.

class ParametricGpuBinaryOpsTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<BinaryTestParam> {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }

  template <typename T>
  void SetOp(const absl::InlinedVector<T, 10>& input_1,
             const TensorShape& shape_1,
             const absl::InlinedVector<T, 10>& input_2,
             const TensorShape& shape_2) {
    TF_ASSERT_OK(NodeDefBuilder("some_name", GetParam().op_name)
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    inputs_.clear();
    AddInputFromArray<T>(shape_1, input_1);
    AddInputFromArray<T>(shape_2, input_2);
  }

  template <typename T, typename BaselineType, typename OutT>
  void RunAndCompare(const absl::InlinedVector<T, 10>& input_1,
                     const TensorShape& shape_1,
                     const absl::InlinedVector<T, 10>& input_2,
                     const TensorShape& shape_2,
                     const absl::InlinedVector<OutT, 10>& output,
                     const TensorShape& output_shape) {
    SetOp<T>(input_1, shape_1, input_2, shape_2);
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value,
                           output_shape);
    test::FillValues<OutT>(&expected_tensor, output);
    test::ExpectEqual(expected_tensor, *GetOutput(0));
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestBroadcastingExpand() {
    auto input_1 = absl::InlinedVector<T, 10>{static_cast<T>(10)};
    auto input_2 = absl::InlinedVector<T, 10>{
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
        static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)};
    absl::InlinedVector<OutT, 10> expected{
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[0]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[1]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[2]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[3]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[4]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[5]))),
    };
    auto expected_shape = TensorShape({6});
    RunAndCompare<T, BaselineType, OutT>(input_1, TensorShape({1}), input_2,
                                         TensorShape({6}), expected,
                                         expected_shape);
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestBroadcastingInDim() {
    auto input_1 = absl::InlinedVector<T, 10>{
        static_cast<T>(10), static_cast<T>(20), static_cast<T>(30)};
    auto input_2 = absl::InlinedVector<T, 10>{
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
        static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)};
    absl::InlinedVector<OutT, 10> expected{
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[0]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[1]),
            static_cast<BaselineType>(input_2[1]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[2]),
            static_cast<BaselineType>(input_2[2]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[3]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[1]),
            static_cast<BaselineType>(input_2[4]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[2]),
            static_cast<BaselineType>(input_2[5]))),
    };
    auto expected_shape = TensorShape({2, 3});
    RunAndCompare<T, BaselineType, OutT>(input_1, TensorShape({3}), input_2,
                                         TensorShape({2, 3}), expected,
                                         expected_shape);
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestBroadcasting() {
    auto input_1 =
        absl::InlinedVector<T, 10>{static_cast<T>(10), static_cast<T>(20)};
    auto input_2 = absl::InlinedVector<T, 10>{
        static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    absl::InlinedVector<OutT, 10> expected{
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[0]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[1]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[0]),
            static_cast<BaselineType>(input_2[2]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[1]),
            static_cast<BaselineType>(input_2[0]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[1]),
            static_cast<BaselineType>(input_2[1]))),
        static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
            static_cast<BaselineType>(input_1[1]),
            static_cast<BaselineType>(input_2[2]))),
    };
    auto expected_shape = TensorShape({2, 3});
    RunAndCompare<T, BaselineType, OutT>(input_1, TensorShape({2, 1}), input_2,
                                         TensorShape({3}), expected,
                                         expected_shape);
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void RunOp() {
    auto input_1 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    absl::InlinedVector<OutT, 10> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
          static_cast<BaselineType>(inp), static_cast<BaselineType>(inp))));
    }
    RunAndCompare<T, BaselineType, OutT>(input_1, TensorShape{2, 3}, input_2,
                                         TensorShape{2, 3}, expected,
                                         TensorShape{2, 3});
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestEqualShapes() {
    auto input_1 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    absl::InlinedVector<OutT, 10> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
          static_cast<BaselineType>(inp), static_cast<BaselineType>(inp))));
    }
    RunAndCompare<T, BaselineType, OutT>(input_1, TensorShape{2, 3}, input_2,
                                         TensorShape{2, 3}, expected,
                                         TensorShape{2, 3});
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestOneIsScalar() {
    auto input_1 = static_cast<T>(42);
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    absl::InlinedVector<OutT, 10> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<OutT>(Expected<BaselineType, BaselineOutT>(
          static_cast<BaselineType>(input_1), static_cast<BaselineType>(inp))));
    }
    RunAndCompare<T, BaselineType, OutT>({input_1}, TensorShape{}, input_2,
                                         TensorShape{2, 3}, expected,
                                         TensorShape{2, 3});
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestIncompatibleShapes() {
    auto input_1 = {static_cast<T>(-0.1), static_cast<T>(-0.0),
                    static_cast<T>(0.0)};
    auto input_2 = {static_cast<T>(-0.1), static_cast<T>(0.0)};

    SetOp<T>(input_1, TensorShape{3}, input_2, TensorShape{2});
    auto status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  }

  template <typename T, typename BaselineType, typename OutT,
            typename BaselineOutT>
  void TestEmptyShapeWithBroadcasting() {
    TensorShape input_shape_a{2, 0, 1};
    TensorShape input_shape_b{2, 0, 5};
    TensorShape expected_shape{2, 0, 5};
    absl::InlinedVector<T, 10> empty_input = {};
    absl::InlinedVector<OutT, 10> expected_result = {};
    RunAndCompare<T, BaselineType, OutT>(empty_input, input_shape_a,
                                         empty_input, input_shape_b,
                                         expected_result, expected_shape);
    RunAndCompare<T, BaselineType, OutT>(empty_input, input_shape_b,
                                         empty_input, input_shape_a,
                                         expected_result, expected_shape);
  }

  template <typename BaselineType, typename BaselineOutT>
  BaselineOutT Expected(BaselineType lhs, BaselineType rhs) {
    if (GetParam().op_name == "AddV2") {
      return static_cast<BaselineOutT>(lhs + rhs);
    }
    if (GetParam().op_name == "Equal") {
      return static_cast<BaselineOutT>(lhs == rhs);
    }
    // Add the logic for creating expected values for the kernel you want to
    // test here.
    // <PLACEHOLDER>
    LOG(FATAL) << "Cannot generate expected result for op "
               << GetParam().op_name;
    return static_cast<BaselineOutT>(lhs);
  }
};

std::vector<BinaryTestParam> GetBinaryTestParameters() {
  std::vector<BinaryTestParam> parameters;
  for (DataType dt :
       std::vector<DataType>{DT_FLOAT, DT_DOUBLE, DT_HALF, DT_INT64}) {
    parameters.emplace_back("AddV2", dt, dt);
  }
  for (DataType dt :
       std::vector<DataType>{DT_FLOAT, DT_DOUBLE, DT_HALF, DT_BOOL, DT_INT8,
                             DT_INT16, DT_INT64}) {
    parameters.emplace_back("Equal", dt, DT_BOOL);
  }
  // Add the parameters (kernel name and data types to test) here.
  // <PLACEHOLDER>
  return parameters;
}

#define GENERATE_DATA_TYPE_SWITCH_CASE(dt, nt, code)           \
  switch (dt) {                                                \
    case DT_FLOAT: {                                           \
      using nt = EnumToDataType<DT_FLOAT>::Type;               \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_DOUBLE: {                                          \
      using nt = EnumToDataType<DT_DOUBLE>::Type;              \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_HALF: {                                            \
      using nt = EnumToDataType<DT_HALF>::Type;                \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_INT8: {                                            \
      using nt = EnumToDataType<DT_INT8>::Type;                \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_INT16: {                                           \
      using nt = EnumToDataType<DT_INT16>::Type;               \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_INT32: {                                           \
      using nt = EnumToDataType<DT_INT32>::Type;               \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_INT64: {                                           \
      using nt = EnumToDataType<DT_INT64>::Type;               \
      code;                                                    \
      break;                                                   \
    }                                                          \
    case DT_BOOL: {                                            \
      using nt = EnumToDataType<DT_BOOL>::Type;                \
      code;                                                    \
      break;                                                   \
    }                                                          \
    default:                                                   \
      LOG(FATAL) << "Unsupported type: " << DataType_Name(dt); \
  }

#define COMMA ,

#define GENERATE_TEST_CALL(test_fn)                                           \
  GENERATE_DATA_TYPE_SWITCH_CASE(                                             \
      GetParam().input_type, NativeInT,                                       \
      GENERATE_DATA_TYPE_SWITCH_CASE(                                         \
          GetParam().output_type, NativeOutT,                                 \
          if (GetParam().input_type == DT_HALF) {                             \
            if (GetParam().output_type == DT_HALF) {                          \
              test_fn<NativeInT COMMA float COMMA NativeOutT COMMA float>();  \
            } else {                                                          \
              test_fn<                                                        \
                  NativeInT COMMA float COMMA NativeOutT COMMA NativeOutT>(); \
            }                                                                 \
          } else {                                                            \
            test_fn<NativeInT COMMA NativeInT COMMA NativeOutT COMMA          \
                        NativeOutT>();                                        \
          }))

TEST_P(ParametricGpuBinaryOpsTest, RunOp) { GENERATE_TEST_CALL(RunOp); }

TEST_P(ParametricGpuBinaryOpsTest, EqShapes) {
  GENERATE_TEST_CALL(TestEqualShapes);
}

TEST_P(ParametricGpuBinaryOpsTest, Scalar) {
  GENERATE_TEST_CALL(TestOneIsScalar);
}

TEST_P(ParametricGpuBinaryOpsTest, BCastExpand) {
  GENERATE_TEST_CALL(TestBroadcastingExpand);
}

TEST_P(ParametricGpuBinaryOpsTest, BCastInDim) {
  GENERATE_TEST_CALL(TestBroadcastingInDim);
}

TEST_P(ParametricGpuBinaryOpsTest, BCast) {
  GENERATE_TEST_CALL(TestBroadcasting);
}

TEST_P(ParametricGpuBinaryOpsTest, IncompatibleShapes) {
  GENERATE_TEST_CALL(TestIncompatibleShapes);
}

TEST_P(ParametricGpuBinaryOpsTest, EmptyShapeBCast) {
  GENERATE_TEST_CALL(TestEmptyShapeWithBroadcasting);
}

INSTANTIATE_TEST_SUITE_P(GpuBinaryOpsTests, ParametricGpuBinaryOpsTest,
                         ::testing::ValuesIn(GetBinaryTestParameters()));
}  // namespace
}  // end namespace tensorflow
