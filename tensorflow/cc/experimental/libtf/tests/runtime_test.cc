/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/experimental/libtf/tests/runtime_test.h"

namespace tf {
namespace libtf {
namespace runtime {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::tf::libtf::impl::TaggedValueTensor;

constexpr char kSimpleModel[] =
    "tensorflow/cc/experimental/libtf/tests/testdata/simple-model";

TEST_P(RuntimeTest, SimpleModelCallableFloatTest) {
  Runtime runtime = RuntimeTest::GetParam()();

  // Import the module and grab the callable
  const std::string module_path =
      tensorflow::GetDataDependencyFilepath(kSimpleModel);

  TF_ASSERT_OK_AND_ASSIGN(Object module,
                          runtime.Load(String(module_path.c_str())));
  std::cout << "Module imported." << std::endl;

  TF_ASSERT_OK_AND_ASSIGN(Callable fn,
                          module.Get<Callable>(String("test_float")));
  TF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor, runtime.CreateHostTensor<float>({}, TF_FLOAT, {2.0f}));
  TF_ASSERT_OK_AND_ASSIGN(Tensor result, fn.Call<Tensor>(Tensor(tensor)));

  float out_val[1];
  TF_ASSERT_OK(result.GetValue(absl::MakeSpan(out_val)));
  EXPECT_EQ(out_val[0], 6.0);
}

TEST_P(RuntimeTest, SimpleModelCallableIntTest) {
  Runtime runtime = RuntimeTest::GetParam()();

  // Import the module and grab the callable
  const std::string module_path =
      tensorflow::GetDataDependencyFilepath(kSimpleModel);
  TF_ASSERT_OK_AND_ASSIGN(Object module,
                          runtime.Load(String(module_path.c_str())));

  TF_ASSERT_OK_AND_ASSIGN(Callable fn,
                          module.Get<Callable>(String("test_int")));

  // Call the function
  TF_ASSERT_OK_AND_ASSIGN(Tensor host_tensor,
                          runtime.CreateHostTensor<int>({}, TF_INT32, {2}));

  TF_ASSERT_OK_AND_ASSIGN(Tensor tensor, fn.Call<Tensor>(Tensor(host_tensor)));

  int out_val[1];
  TF_ASSERT_OK(tensor.GetValue(absl::MakeSpan(out_val)));
  EXPECT_EQ(out_val[0], 6);
}

TEST_P(RuntimeTest, SimpleModelCallableMultipleArgsTest) {
  Runtime runtime = RuntimeTest::GetParam()();

  // Import the module and grab the callable
  const std::string module_path =
      tensorflow::GetDataDependencyFilepath(kSimpleModel);
  TF_ASSERT_OK_AND_ASSIGN(Object module,
                          runtime.Load(String(module_path.c_str())));

  TF_ASSERT_OK_AND_ASSIGN(Callable fn,
                          module.Get<Callable>(String("test_add")));

  TF_ASSERT_OK_AND_ASSIGN(Tensor tensor1,
                          runtime.CreateHostTensor<float>({}, TF_FLOAT, {2.0f}))
  TF_ASSERT_OK_AND_ASSIGN(Tensor tensor2,
                          runtime.CreateHostTensor<float>({}, TF_FLOAT, {3.0f}))

  TF_ASSERT_OK_AND_ASSIGN(Tensor result_tensor,
                          fn.Call<Tensor>(tensor1, tensor2));
  float out_val[1];
  TF_ASSERT_OK(result_tensor.GetValue(absl::MakeSpan(out_val)));
  EXPECT_EQ(out_val[0], 5.0f);
}

TEST_P(RuntimeTest, CreateHostTensorIncompatibleShape) {
  Runtime runtime = RuntimeTest::GetParam()();
  EXPECT_THAT(runtime.CreateHostTensor<float>({2}, TF_FLOAT, {2.0f}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched shape and data size")));
}

TEST_P(RuntimeTest, CreateHostTensorNonFullyDefinedShapeRaises) {
  Runtime runtime = RuntimeTest::GetParam()();
  EXPECT_THAT(runtime.CreateHostTensor<float>({-1}, TF_FLOAT, {2.0f}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shape must be fully-defined")));
}

TEST_P(RuntimeTest, CreateHostTensorIncompatibleDataType) {
  Runtime runtime = RuntimeTest::GetParam()();
  EXPECT_THAT(runtime.CreateHostTensor<float>({1}, TF_BOOL, {2.0f}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of bytes in data buffer")));
}

TEST_P(RuntimeTest, TensorCopyInvalidSize) {
  Runtime runtime = RuntimeTest::GetParam()();
  TF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor, runtime.CreateHostTensor<float>({1}, TF_FLOAT, {2.0f}))
  float val[2];

  EXPECT_THAT(tensor.GetValue(absl::MakeSpan(val)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched number of elements")));
}

}  // namespace runtime
}  // namespace libtf
}  // namespace tf
