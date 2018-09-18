/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class CudnnRNNOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(
    string rnn_mode, string input_mode, string direction
  ) {
    TF_EXPECT_OK(NodeDefBuilder("cudnn_rnn_op", "CudnnRNN")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
					 .Device(DEVICE_GPU)
                     .Attr("rnn_mode", rnn_mode)
                     .Attr("input_mode", input_mode)
                     .Attr("direction", direction)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST_T(T) \
TEST_F(CudnnRNNOpTest, TestCudnnRNNOp##T) { \
	 SetDevice(DEVICE_GPU, \
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice( \
                    "GPU", {}, "/job:a/replica:0/task:0"))); \
  MakeOp<T>("lstm","linear_input","unidirectional"); \
  AddInputFromArray<T>(TensorShape({6,3,1}), { \
	  1,1,1, \
	  1,1,1, \
	  1,1,1, \
	  1,1,1, \
	  1,1,1, \
	  1,1,1}); \
  AddInputFromArray<T>(TensorShape({1, 3,1}), {0, 0, 0}); \
  AddInputFromArray<T>(TensorShape({1, 3,1}), {0, 0, 0}); \
  AddInputFromArray<T>(TensorShape({16}), { \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2}); \
  TF_ASSERT_OK(RunOpKernel()); \
  Tensor expected(allocator(), DataTypeToEnum<T>::value, TensorShape({6, 3, 1})); \
  test::FillValues<T>(&expected, { \
	  0.2153, 0.2153, 0.2153,  \
	  0.3515, 0.3515, 0.3515,  \
	  0.4331, 0.4331, 0.4331,  \
	  0.4821, 0.4821, 0.4821,  \
	  0.5122, 0.5122, 0.5122,  \
	  0.5313, 0.5313, 0.5313}); \
	const Tensor& output = *GetOutput(0); \
    TF_EXPECT_OK(device_->Sync()); \
  test::ExpectTensorNear<T>(expected, output, 0.001); \
}
TF_CALL_float(REGISTER_TEST_T)
TF_CALL_double(REGISTER_TEST_T)
#undef REGISTER_TEST_T


class CudnnRNNVarLenOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(
    string rnn_mode, string input_mode, string direction
  ) {
    TF_EXPECT_OK(NodeDefBuilder("cudnn_rnn_var_len_op", "CudnnRNNVarLen")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_INT32))
					 .Device(DEVICE_GPU)
                     .Attr("rnn_mode", rnn_mode)
                     .Attr("input_mode", input_mode)
                     .Attr("direction", direction)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST_T(T)                                               \
  TEST_F(CudnnRNNVarLenOpTest, TestVarLen##T) {   \
	 SetDevice(DEVICE_GPU, \
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice( \
                    "GPU", {}, "/job:a/replica:0/task:0"))); \
  MakeOp<T>("lstm","linear_input","unidirectional"); \
	  AddInputFromArray<T>(TensorShape({11,1}), { \
	  1,1,1, \
	  1,1,1, \
	  1,1, \
	  1, \
	  1, \
	  1}); \
  AddInputFromArray<T>(TensorShape({1, 3,1}), {0, 0, 0}); \
  AddInputFromArray<T>(TensorShape({1, 3,1}), {0, 0, 0}); \
  AddInputFromArray<T>(TensorShape({16}), { \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2, \
	  0.2,0.2,0.2,0.2}); \
  AddInputFromArray<int32>(TensorShape({3}), {6,3,2}); \
    TF_ASSERT_OK(RunOpKernel());                                       \
    TF_EXPECT_OK(device_->Sync()); \
	const Tensor& output = *GetOutput(0); \
    Tensor expected(allocator(), DataTypeToEnum<T>::value, TensorShape({11,1})); \
    test::FillValues<T>(&expected, { \
	  0.2153, 0.2153, 0.2153,  \
	  0.3515, 0.3515, 0.3515,  \
	  0.4331, 0.4331,  \
	  0.4821,   \
	  0.5122,   \
	  0.5313}); \
    TF_EXPECT_OK(device_->Sync()); \
    test::ExpectTensorNear<T>(expected, output, 0.001);           \
  }
TF_CALL_float(REGISTER_TEST_T)
TF_CALL_double(REGISTER_TEST_T)
#undef REGISTER_TEST_T

}  // namespace tensorflow
