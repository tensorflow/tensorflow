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

/*
#define REGISTER_TEST(T)                                               \
  TEST_F(CudnnRNNVarLenOpTest, TestCropAndResize##T) {                  \
    MakeOp<T>(8,5,3);                                          \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {2.5});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }                                                                    \
                                                                       \
  TEST_F(CudnnRNNVarLenOpTest, TestCropAndResize##T##nearest) {         \
    MakeOp<T>(0, "nearest");                                           \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {4.0});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8)
REGISTER_TEST(uint16)
REGISTER_TEST(int8)
REGISTER_TEST(int16)
REGISTER_TEST(int32)
REGISTER_TEST(int64)

#undef REGISTER_TEST
*/


  /*
    .Input("input: T") len,n,dim
    .Input("input_h: T") layers,n,units
    .Input("input_c: T") layers,n,units
    .Input("params: T") 16
    .Input("sequence_lengths: int32") n,
  */


TEST_F(CudnnRNNOpTest, TestCudnnRNNOp) {
	 SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  MakeOp<float>("lstm","linear_input","unidirectional");
  AddInputFromArray<float>(TensorShape({3,2,1}), {1,1,1,1,1,1});
  AddInputFromArray<float>(TensorShape({1, 2,1}), {1, 1});
  AddInputFromArray<float>(TensorShape({1, 2,1}), {1, 1});
  AddInputFromArray<float>(TensorShape({16}), {
	  1,1,1,1,
	  1,1,1,1,
	  1,1,1,1,
	  1,1,1,1});
  TF_ASSERT_OK(RunOpKernel());
  
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 2, 1}));
  test::FillValues<float>(&expected, {
	  0.94405508f, 0.94405508f, 0.97515064f,
	  0.97515064f, 0.98065156f, 0.98065156f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

TEST_F(CudnnRNNVarLenOpTest, TestCudnnRNNVarLenOp) {
	 SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  MakeOp<float>("lstm","linear_input","unidirectional");
  AddInputFromArray<float>(TensorShape({6, 1}), {1,1,1,1,1,1,});
  AddInputFromArray<float>(TensorShape({1, 2, 1}), {1, 1});
  AddInputFromArray<float>(TensorShape({1, 2, 1}), {1, 1});
  AddInputFromArray<float>(TensorShape({16}), {
	  1,1,1,1,
	  1,1,1,1,
	  1,1,1,1,
	  1,1,1,1});
  AddInputFromArray<int32>(TensorShape({2}), {3,3});
  TF_ASSERT_OK(RunOpKernel());
  
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6, 1}));
  //test::FillValues<float>(&expected, {
//	  0.94405508f, 0.94405508f, 0.97515064f,
//	  0.97515064f, 0.98065156f, 0.98065156f});
  test::FillValues<float>(&expected, {
	  0,0,0,0,0,0});

  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
}

}  // namespace tensorflow
