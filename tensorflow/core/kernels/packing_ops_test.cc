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

#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#include "tensorflow/core/kernels/ops_testutil.h"
namespace tensorflow {

//PackedSequenceAlignment Tests
class PackedSequenceAlignmentOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("packed_sequence_alignment_op", "PackedSequenceAlignment")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
					 .Device(DEVICE_GPU)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST_T(T)                                               \
  TEST_F(PackedSequenceAlignmentOpTest, TestAlignment##T) {                  \
      SetDevice(DEVICE_GPU, \
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(\
                    "GPU", {}, "/job:a/replica:0/task:0")));\
    MakeOp<T>();                                          \
    AddInputFromArray<T>(TensorShape({3}), {6,3,2});     \
    TF_ASSERT_OK(RunOpKernel());                                       \
	const Tensor& alignments = *GetOutput(0); \
	const Tensor& batch_sizes = *GetOutput(1); \
    TF_EXPECT_OK(device_->Sync()); \
                                                                       \
    Tensor expected_alignments(allocator(), DataTypeToEnum<T>::value, TensorShape({6})); \
    test::FillValues<T>(&expected_alignments, {0,3,6,8,9,10});                         \
    Tensor expected_batch_sizes(allocator(), DataTypeToEnum<T>::value, TensorShape({6})); \
    test::FillValues<T>(&expected_batch_sizes, {3,3,2,1,1,1});                         \
    test::ExpectTensorEqual<T>(expected_alignments, alignments);           \
    test::ExpectTensorEqual<T>(expected_batch_sizes, batch_sizes);           \
  }
REGISTER_TEST_T(int8)
REGISTER_TEST_T(int16)
REGISTER_TEST_T(int32)
REGISTER_TEST_T(int64)
#undef REGISTER_TEST_T

//PackSequence Tests
class PackSequenceOpTest : public OpsTestBase {
 protected:
  template <typename T, typename Index>
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("pack_sequence_op", "PackSequence")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<Index>::value))
                     .Input(FakeInput(DataTypeToEnum<Index>::value))
					 .Device(DEVICE_GPU)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST_T_Index(T, Index)                                               \
  TEST_F(PackSequenceOpTest, TestPackSequence##T##Index) {                  \
      SetDevice(DEVICE_GPU, \
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(\
                    "GPU", {}, "/job:a/replica:0/task:0")));\
    MakeOp<T, Index>();                                          \
    AddInputFromArray<T>(TensorShape({6,3,1}), { \
		1,2,3,\
		4,5,6, \
		7,8,9, \
		10,11,12, \
		13,14,15, \
		16,17,18});     \
    AddInputFromArray<Index>(TensorShape({6}), {0,3,6,8,9,10});     \
    AddInputFromArray<Index>(TensorShape({6}), {3,3,2,1,1,1});     \
    TF_ASSERT_OK(RunOpKernel());                                       \
	const Tensor& packed = *GetOutput(0); \
    TF_EXPECT_OK(device_->Sync()); \
                                                                       \
    Tensor expected_packed(allocator(), DataTypeToEnum<T>::value, TensorShape({11,1})); \
    test::FillValues<T>(&expected_packed, {1,2,3,4,5,6,7,8,10,13,16});                         \
    test::ExpectTensorEqual<T>(expected_packed, packed);           \
  }
#define REGISTER_TEST_T(T) \
	REGISTER_TEST_T_Index(T, int32) \
	REGISTER_TEST_T_Index(T, int64)
REGISTER_TEST_T(int32)
REGISTER_TEST_T(int64)
REGISTER_TEST_T(float)
REGISTER_TEST_T(double)
#undef REGISTER_TEST_T
#undef REGISTER_TEST_T_Index

//UnpackSequence Tests
class UnpackSequenceOpTest : public OpsTestBase {
 protected:
  template <typename T, typename Index>
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("unpack_sequence_op", "UnpackSequence")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DataTypeToEnum<Index>::value))
                     .Input(FakeInput(DataTypeToEnum<Index>::value))
					 .Device(DEVICE_GPU)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

#define REGISTER_TEST_T_Index(T, Index)                                               \
  TEST_F(UnpackSequenceOpTest, TestUnpackSequence##T##Index) {                  \
      SetDevice(DEVICE_GPU, \
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(\
                    "GPU", {}, "/job:a/replica:0/task:0")));\
    MakeOp<T, Index>();                                          \
    AddInputFromArray<T>(TensorShape({11,1}), {1,2,3,4,5,6,7,8,10,13,16});     \
    AddInputFromArray<Index>(TensorShape({6}), {0,3,6,8,9,10});     \
    AddInputFromArray<Index>(TensorShape({6}), {3,3,2,1,1,1});     \
    TF_ASSERT_OK(RunOpKernel());                                       \
	const Tensor& sequence = *GetOutput(0); \
    TF_EXPECT_OK(device_->Sync()); \
                                                                       \
    Tensor expected_sequence(allocator(), DataTypeToEnum<T>::value, TensorShape({6,3,1})); \
    test::FillValues<T>(&expected_sequence, { \
		1,2,3, \
		4,5,6, \
		7,8,0, \
		10,0,0, \
		13,0,0, \
		16,0,0});                         \
    test::ExpectTensorEqual<T>(expected_sequence, sequence);           \
  }
#define REGISTER_TEST_T(T) \
	REGISTER_TEST_T_Index(T, int32) \
	REGISTER_TEST_T_Index(T, int64)
REGISTER_TEST_T(int32)
REGISTER_TEST_T(int64)
REGISTER_TEST_T(float)
REGISTER_TEST_T(double)
#undef REGISTER_TEST_T
#undef REGISTER_TEST_T_Index



}  // namespace tensorflow
