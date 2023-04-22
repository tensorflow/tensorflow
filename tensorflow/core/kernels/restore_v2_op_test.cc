/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// Make an input tensor with filled results.
template <typename T>
Tensor MakeInput(const TensorShape& shape,
                 std::function<T(int)> input_mapping) {
  Tensor input(DataTypeToEnum<T>::v(), shape);
  test::FillFn(&input, input_mapping);
  return input;
}

class RestoreV2OpTest : public OpsTestBase {
 protected:
  // Makes an operation to restore two tensors
  void MakeRestoreOp(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "RestoreV2")
                     .Input(FakeInput())    // prefix
                     .Input(FakeInput())    // tensor_names
                     .Input(FakeInput())    // shape_and_slices
                     .Attr("dtypes", {dt})  // dtypes
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void RunTest(StringPiece save_op_to_use) {
    const string filename =
        io::JoinPath(testing::TmpDir(), "tensor_simple-", save_op_to_use);
    const std::vector<string> tensor_names = {
        "tensor_bool",  "tensor_int",    "tensor_float",     "tensor_double",
        "tensor_qint8", "tensor_qint32", "tensor_uint8",     "tensor_int8",
        "tensor_int16", "tensor_int64",  "tensor_complex64", "tensor_half"};

    // We first need to write using the desired save op.
    {
      // Initialize an operation.
      NodeDef save;
      if (save_op_to_use != "Save") {
        TF_ASSERT_OK(
            NodeDefBuilder("myop", save_op_to_use)
                .Input(FakeInput())  // prefix
                .Input(FakeInput())  // tensor_names
                .Input(FakeInput())  // shape_and_slices
                .Input(FakeInput({DT_BOOL, DT_INT32, DT_FLOAT, DT_DOUBLE,
                                  DT_QINT8, DT_QINT32, DT_UINT8, DT_INT8,
                                  DT_INT16, DT_COMPLEX64, DT_HALF}))  // tensors
                .Finalize(&save));
      } else {
        TF_ASSERT_OK(
            NodeDefBuilder("myop", save_op_to_use)
                .Input(FakeInput())  // file
                .Input(FakeInput())  // tensor_names
                .Input(FakeInput({DT_BOOL, DT_INT32, DT_FLOAT, DT_DOUBLE,
                                  DT_QINT8, DT_QINT32, DT_UINT8, DT_INT8,
                                  DT_INT16, DT_COMPLEX64, DT_HALF}))  // tensors
                .Finalize(&save));
      }

      std::unique_ptr<Device> device(
          DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

      gtl::InlinedVector<TensorValue, 4> inputs;

      Status status;
      std::unique_ptr<OpKernel> op(
          CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(), save,
                         TF_GRAPH_DEF_VERSION, &status));
      TF_EXPECT_OK(status);

      // Run it

      // Input #0 is the file name
      Tensor input_0(DT_STRING, TensorShape({}));
      input_0.scalar<tstring>()() = filename;
      inputs.push_back({nullptr, &input_0});

      // Input #1 is the tensor names
      Tensor input_1 = MakeInput<tstring>(
          TensorShape({static_cast<int>(tensor_names.size())}),
          [&tensor_names](int x) -> string { return tensor_names[x]; });
      inputs.push_back({nullptr, &input_1});

      Tensor shape_and_slices = MakeInput<tstring>(
          TensorShape({static_cast<int>(tensor_names.size())}),
          [](int x) -> string { return "" /* saves in full */; });
      if (save_op_to_use != "Save") {
        inputs.push_back({nullptr, &shape_and_slices});
      }

      // Input #2 is a 1-d bool tensor
      Tensor input_2 = MakeInput<bool>(TensorShape({2}),
                                       [](int x) -> bool { return x != 0; });
      inputs.push_back({nullptr, &input_2});
      // Input #3 is a 1-d integer tensor
      Tensor input_3 = MakeInput<int32>(TensorShape({10}),
                                        [](int x) -> int32 { return x + 1; });
      inputs.push_back({nullptr, &input_3});
      // Input #4 is a 2-d float tensor
      Tensor input_4 = MakeInput<float>(
          TensorShape({2, 4}),
          [](int x) -> float { return static_cast<float>(x) / 10; });
      inputs.push_back({nullptr, &input_4});
      // Input #5 is a 2-d double tensor
      Tensor input_5 = MakeInput<double>(
          TensorShape({2, 4}),
          [](int x) -> double { return static_cast<double>(x) / 20; });
      inputs.push_back({nullptr, &input_5});
      // Input #6 is a 2-d qint8 tensor
      Tensor input_6 = MakeInput<qint8>(
          TensorShape({3, 2}),
          [](int x) -> qint8 { return *reinterpret_cast<qint8*>(&x); });
      inputs.push_back({nullptr, &input_6});
      // Input #7 is a 2-d qint32 tensor
      Tensor input_7 =
          MakeInput<qint32>(TensorShape({2, 3}), [](int x) -> qint32 {
            return *reinterpret_cast<qint32*>(&x) * qint8(2);
          });
      inputs.push_back({nullptr, &input_7});
      // Input #8 is a 1-d uint8 tensor
      Tensor input_8 = MakeInput<uint8>(TensorShape({11}),
                                        [](int x) -> uint8 { return x + 1; });
      inputs.push_back({nullptr, &input_8});
      // Input #9 is a 1-d int8 tensor
      Tensor input_9 = MakeInput<int8>(TensorShape({7}),
                                       [](int x) -> int8 { return x - 7; });
      inputs.push_back({nullptr, &input_9});
      // Input #10 is a 1-d int16 tensor
      Tensor input_10 = MakeInput<int16>(TensorShape({7}),
                                         [](int x) -> int16 { return x - 8; });
      inputs.push_back({nullptr, &input_10});
      // Input #11 is a 1-d int64 tensor
      Tensor input_11 = MakeInput<int64>(TensorShape({9}),
                                         [](int x) -> int64 { return x - 9; });
      inputs.push_back({nullptr, &input_11});
      // Input #12 is a 1-d complex64 tensor
      Tensor input_13 = MakeInput<complex64>(
          TensorShape({2, 3}),
          [](int x) -> complex64 { return complex64(100 + x, 200 + x); });
      inputs.push_back({nullptr, &input_13});
      // Input #13 is a 2-d half tensor
      Tensor input_14 =
          MakeInput<Eigen::half>(TensorShape({2, 4}), [](int x) -> Eigen::half {
            return static_cast<Eigen::half>(x) / Eigen::half(5);
          });
      inputs.push_back({nullptr, &input_14});
      OpKernelContext::Params params;
      params.device = device.get();
      params.frame_iter = FrameAndIter(0, 0);
      params.inputs = &inputs;
      params.op_kernel = op.get();
      std::vector<AllocatorAttributes> attrs;
      test::SetOutputAttrs(&params, &attrs);

      OpKernelContext ctx(&params);
      op->Compute(&ctx);
      TF_EXPECT_OK(ctx.status());
    }

    // Now we restore

    // The 1-d bool tensor
    {
      MakeRestoreOp(DT_BOOL);
      AddInput<tstring>(TensorShape({}),
                        [&filename](int x) -> tstring { return filename; });
      AddInput<tstring>(TensorShape({1}),
                        [&](int x) -> tstring { return tensor_names[0]; });
      AddInput<tstring>(TensorShape({1}), [&](int x) -> tstring {
        return "";
      });  // Restores in full.
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(i != 0, output->flat<bool>()(i));
      }
    }
    // The 1-d integer tensor
    {
      MakeRestoreOp(DT_INT32);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[1];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({10});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i + 1, output->flat<int32>()(i));
      }
    }
    // The 2-d float tensor
    {
      MakeRestoreOp(DT_FLOAT);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[2];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2, 4});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(static_cast<float>(i) / 10, output->flat<float>()(i));
      }
    }
    // The 2-d double tensor
    {
      MakeRestoreOp(DT_DOUBLE);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[3];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2, 4});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(static_cast<double>(i) / 20, output->flat<double>()(i));
      }
    }
    // The 2-d qint8 tensor
    {
      MakeRestoreOp(DT_QINT8);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[4];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({3, 2});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(*reinterpret_cast<qint8*>(&i), output->flat<qint8>()(i));
      }
    }
    // The 2-d qint32 tensor
    {
      MakeRestoreOp(DT_QINT32);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[5];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2, 3});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(*reinterpret_cast<qint32*>(&i) * qint8(2),
                  output->flat<qint32>()(i));
      }
    }
    // The 1-d uint8 tensor
    {
      MakeRestoreOp(DT_UINT8);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[6];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({11});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 11; ++i) {
        EXPECT_EQ(i + 1, output->flat<uint8>()(i));
      }
    }
    // The 1-d int8 tensor
    {
      MakeRestoreOp(DT_INT8);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[7];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({7});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(i - 7, output->flat<int8>()(i));
      }
    }
    // The 1-d int16 tensor
    {
      MakeRestoreOp(DT_INT16);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[8];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({7});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(i - 8, output->flat<int16>()(i));
      }
    }
    // The 1-d int64 tensor
    {
      MakeRestoreOp(DT_INT64);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[9];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({9});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(i - 9, output->flat<int64>()(i));
      }
    }
    // The 2-d complex64 tensor
    {
      MakeRestoreOp(DT_COMPLEX64);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[10];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2, 3});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(complex64(100 + i, 200 + i), output->flat<complex64>()(i));
      }
    }
    // The 2-d half tensor
    {
      MakeRestoreOp(DT_HALF);
      (*mutable_input(1).tensor).flat<tstring>()(0) = tensor_names[11];
      TF_ASSERT_OK(RunOpKernel());
      Tensor* output = GetOutput(0);
      TensorShape expected({2, 4});
      EXPECT_TRUE(output->shape().IsSameSize(expected));
      for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(static_cast<Eigen::half>(i) / Eigen::half(5),
                  output->flat<Eigen::half>()(i));
      }
    }
  }
};

// The intended use case (write in V2, read in V2).
TEST_F(RestoreV2OpTest, RestoreAfterSaveV2) { RunTest("SaveV2"); }
// For backward compatibility.
TEST_F(RestoreV2OpTest, RestoreAfterSaveSlicesV1) { RunTest("SaveSlices"); }
TEST_F(RestoreV2OpTest, RestoreAfterSaveV1) { RunTest("Save"); }

}  // namespace
}  // namespace tensorflow
