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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

class RestoreOpTest : public OpsTestBase {
 protected:
  // Makes an operation to restore two tensors
  void MakeRestoreOp(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "Restore")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("dt", dt)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// Make an input tensor with filled results.
template <typename T>
Tensor MakeInput(const TensorShape& shape,
                 std::function<T(int)> input_mapping) {
  Tensor input(DataTypeToEnum<T>::v(), shape);
  test::FillFn(&input, input_mapping);
  return input;
}

TEST_F(RestoreOpTest, RestoreSimple) {
  const string filename = io::JoinPath(testing::TmpDir(), "tensor_simple");
  const std::vector<string> tensor_names = {
      "tensor_bool",  "tensor_int",    "tensor_float",  "tensor_double",
      "tensor_qint8", "tensor_qint32", "tensor_uint8",  "tensor_int8",
      "tensor_int16", "tensor_int64",  "tensor_string", "tensor_complex64",
      "tensor_half"};

  // We first need to write a tensor using the save_op
  {
    // Initialize an operation
    NodeDef save;
    TF_ASSERT_OK(
        NodeDefBuilder("myop", "Save")
            .Input(FakeInput())
            .Input(FakeInput())
            .Input(FakeInput({DT_BOOL, DT_INT32, DT_FLOAT, DT_DOUBLE, DT_QINT8,
                              DT_QINT32, DT_UINT8, DT_INT8, DT_INT16, DT_STRING,
                              DT_COMPLEX64, DT_HALF}))
            .Finalize(&save));

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

    gtl::InlinedVector<TensorValue, 4> inputs;

    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                                cpu_allocator(), save,
                                                TF_GRAPH_DEF_VERSION, &status));
    TF_EXPECT_OK(status);

    // Run it

    // Input #0 is the file name
    Tensor input_0(DT_STRING, TensorShape({}));
    input_0.scalar<string>()() = filename;
    inputs.push_back({nullptr, &input_0});

    // Input #1 is the tensor names
    Tensor input_1 = MakeInput<string>(
        TensorShape({static_cast<int>(tensor_names.size())}),
        [&tensor_names](int x) -> string { return tensor_names[x]; });
    inputs.push_back({nullptr, &input_1});

    // Input #2 is a 1-d bool tensor
    Tensor input_2 =
        MakeInput<bool>(TensorShape({2}), [](int x) -> bool { return x != 0; });
    inputs.push_back({nullptr, &input_2});
    // Input #3 is a 1-d integer tensor
    Tensor input_3 = MakeInput<int32>(TensorShape({10}),
                                      [](int x) -> int32 { return x + 1; });
    inputs.push_back({nullptr, &input_3});
    // Input #4 is a 2-d float tensor
    Tensor input_4 = MakeInput<float>(TensorShape({2, 4}), [](int x) -> float {
      return static_cast<float>(x) / 10;
    });
    inputs.push_back({nullptr, &input_4});
    // Input #5 is a 2-d double tensor
    Tensor input_5 = MakeInput<double>(
        TensorShape({2, 4}),
        [](int x) -> double { return static_cast<double>(x) / 20; });
    inputs.push_back({nullptr, &input_5});
    // Input #6 is a 2-d qint8 tensor
    Tensor input_6 = MakeInput<qint8>(TensorShape({3, 2}), [](int x) -> qint8 {
      return *reinterpret_cast<qint8*>(&x);
    });
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
    Tensor input_9 =
        MakeInput<int8>(TensorShape({7}), [](int x) -> int8 { return x - 7; });
    inputs.push_back({nullptr, &input_9});
    // Input #10 is a 1-d int16 tensor
    Tensor input_10 = MakeInput<int16>(TensorShape({7}),
                                       [](int x) -> int16 { return x - 8; });
    inputs.push_back({nullptr, &input_10});
    // Input #11 is a 1-d int64 tensor
    Tensor input_11 = MakeInput<int64>(TensorShape({9}),
                                       [](int x) -> int64 { return x - 9; });
    inputs.push_back({nullptr, &input_11});
    // Input #12 is a 1-d string tensor
    Tensor input_12 = MakeInput<string>(
        TensorShape({2}), [](int x) -> string { return x ? "yes" : "no"; });
    inputs.push_back({nullptr, &input_12});
    // Input #13 is a 1-d complex64 tensor
    Tensor input_13 = MakeInput<complex64>(
        TensorShape({2, 3}),
        [](int x) -> complex64 { return complex64(100 + x, 200 + x); });
    inputs.push_back({nullptr, &input_13});
    // Input #14 is a 2-d half tensor
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
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params.slice_reader_cache = &slice_reader_cache_wrapper;

    OpKernelContext ctx(&params);
    op->Compute(&ctx);
    TF_EXPECT_OK(ctx.status());
  }

  // Now we restore

  // The 1-d bool tensor
  {
    MakeRestoreOp(DT_BOOL);
    AddInput<string>(TensorShape({}),
                     [&filename](int x) -> string { return filename; });
    AddInput<string>(TensorShape({}),
                     [&](int x) -> string { return tensor_names[0]; });
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[1];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[2];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[3];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[4];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[5];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[6];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[7];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[8];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[9];
    TF_ASSERT_OK(RunOpKernel());
    Tensor* output = GetOutput(0);
    TensorShape expected({9});
    EXPECT_TRUE(output->shape().IsSameSize(expected));
    for (int i = 0; i < 9; ++i) {
      EXPECT_EQ(i - 9, output->flat<int64>()(i));
    }
  }
  // The 1-d string tensor
  {
    MakeRestoreOp(DT_STRING);
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[10];
    TF_ASSERT_OK(RunOpKernel());
    Tensor* output = GetOutput(0);
    TensorShape expected({2});
    EXPECT_TRUE(output->shape().IsSameSize(expected));
    EXPECT_EQ("no", output->flat<string>()(0));
    EXPECT_EQ("yes", output->flat<string>()(1));
  }
  // The 2-d complex64 tensor
  {
    MakeRestoreOp(DT_COMPLEX64);
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[11];
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
    (*mutable_input(1).tensor).scalar<string>()() = tensor_names[12];
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

class RestoreSliceOpTest : public OpsTestBase {
 protected:
  void MakeRestoreSliceOp(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "RestoreSlice")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("dt", dt)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RestoreSliceOpTest, RestoreInt) {
  const string filename = io::JoinPath(testing::TmpDir(), "tensor_int");
  const string tensor_name = "tensor_int";

  // We first need to write a tensor using the save_op
  {
    // Initialize an operation
    NodeDef save;
    TF_ASSERT_OK(NodeDefBuilder("save", "Save")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput({DT_INT32}))
                     .Finalize(&save));

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

    gtl::InlinedVector<TensorValue, 4> inputs;

    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                                cpu_allocator(), save,
                                                TF_GRAPH_DEF_VERSION, &status));
    TF_EXPECT_OK(status);

    // Run it

    // Input #0 is the file name
    Tensor input_0(DT_STRING, TensorShape({}));
    input_0.scalar<string>()() = filename;
    inputs.push_back({nullptr, &input_0});

    // Input #1 is the tensor name
    Tensor input_1(DT_STRING, TensorShape({}));
    input_1.scalar<string>()() = tensor_name;
    inputs.push_back({nullptr, &input_1});

    // Input #2 is a 4x16 integer tensor.
    Tensor input_2(DT_INT32, TensorShape({4, 16}));
    for (int64 i = 0; i < input_2.NumElements(); ++i) {
      input_2.flat<int32>()(i) = i + 1;
    }
    inputs.push_back({nullptr, &input_2});

    OpKernelContext::Params params;
    params.device = device.get();
    params.frame_iter = FrameAndIter(0, 0);
    params.inputs = &inputs;
    params.op_kernel = op.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(&params, &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params.slice_reader_cache = &slice_reader_cache_wrapper;

    OpKernelContext ctx(&params);
    op->Compute(&ctx);
    TF_EXPECT_OK(ctx.status());
  }

  // Now we restore
  MakeRestoreSliceOp(DT_INT32);
  string shape_and_slice = "4 16 0,2:-";
  // Add a file name
  AddInput<string>(TensorShape({}),
                   [&filename](int x) -> string { return filename; });
  // Add the tensor names
  AddInput<string>(TensorShape({}),
                   [&tensor_name](int x) -> string { return tensor_name; });
  // Add the tensor shape and slice
  AddInput<string>(TensorShape({}), [&shape_and_slice](int x) -> string {
    return shape_and_slice;
  });

  TF_ASSERT_OK(RunOpKernel());

  // Check that we have an integer tensor
  Tensor* output = GetOutput(0);
  TensorShape expected({2, 16});
  EXPECT_TRUE(output->shape().IsSameSize(expected));
  for (int64 i = 0; i < expected.num_elements(); ++i) {
    EXPECT_EQ(i + 1, output->flat<int32>()(i));
  }
}

}  // namespace
}  // namespace tensorflow
