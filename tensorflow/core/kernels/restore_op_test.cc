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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace {

class RestoreOpTest : public OpsTestBase {
 protected:
  // Makes an operation to restore two tensors
  void MakeRestoreOp(DataType dt) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "Restore")
                  .Input(FakeInput())
                  .Input(FakeInput())
                  .Attr("dt", dt)
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(RestoreOpTest, RestoreInt) {
  const string filename = io::JoinPath(testing::TmpDir(), "tensor_int");
  const string tensor_name = "tensor_int";

  // We first need to write a tensor using the save_op
  {
    // Initialize an operation
    NodeDef save;
    ASSERT_OK(NodeDefBuilder("save", "Save")
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput({DT_INT32}))
                  .Finalize(&save));

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

    gtl::InlinedVector<TensorValue, 4> inputs;

    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        DEVICE_CPU, device.get(), cpu_allocator(), save, &status));
    EXPECT_OK(status);

    // Run it

    // Input #0 is the file name
    Tensor input_0(DT_STRING, TensorShape({}));
    input_0.scalar<string>()() = filename;
    inputs.push_back({nullptr, &input_0});

    // Input #1 is the tensor name
    Tensor input_1(DT_STRING, TensorShape({}));
    input_1.scalar<string>()() = tensor_name;
    inputs.push_back({nullptr, &input_1});

    // Input #2 is an integer tensor: it's a 1-d array.
    Tensor input_2(DT_INT32, TensorShape({10}));
    for (int i = 0; i < 10; ++i) {
      input_2.flat<int32>()(i) = i + 1;
    }
    inputs.push_back({nullptr, &input_2});

    OpKernelContext::Params params;
    params.device = device.get();
    params.frame_iter = FrameAndIter(0, 0);
    params.inputs = &inputs;
    params.op_kernel = op.get();
    params.output_alloc_attr = [&device, &op, &params](int index) {
      AllocatorAttributes attr;
      const bool on_host = (op->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      return attr;
    };
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params.slice_reader_cache = &slice_reader_cache_wrapper;

    OpKernelContext ctx(params);
    op->Compute(&ctx);
    EXPECT_OK(ctx.status());
  }

  // Now we restore
  MakeRestoreOp(DT_INT32);
  // Add a file name
  AddInput<string>(TensorShape({}),
                   [&filename](int x) -> string { return filename; });
  // Add the tensor names
  AddInput<string>(TensorShape({}),
                   [&tensor_name](int x) -> string { return tensor_name; });

  ASSERT_OK(RunOpKernel());

  // Check that we have an integer tensor
  Tensor* output = GetOutput(0);
  TensorShape expected({10});
  EXPECT_TRUE(output->shape().IsSameSize(expected));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(i + 1, output->flat<int32>()(i));
  }
}

TEST_F(RestoreOpTest, RestoreFloat) {
  const string filename = io::JoinPath(testing::TmpDir(), "tensor_float");
  const string tensor_name = "tensor_float";

  // We first need to write a tensor using the save_op
  {
    // Initialize an operation
    NodeDef save;
    ASSERT_OK(NodeDefBuilder("save", "Save")
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput({DT_FLOAT}))
                  .Finalize(&save));

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));
    gtl::InlinedVector<TensorValue, 4> inputs;

    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        DEVICE_CPU, device.get(), cpu_allocator(), save, &status));
    EXPECT_OK(status);

    // Run it

    // Input #0 is the file name
    Tensor input_0(DT_STRING, TensorShape({}));
    input_0.scalar<string>()() = filename;
    inputs.push_back({nullptr, &input_0});

    // Input #1 is the tensor name
    Tensor input_1(DT_STRING, TensorShape({}));
    input_1.scalar<string>()() = tensor_name;
    inputs.push_back({nullptr, &input_1});

    // Input #2 is a float tensor: it's a 2-d array.
    Tensor input_2(DT_FLOAT, TensorShape({2, 4}));
    for (int i = 0; i < 8; ++i) {
      input_2.flat<float>()(i) = static_cast<float>(i) / 10;
    }
    inputs.push_back({nullptr, &input_2});

    OpKernelContext::Params params;
    params.device = device.get();
    params.frame_iter = FrameAndIter(0, 0);
    params.inputs = &inputs;
    params.op_kernel = op.get();
    params.output_alloc_attr = [&device, &op, &params](int index) {
      AllocatorAttributes attr;
      const bool on_host = (op->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      return attr;
    };
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params.slice_reader_cache = &slice_reader_cache_wrapper;

    OpKernelContext ctx(params);
    op->Compute(&ctx);
    EXPECT_OK(ctx.status());
  }

  // Now we restore
  MakeRestoreOp(DT_FLOAT);
  // Add a file name
  AddInput<string>(TensorShape({}),
                   [&filename](int x) -> string { return filename; });
  // Add the tensor names
  AddInput<string>(TensorShape({}),
                   [&tensor_name](int x) -> string { return tensor_name; });

  ASSERT_OK(RunOpKernel());

  // Check that we have a float tensor.
  Tensor* output = GetOutput(0);
  TensorShape expected({2, 4});
  EXPECT_TRUE(output->shape().IsSameSize(expected));
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(static_cast<float>(i) / 10, output->flat<float>()(i));
  }
}

class RestoreSliceOpTest : public OpsTestBase {
 protected:
  void MakeRestoreSliceOp(DataType dt) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "RestoreSlice")
                  .Input(FakeInput())
                  .Input(FakeInput())
                  .Input(FakeInput())
                  .Attr("dt", dt)
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(RestoreSliceOpTest, RestoreInt) {
  const string filename = io::JoinPath(testing::TmpDir(), "tensor_int");
  const string tensor_name = "tensor_int";

  // We first need to write a tensor using the save_op
  {
    // Initialize an operation
    NodeDef save;
    ASSERT_OK(NodeDefBuilder("save", "Save")
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput(DT_STRING))
                  .Input(FakeInput({DT_INT32}))
                  .Finalize(&save));

    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

    gtl::InlinedVector<TensorValue, 4> inputs;

    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        DEVICE_CPU, device.get(), cpu_allocator(), save, &status));
    EXPECT_OK(status);

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
    params.output_alloc_attr = [&device, &op, &params](int index) {
      AllocatorAttributes attr;
      const bool on_host = (op->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      return attr;
    };
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params.slice_reader_cache = &slice_reader_cache_wrapper;

    OpKernelContext ctx(params);
    op->Compute(&ctx);
    EXPECT_OK(ctx.status());
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

  ASSERT_OK(RunOpKernel());

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
