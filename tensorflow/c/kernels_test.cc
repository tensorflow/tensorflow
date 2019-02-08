/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/kernels.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb_text.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

struct MyCustomKernel {
  bool created;
  bool compute_called;
};

static bool delete_called = false;

static void* MyCreateFunc(TF_OpKernelConstruction* ctx) {
  struct MyCustomKernel* s = new struct MyCustomKernel;
  s->created = true;
  s->compute_called = false;

  // Exercise attribute reads.
  TF_DataType type;
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrType(ctx, "SomeDataTypeAttr", &type, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  EXPECT_EQ(TF_FLOAT, type);
  TF_DeleteStatus(status);

  return s;
}

static void MyComputeFunc(void* kernel, TF_OpKernelContext* ctx) {
  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  s->compute_called = true;
  if (ctx != nullptr) {
    EXPECT_EQ(43, TF_StepId(ctx));
  }
}

static void MyDeleteFunc(void* kernel) {
  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  EXPECT_TRUE(s->created);
  EXPECT_TRUE(s->compute_called);
  delete_called = true;
  delete s;
}

namespace tensorflow {

static std::unique_ptr<OpKernel> GetFakeKernel(const char* device_name,
                                               const char* op_name,
                                               Status* status) {
  NodeDef def;
  def.set_op(op_name);
  def.set_device(device_name);
  def.add_input("input1");
  def.add_input("input2");

  AttrValue v;
  v.set_type(DataType::DT_FLOAT);
  (*def.mutable_attr())["SomeDataTypeAttr"] = v;

  return CreateOpKernel(DeviceType(device_name), nullptr, nullptr, def, 1,
                        status);
}

// Tests registration of a single C kernel and checks that calls through the
// C/C++ boundary are being made.
TEST(TestKernel, TestRegisterKernelBuilder) {
  const char* kernel_name = "SomeKernelName";
  const char* op_name = "FooOp";
  const char* device_name = "FakeDeviceName1";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("SomeDataTypeAttr: type");

  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(kernel_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    KernelList list;
    list.ParseFromArray(buf->data, buf->length);
    ASSERT_EQ(1, list.kernel_size());
    ASSERT_EQ(device_name, list.kernel(0).device_type());
    TF_DeleteBuffer(buf);
    TF_DeleteStatus(status);
  }

  {
    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);
  }

  ASSERT_TRUE(delete_called);
}

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }

 private:
  bool save_;
};

TEST(TestKernel, TestInputAndOutputCount) {
  const char* kernel_name = "InputOutputCounterKernel";
  const char* op_name = "BarOp";
  const char* device_name = "FakeDeviceName2";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("SomeDataTypeAttr: type");

  static int num_inputs = 0;
  static int num_outputs = 0;

  // A kernel whose Compute function has a side-effect of updating num_inputs
  // and num_outputs. Various functions on TF_OpKernelContext are also
  // exercised.
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    num_inputs = TF_NumInputs(ctx);
    num_outputs = TF_NumOutputs(ctx);

    TF_Tensor* input = nullptr;
    TF_Status* s = TF_NewStatus();
    TF_GetInput(ctx, 0, &input, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << "Failed to get input: " << TF_Message(s);
    EXPECT_EQ(123, *static_cast<tensorflow::uint8*>(TF_TensorData(input)));
    TF_GetInput(ctx, -1, &input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));
    TF_GetInput(ctx, 3, &input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));

    // Copy the input tensor to output.
    TF_SetOutput(ctx, 0, input, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));

    TF_SetOutput(ctx, 24, input, s);
    EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s));

    EXPECT_EQ(TF_UINT8, TF_ExpectedOutputDataType(ctx, 0));

    TF_DeleteStatus(s);
    if (input != nullptr) {
      TF_DeleteTensor(input);
    }
  };

  TF_KernelBuilder* builder = TF_NewKernelBuilder(op_name, device_name, nullptr,
                                                  my_compute_func, nullptr);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(kernel_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr, false);
    p.device = &dummy_device;
    p.step_id = 43;

    Tensor t(tensorflow::uint8(123));

    gtl::InlinedVector<TensorValue, 4> inputs;
    // Simulate 2 inputs
    inputs.emplace_back(&t);
    inputs.emplace_back();
    p.inputs = &inputs;

    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());

    p.op_kernel = kernel.get();
    OpKernelContext ctx(&p);
    kernel->Compute(&ctx);

    ASSERT_EQ(2, num_inputs);
    ASSERT_EQ(1, num_outputs);
    ASSERT_EQ(123, ctx.mutable_output(0)->scalar<tensorflow::uint8>()());
  }
}

TEST(TestKernel, DeleteKernelBuilderIsOkOnNull) {
  TF_DeleteKernelBuilder(nullptr);
}

}  // namespace tensorflow
