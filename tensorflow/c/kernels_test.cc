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
  LOG(INFO) << "Wow, actually got into creation";
  struct MyCustomKernel* s = new struct MyCustomKernel;
  s->created = true;
  s->compute_called = false;
  return s;
}

static void MyComputeFunc(void* kernel, TF_OpKernelContext* ctx) {
  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  s->compute_called = true;
}

static void MyDeleteFunc(void* kernel) {
  struct MyCustomKernel* s = static_cast<struct MyCustomKernel*>(kernel);
  EXPECT_TRUE(s->created);
  EXPECT_TRUE(s->compute_called);
  delete_called = true;
  delete s;
}

// Tests registration of a single C kernel and checks that calls through the
// C/C++ boundary are being made.
TEST(TestKernel, TestRegisterKernelBuilder) {
  const char* kernel_name = "SomeKernelName";
  const char* op_name = "FooOp";
  const char* device_name = "barDev";

  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(kernel_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_Buffer* buf = TF_GetRegisteredKernelsForOp("FooOp", status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    ::tensorflow::KernelList list;
    list.ParseFromArray(buf->data, buf->length);
    ASSERT_EQ(1, list.kernel_size());
    ASSERT_EQ("barDev", list.kernel(0).device_type());
    TF_DeleteBuffer(buf);
    TF_DeleteStatus(status);
  }

  REGISTER_OP("FooOp")
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8");

  {
    ::tensorflow::NodeDef def;
    def.set_op("FooOp");
    def.set_device("bar");
    def.add_input("input1");
    def.add_input("input2");
    ::tensorflow::Status status;
    std::unique_ptr<::tensorflow::OpKernel> kernel =
        ::tensorflow::CreateOpKernel(::tensorflow::DeviceType("barDev"),
                                     nullptr, nullptr, def, 1, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);
  }

  ASSERT_TRUE(delete_called);
}
