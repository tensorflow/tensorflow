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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/c/kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

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

  // Exercise kernel NodeDef name read
  TF_StringView name_string_view = TF_OpKernelConstruction_GetName(ctx);
  std::string node_name = "SomeNodeName";
  std::string candidate_node_name =
      std::string(name_string_view.data, name_string_view.len);
  EXPECT_EQ(node_name, candidate_node_name);
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
                                               const char* node_name,
                                               Status* status) {
  NodeDef def;
  def.set_op(op_name);
  def.set_name(node_name);
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
  const char* node_name = "SomeNodeName";
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
    TF_RegisterKernelBuilder(node_name, builder, status);
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
        GetFakeKernel(device_name, op_name, node_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);
  }

  ASSERT_TRUE(delete_called);
}

// REGISTER_OP for TF_OpKernelConstruction_GetAttr* test cases.
// Registers two ops, each with a single attribute called 'Attr'.
// The attribute in one op will have a type 'type', the other
// will have list(type).
#define ATTR_TEST_REGISTER_OP(name, type)                     \
  REGISTER_OP("TestKernelAttr" #name)                         \
      .Attr("Attr: " #type)                                   \
      .SetShapeFn(tensorflow::shape_inference::UnknownShape); \
  REGISTER_OP("TestKernelAttr" #name "List")                  \
      .Attr("Attr: list(" #type ")")                          \
      .SetShapeFn(tensorflow::shape_inference::UnknownShape)
ATTR_TEST_REGISTER_OP(String, string);
ATTR_TEST_REGISTER_OP(Int, int);
ATTR_TEST_REGISTER_OP(Float, float);
ATTR_TEST_REGISTER_OP(Bool, bool);
ATTR_TEST_REGISTER_OP(Type, type);
#undef ATTR_TEST_REGISTER_OP

// Helper macros for the TF_OpKernelConstruction_GetAttr* tests.
#define EXPECT_TF_SIZE(attr_name, expected_list_size, expected_total_size) \
  do {                                                                     \
    int32_t list_size, total_size;                                         \
    TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, &list_size,        \
                                        &total_size, status);              \
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);            \
    EXPECT_EQ(expected_list_size, list_size);                              \
    EXPECT_EQ(expected_total_size, total_size);                            \
  } while (0)

typedef void* (*MyCreateFuncWithAttr)(TF_OpKernelConstruction*);
class TestKernelAttr : public ::testing::Test {
 public:
  TestKernelAttr() {}
  ~TestKernelAttr() override {}

  std::unique_ptr<OpKernel> GetFakeKernelWithAttr(const char* op_name,
                                                  AttrValue v, Status* status) {
    NodeDef def;
    def.set_op(op_name);
    def.set_name("FakeNode");
    def.set_device("FakeDevice");
    (*def.mutable_attr())["Attr"] = v;
    return CreateOpKernel(DeviceType("FakeDevice"), nullptr, nullptr, def, 1,
                          status);
  }

  void CreateAndCallKernelWithAttr(MyCreateFuncWithAttr MyCreateFuncAttr,
                                   const char* op_name, AttrValue& v) {
    TF_KernelBuilder* builder = TF_NewKernelBuilder(
        op_name, "FakeDevice", MyCreateFuncAttr, &MyComputeFunc, &MyDeleteFunc);
    {
      TF_Status* status = TF_NewStatus();
      TF_RegisterKernelBuilder("FakeNode", builder, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status));
      TF_DeleteStatus(status);
    }
    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernelWithAttr(op_name, v, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());
    kernel->Compute(nullptr);

    ASSERT_TRUE(delete_called);
  }
};

TEST_F(TestKernelAttr, String) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    std::unique_ptr<char[]> val(new char[5]);
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ 5);
    TF_OpKernelConstruction_GetAttrString(ctx, "Attr", val.get(),
                                          /*max_length*/ 5, status);

    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ("bunny", string(static_cast<const char*>(val.get()), 5));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_s("bunny");
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrString", v);
}

TEST_F(TestKernelAttr, StringList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    std::vector<string> list = {"bugs", "bunny", "duck"};
    int list_total_size = 0;
    for (const auto& s : list) {
      list_total_size += s.size();
    }

    TF_Status* status = TF_NewStatus();
    std::unique_ptr<char*[]> values(new char*[list.size()]);
    std::unique_ptr<size_t[]> lens(new size_t[list.size()]);
    std::unique_ptr<char[]> storage(new char[list_total_size]);
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list.size(),
                   /*expected_total_size*/ list_total_size);
    TF_OpKernelConstruction_GetAttrStringList(
        ctx, "Attr", values.get(), lens.get(), list.size(), storage.get(),
        list_total_size, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    for (size_t i = 0; i < list.size(); ++i) {
      EXPECT_EQ(list[i].size(), lens[i]) << i;
      EXPECT_EQ(list[i], string(static_cast<const char*>(values[i]), lens[i]))
          << i;
    }
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  std::string attr_in[] = {"bugs", "bunny", "duck"};
  SetAttrValue(gtl::ArraySlice<std::string>(attr_in, 3), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrStringList", v);
}

TEST_F(TestKernelAttr, Int) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    int64_t val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrInt64(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(1234, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_i(1234);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrInt", v);
}

TEST_F(TestKernelAttr, IntList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const int64_t list[] = {1, 2, 3, 4};
    const size_t list_size = TF_ARRAYSIZE(list);
    int64_t values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrInt64List(ctx, "Attr", values, list_size,
                                             status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  int64 attr_in[] = {1, 2, 3, 4};
  SetAttrValue(gtl::ArraySlice<int64>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrIntList", v);
}

TEST_F(TestKernelAttr, Float) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    float val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrFloat(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_FLOAT_EQ(2.718, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_f(2.718);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrFloat", v);
}

TEST_F(TestKernelAttr, FloatList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const float list[] = {1.414, 2.718, 3.1415};
    const size_t list_size = TF_ARRAYSIZE(list);
    float values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrFloatList(ctx, "Attr", values, list_size,
                                             status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  float attr_in[] = {1.414, 2.718, 3.1415};
  SetAttrValue(gtl::ArraySlice<float>(attr_in, 3), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrFloatList", v);
}

TEST_F(TestKernelAttr, Bool) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    unsigned char val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrBool(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(1, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_b(true);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrBool", v);
}

TEST_F(TestKernelAttr, BoolList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const unsigned char list[] = {1, 0, 1, 0};
    const size_t list_size = TF_ARRAYSIZE(list);
    unsigned char values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrBoolList(ctx, "Attr", values, list_size,
                                            status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  bool attr_in[] = {true, false, true, false};
  SetAttrValue(gtl::ArraySlice<bool>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrBoolList", v);
}

TEST_F(TestKernelAttr, Type) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    TF_DataType val;
    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ -1,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrType(ctx, "Attr", &val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(TF_FLOAT, val);
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  v.set_type(DT_FLOAT);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrType", v);
}

TEST_F(TestKernelAttr, TypeList) {
  auto my_create_func = [](TF_OpKernelConstruction* ctx) {
    struct MyCustomKernel* s = new struct MyCustomKernel;
    s->created = true;
    s->compute_called = false;

    const TF_DataType list[] = {TF_FLOAT, TF_DOUBLE, TF_HALF, TF_COMPLEX128};
    const size_t list_size = TF_ARRAYSIZE(list);
    TF_DataType values[list_size];

    TF_Status* status = TF_NewStatus();
    EXPECT_TF_SIZE(/*attr_name*/ "Attr", /*expected_list_size*/ list_size,
                   /*expected_total_size*/ -1);
    TF_OpKernelConstruction_GetAttrTypeList(ctx, "Attr", values, list_size,
                                            status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_TRUE(
        std::equal(std::begin(list), std::end(list), std::begin(values)));
    TF_DeleteStatus(status);
    return static_cast<void*>(s);
  };

  AttrValue v;
  DataType attr_in[] = {DT_FLOAT, DT_DOUBLE, DT_HALF, DT_COMPLEX128};
  SetAttrValue(gtl::ArraySlice<DataType>(attr_in, 4), &v);
  CreateAndCallKernelWithAttr(my_create_func, "TestKernelAttrTypeList", v);
}
#undef EXPECT_TF_SIZE

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }
};

TEST(TestKernel, TestInputAndOutputCount) {
  const char* node_name = "InputOutputCounterKernel";
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
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr);
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
        GetFakeKernel(device_name, op_name, node_name, &status);
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

std::string ExpectedString(const char* type) {
  const auto format_str = R"str(kernel {
  op: "TypeOp%s"
  device_type: "FakeDeviceName1"
  constraint {
    name: "T"
    allowed_values {
      list {
        type: %s
      }
    }
  }
}
)str";
  return absl::StrFormat(format_str, type, type);
}

#define TEST_KERNEL_TYPE_CONSTRAINT(tf_type, dtype)                          \
  TEST(TestKernel, TestTypeConstraint##tf_type) {                            \
    const char* node_name = "SomeNodeName";                                  \
    const char* op_name = "TypeOp" #dtype;                                   \
    const char* device_name = "FakeDeviceName1";                             \
                                                                             \
    REGISTER_OP(op_name)                                                     \
        .Input("input1: double")                                             \
        .Input("input2: uint8")                                              \
        .Output("output1: uint8")                                            \
        .Attr("T: type");                                                    \
                                                                             \
    TF_KernelBuilder* builder = TF_NewKernelBuilder(                         \
        op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc); \
    TF_Status* status = TF_NewStatus();                                      \
    TF_KernelBuilder_TypeConstraint(builder, "T", TF_DataType::tf_type,      \
                                    status);                                 \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
    TF_RegisterKernelBuilder(node_name, builder, status);                    \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
                                                                             \
    TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);          \
    EXPECT_EQ(TF_OK, TF_GetCode(status));                                    \
    KernelList list;                                                         \
    list.ParseFromArray(buf->data, buf->length);                             \
    ASSERT_EQ(ExpectedString(#dtype), list.DebugString());                   \
                                                                             \
    TF_DeleteBuffer(buf);                                                    \
    TF_DeleteStatus(status);                                                 \
    TF_DeleteKernelBuilder(builder);                                         \
    ASSERT_TRUE(delete_called);                                              \
  }

TEST_KERNEL_TYPE_CONSTRAINT(TF_HALF, DT_HALF);
TEST_KERNEL_TYPE_CONSTRAINT(TF_BFLOAT16, DT_BFLOAT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_FLOAT, DT_FLOAT);
TEST_KERNEL_TYPE_CONSTRAINT(TF_DOUBLE, DT_DOUBLE);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT64, DT_UINT64);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT32, DT_UINT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT16, DT_UINT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_UINT8, DT_UINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_INT8, DT_INT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_INT32, DT_INT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_COMPLEX64, DT_COMPLEX64);
TEST_KERNEL_TYPE_CONSTRAINT(TF_COMPLEX128, DT_COMPLEX128);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT8, DT_QINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QUINT8, DT_QUINT8);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT32, DT_QINT32);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QINT16, DT_QINT16);
TEST_KERNEL_TYPE_CONSTRAINT(TF_QUINT16, DT_QUINT16);

TEST(TestKernel, TestHostMemory) {
  const char* node_name = "SomeNodeName";
  const char* op_name = "HostMemoryOp";
  const char* device_name = "FakeDeviceName1";

  REGISTER_OP(op_name)
      .Input("input1: double")
      .Input("input2: uint8")
      .Output("output1: uint8")
      .Attr("T: type");

  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, device_name, &MyCreateFunc, &MyComputeFunc, &MyDeleteFunc);
  TF_KernelBuilder_HostMemory(builder, "input2");
  TF_KernelBuilder_HostMemory(builder, "output1");
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(node_name, builder, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_Buffer* buf = TF_GetRegisteredKernelsForOp(op_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  KernelList list;
  list.ParseFromArray(buf->data, buf->length);
  const auto expected_str = R"str(kernel {
  op: "HostMemoryOp"
  device_type: "FakeDeviceName1"
  host_memory_arg: "input2"
  host_memory_arg: "output1"
}
)str";
  ASSERT_EQ(expected_str, list.DebugString());

  TF_DeleteBuffer(buf);
  TF_DeleteStatus(status);
  TF_DeleteKernelBuilder(builder);
  ASSERT_TRUE(delete_called);
}

class DeviceKernelOpTest : public OpsTestBase {
 protected:
  void SetupOp(const char* op_name, const char* node_name,
               void (*compute_func)(void*, TF_OpKernelContext*)) {
    TF_KernelBuilder* builder = TF_NewKernelBuilder(
        op_name, device_name_, nullptr, compute_func, nullptr);
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice(device_name_, {}, "/job:a/replica:0/task:0"));
    OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
#endif
    TF_ASSERT_OK(NodeDefBuilder(op_name, op_name).Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  const char* device_name_ = tensorflow::DEVICE_GPU;
#else
  const char* device_name_ = tensorflow::DEVICE_CPU;
#endif
};

// Validates that the tensor has shape and type corresponding to
// dims and dtype.
void validate_tensor(TF_Tensor* tensor, int64_t* dims, int64_t num_dims,
                     TF_DataType dtype);

// Copies data of length tensor_size_bytes from values to tensor.
template <typename T>
void set_tensor_data(TF_Tensor* tensor, T* values, size_t tensor_size_bytes,
                     TF_OpKernelContext* ctx);

REGISTER_OP("StreamOp").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestStream) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    SP_Stream stream = TF_GetStream(ctx, s);
    // Stream is always null if device is not a pluggable device. More test
    // cases will be added when pluggable device mechanism is supported.
    EXPECT_EQ(stream, nullptr);
    EXPECT_NE(TF_OK, TF_GetCode(s));
    TF_DeleteStatus(s);
  };

  SetupOp("StreamOp", "StreamOp", my_compute_func);
  TF_ASSERT_OK(RunOpKernel());
}

REGISTER_OP("AllocateOutputOp1").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateOutputSizeOne) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    // Allocate output
    TF_Status* s = TF_NewStatus();
    int64_t dim = 1;
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT);
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*len=*/tensor_size_bytes, s);
    validate_tensor(output, &dim, 1, TF_FLOAT);

    // Set output to 3
    float values[1] = {3.0f};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp1", "AllocateOutput1", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [1] values: 3>",
            output->DebugString(100));
}

REGISTER_OP("AllocateOutputOp0").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateEmptyOutput) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    // Allocate empty output
    int64_t dim = 0;
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*len=*/0, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp0", "AllocateOutput0", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [0] values: >",
            output->DebugString(100));
}

REGISTER_OP("AllocateOutputOp2x3").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateOutputSize2x3) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    // Allocate 2x3 output
    int64_t dim[2] = {2, 3};
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT) * 6;
    TF_Tensor* output = TF_AllocateOutput(
        /*context=*/ctx, /*index=*/0, /*dtype=*/TF_FLOAT, /*dims=*/dim,
        /*num_dims=*/2, /*len=*/tensor_size_bytes, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, dim, 2, TF_FLOAT);

    // Set output to [1 2 3 4 5 6]
    float values[6] = {1, 2, 3, 4, 5, 6};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateOutputOp2x3", "AllocateOutput2x3", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [2,3] values: [1 2 3][4 5 6]>",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp1").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempSizeOne) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    // Allocate scalar TF_Tensor
    TF_Status* s = TF_NewStatus();
    int64_t dim = 1;
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*allocator_attributes*/ &alloc_attrs, s);
    size_t tensor_size_bytes = TF_DataTypeSize(TF_FLOAT);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);

    // Set TF_Tensor value to 3
    float values[1] = {3.0f};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp1", "AllocateTemp1", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [1] values: 3>",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp0").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempEmpty) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    // Allocate empty TF_Tensor
    int64_t dim = 0;
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/&dim,
        /*num_dims=*/1, /*allocator_attributes*/ &alloc_attrs, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, &dim, 1, TF_FLOAT);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp0", "AllocateTemp0", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [0] values: >",
            output->DebugString(100));
}

REGISTER_OP("AllocateTempOp2x3").Output("output1: float");

TEST_F(DeviceKernelOpTest, TestAllocateTempSize2x3) {
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    size_t tensor_size_bytes = 6 * TF_DataTypeSize(TF_FLOAT);
    // Allocate 2x3 TF_Tensor
    int64_t dim[2] = {2, 3};
    TF_AllocatorAttributes alloc_attrs;
    alloc_attrs.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    alloc_attrs.on_host = 0;
#else
    alloc_attrs.on_host = 1;
#endif
    TF_Tensor* output = TF_AllocateTemp(
        /*context=*/ctx, /*dtype=*/TF_FLOAT, /*dims=*/dim,
        /*num_dims=*/2, /*allocator_attributes*/ &alloc_attrs, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    validate_tensor(output, dim, 2, TF_FLOAT);

    // Set TF_Tensor values to [1 2 3 4 5 6]
    float values[6] = {1, 2, 3, 4, 5, 6};
    set_tensor_data<float>(output, values, tensor_size_bytes, ctx);
    TF_SetOutput(ctx, 0, output, s);
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  SetupOp("AllocateTempOp2x3", "AllocateTempOp2x3", my_compute_func);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ("Tensor<type: float shape: [2,3] values: [1 2 3][4 5 6]>",
            output->DebugString(100));
}

TEST_F(DeviceKernelOpTest, TestForwardInputOrAllocateOutput) {
  const char* node_name = "TestForwardInputOrAllocateOutputKernel";
  const char* op_name = "BazOp";
  const char* device_name = "FakeDeviceName";

  REGISTER_OP(op_name)
      .Input("input1: float")
      .Input("input2: float")
      .Output("output1: float")
      .Attr("SomeDataTypeAttr: type");

  // A kernel whose Compute function that forwards a scalar input to output
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    int candidate_input_indices[1] = {0};
    int forwarded_input;
    int64_t output_dims[1] = {};
    TF_Tensor* output = TF_ForwardInputOrAllocateOutput(
        /*context=*/ctx, candidate_input_indices,
        /*num_candidate_input_indices=*/1,
        /*output_index=*/0, output_dims, /*output_num_dims=*/0,
        &forwarded_input, /*status=*/s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    EXPECT_EQ(forwarded_input, 0);
    EXPECT_EQ(TF_FLOAT, TF_TensorType(output));
    EXPECT_EQ(0, TF_NumDims(output));
    TF_DeleteStatus(s);
    TF_DeleteTensor(output);
  };

  TF_KernelBuilder* builder = TF_NewKernelBuilder(op_name, device_name, nullptr,
                                                  my_compute_func, nullptr);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr);
    p.device = &dummy_device;
    AllocatorAttributes alloc_attrs;
    p.output_attr_array = &alloc_attrs;

    Tensor t(123.0f);

    gtl::InlinedVector<TensorValue, 4> inputs;
    // GetFakeKernel requires a NodeDef with two inputs
    inputs.emplace_back(&t);
    inputs.emplace_back();
    p.inputs = &inputs;

    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, node_name, &status);
    TF_EXPECT_OK(status);
    ASSERT_NE(nullptr, kernel.get());

    p.op_kernel = kernel.get();
    OpKernelContext ctx(&p);
    kernel->Compute(&ctx);
    ASSERT_EQ(123, ctx.mutable_output(0)->scalar<float>()());
  }
}

void validate_tensor(TF_Tensor* tensor, int64_t* dims, int64_t num_dims,
                     TF_DataType dtype) {
  EXPECT_EQ(TF_FLOAT, TF_TensorType(tensor));
  EXPECT_EQ(num_dims, TF_NumDims(tensor));
  for (int i = 0; i < num_dims; ++i) {
    EXPECT_EQ(dims[i], TF_Dim(tensor, i));
  }
}

template <typename T>
void set_tensor_data(TF_Tensor* tensor, T* values, size_t tensor_size_bytes,
                     TF_OpKernelContext* ctx) {
  T* data = reinterpret_cast<T*>(TF_TensorData(tensor));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  OpKernelContext* cc_ctx = reinterpret_cast<OpKernelContext*>(ctx);
  cc_ctx->eigen_gpu_device().memcpyHostToDevice(data, values,
                                                tensor_size_bytes);
#else
  memcpy(data, values, tensor_size_bytes);
#endif
}
}  // namespace tensorflow
