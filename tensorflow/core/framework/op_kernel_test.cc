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

#include "tensorflow/core/framework/op_kernel.h"

#include <memory>
#include <utility>
#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/version.h"

class DummyKernel : public tensorflow::OpKernel {
 public:
  explicit DummyKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(tensorflow::OpKernelContext* context) override {}
};

// Test that registration works outside a namespace.
REGISTER_OP("Test1").Input("a: float").Input("b: int32").Output("o: uint8");
REGISTER_KERNEL_BUILDER(Name("Test1").Device(tensorflow::DEVICE_CPU),
                        DummyKernel);

namespace foo {
bool match_signature_ = false;

// Test that registration works inside a different namespace.
class TestOp2 : public ::tensorflow::OpKernel {
 public:
  explicit TestOp2(::tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    ::tensorflow::Status status = context->MatchSignature(
        {::tensorflow::DT_INT32}, {::tensorflow::DT_INT32});
    match_signature_ = status.ok();
    context->SetStatus(status);
  }
  void Compute(::tensorflow::OpKernelContext* context) override {}
};

REGISTER_OP("Test2").Input("i: T").Output("o: T").Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("Test2")
                            .Device(::tensorflow::DEVICE_GPU)
                            .HostMemory("i")
                            .HostMemory("o"),
                        TestOp2);
}  // namespace foo

namespace tensorflow {

// Two operations with the same name but different devices.
REGISTER_OP("Test3").Input("a: T").Input("b: T").Attr("T: type");

class TestOp3Cpu : public tensorflow::OpKernel {
 public:
  explicit TestOp3Cpu(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

REGISTER_KERNEL_BUILDER(
    Name("Test3").Device(DEVICE_CPU).TypeConstraint<int8>("T"), TestOp3Cpu);

namespace {

class TestOp3Gpu : public tensorflow::OpKernel {
 public:
  explicit TestOp3Gpu(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

REGISTER_KERNEL_BUILDER(
    Name("Test3").Device(DEVICE_GPU).TypeConstraint<float>("T"), TestOp3Cpu);

// An Op registered for both
REGISTER_OP("Test4").Input("i: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("Test4").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("Test4").Device(DEVICE_GPU), DummyKernel);

static std::vector<DeviceType> DeviceTypes() {
  return {DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)};
}

class OpKernelTest : public ::testing::Test {
 public:
  OpKernelTest() : device_(Env::Default()) {}

 protected:
  NodeDef CreateNodeDef(const string& op_type, const DataTypeVector& inputs) {
    NodeDefBuilder builder(op_type + "-op", op_type);
    for (DataType dt : inputs) {
      builder.Input(FakeInput(dt));
    }
    NodeDef node_def;
    TF_CHECK_OK(builder.Finalize(&node_def));
    return node_def;
  }

  void ExpectEqual(const string& what, const DataTypeVector& expected,
                   const DataTypeVector& observed) {
    EXPECT_EQ(expected.size(), observed.size()) << what;
    const size_t size = std::min(expected.size(), observed.size());
    for (size_t i = 0; i < size; ++i) {
      bool match = TypesCompatible(expected[i], observed[i]);
      EXPECT_TRUE(match) << what << " i:" << i << ", expected: " << expected[i]
                         << ", observed: " << observed[i];
    }
  }

  void ExpectSuccess(const string& op_type, DeviceType device_type,
                     const DataTypeVector& inputs,
                     const DataTypeVector& outputs) {
    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        std::move(device_type), &device_, cpu_allocator(),
        CreateNodeDef(op_type, inputs), TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(op != nullptr);
    if (op != nullptr) {
      ExpectEqual("inputs", op->input_types(), inputs);
      ExpectEqual("outputs", op->output_types(), outputs);
    }
  }

  void ExpectFailure(const string& ascii_node_def, DeviceType device_type,
                     error::Code code) {
    NodeDef node_def;
    protobuf::TextFormat::ParseFromString(ascii_node_def, &node_def);
    Status status;
    std::unique_ptr<OpKernel> op(
        CreateOpKernel(std::move(device_type), &device_, cpu_allocator(),
                       node_def, TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.error_message();
      EXPECT_EQ(code, status.code());
    }
  }

 private:
  DeviceBase device_;
};

TEST_F(OpKernelTest, SuccessCpu) {
  ExpectSuccess("Test1", DEVICE_CPU, {DT_FLOAT, DT_INT32}, {DT_UINT8});
  ExpectSuccess("Test1", DEVICE_CPU, {DT_FLOAT_REF, DT_INT32}, {DT_UINT8});
}

TEST_F(OpKernelTest, SuccessGpu) {
  foo::match_signature_ = false;
  ExpectSuccess("Test2", DEVICE_GPU, {DT_INT32}, {DT_INT32});
  EXPECT_TRUE(foo::match_signature_);
}

TEST_F(OpKernelTest, SuccessBothCpuAndGpu) {
  ExpectSuccess("Test3", DEVICE_CPU, {DT_INT8, DT_INT8}, {});
  ExpectSuccess("Test3", DEVICE_GPU, {DT_FLOAT, DT_FLOAT}, {});
}

TEST_F(OpKernelTest, CpuTypeRegistered) {
  NodeDef ndef = CreateNodeDef("Test1", {DT_FLOAT, DT_INT32});
  DeviceTypeVector devs;
  TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
  EXPECT_EQ(1, devs.size());
  EXPECT_EQ(DeviceType(DEVICE_CPU), devs[0]);
}

TEST_F(OpKernelTest, CpuAndGpuTypeRegistered) {
  {
    // Try a node def of an op that is registered for a specific type
    // only on CPU.
    NodeDef ndef = CreateNodeDef("Test3", {DT_INT8, DT_INT8});
    DeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(1, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[0]);
  }
  {
    // Try a node def of an op that is registered for a specific type
    // only on GPU.
    NodeDef ndef = CreateNodeDef("Test3", {DT_FLOAT, DT_FLOAT});
    DeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(1, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[0]);
  }
  {
    // Try a node def of an op that is only registered for other types.
    NodeDef ndef = CreateNodeDef("Test3", {DT_STRING, DT_STRING});
    DeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(0, devs.size());
  }

  {
    // Try a node def of an op that is registered for both.
    NodeDef ndef = CreateNodeDef("Test4", {DT_FLOAT});
    DeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(2, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[0]);
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[1]);
  }
}

TEST_F(OpKernelTest, NotFound) {
  const auto not_found = error::NOT_FOUND;
  // Something with that op type name exists, but only with a
  // different DeviceType.
  ExpectFailure(CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}).DebugString(),
                DEVICE_GPU, not_found);
  ExpectFailure(CreateNodeDef("Test3", {DT_INT8, DT_INT8}).DebugString(),
                DEVICE_GPU, not_found);
  ExpectFailure(CreateNodeDef("Test3", {DT_FLOAT, DT_FLOAT}).DebugString(),
                DEVICE_CPU, not_found);

  // No kernel with that signature registered.
  ExpectFailure(CreateNodeDef("Test3", {DT_INT32, DT_INT32}).DebugString(),
                DEVICE_GPU, not_found);

  // Nothing with that op type name exists.
  ExpectFailure("name: 'NF' op: 'Testnotfound'", DEVICE_CPU, not_found);
  ExpectFailure("name: 'NF' op: 'Testnotfound'", DEVICE_GPU, not_found);
}

TEST_F(OpKernelTest, TooFewInputs) {
  const auto invalid = error::INVALID_ARGUMENT;
  NodeDef node_def = CreateNodeDef("Test1", {DT_FLOAT, DT_INT32});
  node_def.clear_input();
  ExpectFailure(node_def.DebugString(), DEVICE_CPU, invalid);
  node_def.add_input("a");
  ExpectFailure(node_def.DebugString(), DEVICE_CPU, invalid);
}

TEST_F(OpKernelTest, TooManyInputs) {
  const auto invalid = error::INVALID_ARGUMENT;
  NodeDef node_def = CreateNodeDef("Test1", {DT_FLOAT, DT_INT32});
  node_def.add_input("c");
  ExpectFailure(node_def.DebugString(), DEVICE_CPU, invalid);
}

TEST_F(OpKernelTest, MatchSignatureFailes) {
  const auto invalid = error::INVALID_ARGUMENT;
  foo::match_signature_ = true;
  ExpectFailure(CreateNodeDef("Test2", {DT_FLOAT}).DebugString(), DEVICE_GPU,
                invalid);
  EXPECT_FALSE(foo::match_signature_);
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

TEST_F(OpKernelTest, SaveTempFalse) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  params.record_tensor_accesses = false;
  params.device = new DummyDevice(env, params.record_tensor_accesses);
  Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  OpKernelContext* ctx = new OpKernelContext(&params);

  Tensor t;
  TF_EXPECT_OK(ctx->allocate_temp(DT_FLOAT, TensorShape(), &t));

  TensorReferenceVector referenced_tensors;
  ctx->retrieve_accessed_tensors(&referenced_tensors);
  EXPECT_EQ(0, referenced_tensors.size());

  delete ctx;
  delete params.device;
}

TEST_F(OpKernelTest, SaveTempTrue) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  params.record_tensor_accesses = true;
  params.device = new DummyDevice(env, params.record_tensor_accesses);
  Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  OpKernelContext* ctx = new OpKernelContext(&params);

  Tensor t;
  TF_EXPECT_OK(ctx->allocate_temp(DT_FLOAT, TensorShape(), &t));

  TensorReferenceVector referenced_tensors;
  ctx->retrieve_accessed_tensors(&referenced_tensors);
  EXPECT_EQ(1, referenced_tensors.size());
  for (auto& ref : referenced_tensors) {
    ref.Unref();
  }

  delete ctx;
  delete params.device;
}

TEST_F(OpKernelTest, InputDtype) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  params.record_tensor_accesses = false;
  params.device = new DummyDevice(env, params.record_tensor_accesses);
  Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({}));
  Tensor b(DT_INT32, TensorShape({}));
  Tensor c(DT_UINT8, TensorShape({}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b),
                                            TensorValue(&c)};
  params.inputs = &inputs;
  OpKernelContext* ctx = new OpKernelContext(&params);

  DataType dtype;
  EXPECT_FALSE(ctx->input_dtype("non_existent_input", &dtype).ok());
  ASSERT_TRUE(ctx->input_dtype("a", &dtype).ok());
  EXPECT_EQ(dtype, DT_FLOAT);
  ASSERT_TRUE(ctx->input_dtype("b", &dtype).ok());
  EXPECT_EQ(dtype, DT_INT32);
  delete ctx;
  delete params.device;
}

class OpKernelBuilderTest : public ::testing::Test {
 protected:
  // Each attr is described by a "name|type|value".
  NodeDef CreateNodeDef(const string& op_type,
                        const std::vector<string>& attrs) {
    NodeDef node_def;
    node_def.set_name(op_type + "-op");
    node_def.set_op(op_type);
    for (const string& attr_desc : attrs) {
      std::vector<string> parts = str_util::Split(attr_desc, '|');
      CHECK_EQ(parts.size(), 3);
      AttrValue attr_value;
      CHECK(ParseAttrValue(parts[1], parts[2], &attr_value)) << attr_desc;
      node_def.mutable_attr()->insert(
          AttrValueMap::value_type(parts[0], attr_value));
    }
    return node_def;
  }

  std::unique_ptr<OpKernel> ExpectSuccess(const string& op_type,
                                          const DeviceType& device_type,
                                          const std::vector<string>& attrs,
                                          DataTypeSlice input_types = {}) {
    Status status;
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel()
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(op != nullptr);
    if (op != nullptr) {
      EXPECT_EQ(input_types.size(), op->num_inputs());
      EXPECT_EQ(0, op->num_outputs());
    }

    // Test SupportedDeviceTypesForNode()
    DeviceTypeVector devices;
    TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
    bool found = false;
    for (const DeviceType& dt : devices) {
      if (dt == device_type) {
        found = true;
      }
    }
    EXPECT_TRUE(found) << "Missing " << device_type << " from "
                       << devices.size() << " devices.";

    // In case the caller wants to use the OpKernel
    return op;
  }

  void ExpectFailure(const string& op_type, const DeviceType& device_type,
                     const std::vector<string>& attrs, error::Code code) {
    Status status;
    const NodeDef def = CreateNodeDef(op_type, attrs);
    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel().
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.error_message();
      EXPECT_EQ(code, status.code());

      // Test SupportedDeviceTypesForNode().
      DeviceTypeVector devices;
      if (errors::IsNotFound(status)) {
        TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
        for (const DeviceType& dt : devices) {
          EXPECT_NE(dt, device_type);
        }
      } else {
        Status status2 =
            SupportedDeviceTypesForNode(DeviceTypes(), def, &devices);
        EXPECT_EQ(status.code(), status2.code());
      }
    }
  }

  string GetKernelClassName(const string& op_type,
                            const DeviceType& device_type,
                            const std::vector<string>& attrs,
                            DataTypeSlice input_types = {}) {
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    const KernelDef* kernel_def = nullptr;
    string kernel_class_name;
    const Status status =
        FindKernelDef(device_type, def, &kernel_def, &kernel_class_name);
    if (status.ok()) {
      return kernel_class_name;
    } else if (errors::IsNotFound(status)) {
      return "not found";
    } else {
      return status.ToString();
    }
  }
};

REGISTER_OP("BuildCPU");
REGISTER_KERNEL_BUILDER(Name("BuildCPU").Device(DEVICE_CPU), DummyKernel);

TEST_F(OpKernelBuilderTest, BuilderCPU) {
  ExpectSuccess("BuildCPU", DEVICE_CPU, {});
  EXPECT_EQ("DummyKernel", GetKernelClassName("BuildCPU", DEVICE_CPU, {}));
  ExpectFailure("BuildCPU", DEVICE_GPU, {}, error::NOT_FOUND);
  EXPECT_EQ("not found", GetKernelClassName("BuildCPU", DEVICE_GPU, {}));
}

REGISTER_OP("BuildGPU");
REGISTER_KERNEL_BUILDER(Name("BuildGPU").Device(DEVICE_GPU), DummyKernel);

TEST_F(OpKernelBuilderTest, BuilderGPU) {
  ExpectFailure("BuildGPU", DEVICE_CPU, {}, error::NOT_FOUND);
  ExpectSuccess("BuildGPU", DEVICE_GPU, {});
}

REGISTER_OP("BuildBoth");
REGISTER_KERNEL_BUILDER(Name("BuildBoth").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("BuildBoth").Device(DEVICE_GPU), DummyKernel);

TEST_F(OpKernelBuilderTest, BuilderBoth) {
  ExpectSuccess("BuildBoth", DEVICE_CPU, {});
  ExpectSuccess("BuildBoth", DEVICE_GPU, {});
}

REGISTER_OP("BuildTypeAttr").Attr("T: type");
REGISTER_KERNEL_BUILDER(
    Name("BuildTypeAttr").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    DummyKernel);

TEST_F(OpKernelBuilderTest, BuilderTypeAttr) {
  ExpectSuccess("BuildTypeAttr", DEVICE_CPU, {"T|type|DT_FLOAT"});
  ExpectFailure("BuildTypeAttr", DEVICE_CPU, {"T|type|DT_BOOL"},
                error::NOT_FOUND);
  ExpectFailure("BuildTypeAttr", DEVICE_CPU, {}, error::INVALID_ARGUMENT);
  ExpectFailure("BuildTypeAttr", DEVICE_CPU, {"T|int|7"},
                error::INVALID_ARGUMENT);
}

REGISTER_OP("BuildTypeListAttr").Attr("T: list(type)");
REGISTER_KERNEL_BUILDER(
    Name("BuildTypeListAttr").Device(DEVICE_CPU).TypeConstraint<bool>("T"),
    DummyKernel);

TEST_F(OpKernelBuilderTest, BuilderTypeListAttr) {
  ExpectSuccess("BuildTypeListAttr", DEVICE_CPU, {"T|list(type)|[]"});
  EXPECT_EQ("DummyKernel", GetKernelClassName("BuildTypeListAttr", DEVICE_CPU,
                                              {"T|list(type)|[]"}));

  ExpectSuccess("BuildTypeListAttr", DEVICE_CPU, {"T|list(type)|[DT_BOOL]"});
  EXPECT_EQ("DummyKernel", GetKernelClassName("BuildTypeListAttr", DEVICE_CPU,
                                              {"T|list(type)|[]"}));

  ExpectSuccess("BuildTypeListAttr", DEVICE_CPU,
                {"T|list(type)|[DT_BOOL, DT_BOOL]"});

  ExpectFailure("BuildTypeListAttr", DEVICE_CPU, {"T|list(type)|[DT_FLOAT]"},
                error::NOT_FOUND);
  EXPECT_EQ("not found", GetKernelClassName("BuildTypeListAttr", DEVICE_CPU,
                                            {"T|list(type)|[DT_FLOAT]"}));

  ExpectFailure("BuildTypeListAttr", DEVICE_CPU, {}, error::INVALID_ARGUMENT);
  EXPECT_TRUE(str_util::StrContains(
      GetKernelClassName("BuildTypeListAttr", DEVICE_CPU, {}),
      "Invalid argument: "));

  ExpectFailure("BuildTypeListAttr", DEVICE_CPU, {"T|int|7"},
                error::INVALID_ARGUMENT);
}

REGISTER_OP("DuplicateKernel");
REGISTER_KERNEL_BUILDER(Name("DuplicateKernel").Device(DEVICE_CPU),
                        DummyKernel);
REGISTER_KERNEL_BUILDER(Name("DuplicateKernel").Device(DEVICE_CPU),
                        DummyKernel);

TEST_F(OpKernelBuilderTest, DuplicateKernel) {
  const NodeDef ndef = CreateNodeDef("DuplicateKernel", {});
  DeviceTypeVector devs;
  Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(
      status.error_message(), "Multiple OpKernel registrations match NodeDef"));

  ExpectFailure("DuplicateKernel", DEVICE_CPU, {}, error::INVALID_ARGUMENT);
}

REGISTER_OP("DuplicateKernelForT").Attr("T: type");
REGISTER_KERNEL_BUILDER(
    Name("DuplicateKernelForT").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    DummyKernel);
REGISTER_KERNEL_BUILDER(
    Name("DuplicateKernelForT").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    DummyKernel);

TEST_F(OpKernelBuilderTest, DuplicateKernelForT) {
  const NodeDef ndef =
      CreateNodeDef("DuplicateKernelForT", {"T|type|DT_FLOAT"});
  DeviceTypeVector devs;
  Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(
      status.error_message(), "Multiple OpKernel registrations match NodeDef"));

  ExpectFailure("DuplicateKernelForT", DEVICE_CPU, {"T|type|DT_FLOAT"},
                error::INVALID_ARGUMENT);
  ExpectFailure("DuplicateKernelForT", DEVICE_CPU, {"T|type|DT_BOOL"},
                error::NOT_FOUND);
}

REGISTER_OP("BadConstraint").Attr("dtype: type");
REGISTER_KERNEL_BUILDER(Name("BadConstraint")
                            .Device(DEVICE_CPU)
                            // Mistake: "T" should be "dtype".
                            .TypeConstraint<float>("T"),
                        DummyKernel);

TEST_F(OpKernelBuilderTest, BadConstraint) {
  const NodeDef ndef = CreateNodeDef("BadConstraint", {});
  DeviceTypeVector devs;
  Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(),
                            "OpKernel 'BadConstraint' has constraint on attr "
                            "'T' not in NodeDef"));

  ExpectFailure("BadConstraint", DEVICE_CPU, {"dtype|type|DT_FLOAT"},
                error::INVALID_ARGUMENT);
}

REGISTER_OP("ListOut").Output("a: int32").Output("b: T").Attr("T: list(type)");
REGISTER_KERNEL_BUILDER(Name("ListOut").Device(tensorflow::DEVICE_CPU),
                        DummyKernel);

TEST_F(OpKernelBuilderTest, OpOutputList) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  params.record_tensor_accesses = false;
  std::unique_ptr<DummyDevice> device(
      new DummyDevice(env, params.record_tensor_accesses));
  params.device = device.get();
  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(
      DEVICE_CPU, params.device, cpu_allocator(),
      CreateNodeDef("ListOut", {"T|list(type)|[DT_FLOAT, DT_INT32]"}),
      TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok()) << status.ToString();
  params.op_kernel = op.get();
  gtl::InlinedVector<TensorValue, 4> inputs{};
  params.inputs = &inputs;
  std::unique_ptr<OpKernelContext> ctx(new OpKernelContext(&params));

  EXPECT_EQ(DT_INT32, ctx->expected_output_dtype(0));
  OpOutputList out_list;
  EXPECT_FALSE(ctx->output_list("non_existent_output", &out_list).ok());
  ASSERT_TRUE(ctx->output_list("b", &out_list).ok());
  EXPECT_EQ(DT_FLOAT, out_list.expected_output_dtype(0));
  EXPECT_EQ(DT_INT32, out_list.expected_output_dtype(1));
}

class GetAttrKernel : public ::tensorflow::OpKernel {
 public:
  explicit GetAttrKernel(OpKernelConstruction* context) : OpKernel(context) {
    string attr_name;
    OP_REQUIRES_OK(context, context->GetAttr("attr_name", &attr_name));

    status.emplace_back("s", context->GetAttr(attr_name, &s));
    status.emplace_back("s_list", context->GetAttr(attr_name, &s_list));
    status.emplace_back("i", context->GetAttr(attr_name, &i));
    status.emplace_back("i_list", context->GetAttr(attr_name, &i_list));
    status.emplace_back("i32", context->GetAttr(attr_name, &i32));
    status.emplace_back("i32_list", context->GetAttr(attr_name, &i32_list));
    status.emplace_back("f", context->GetAttr(attr_name, &f));
    status.emplace_back("f_list", context->GetAttr(attr_name, &f_list));
    status.emplace_back("b", context->GetAttr(attr_name, &b));
    status.emplace_back("b_list", context->GetAttr(attr_name, &b_list));
    status.emplace_back("type", context->GetAttr(attr_name, &type));
    status.emplace_back("type_list", context->GetAttr(attr_name, &type_list));
    status.emplace_back("type_vector",
                        context->GetAttr(attr_name, &type_vector));
    status.emplace_back("shape_proto",
                        context->GetAttr(attr_name, &shape_proto));
    status.emplace_back("shape_proto_list",
                        context->GetAttr(attr_name, &shape_proto_list));
    status.emplace_back("shape", context->GetAttr(attr_name, &shape));
    status.emplace_back("shape_list", context->GetAttr(attr_name, &shape_list));
  }
  void Compute(::tensorflow::OpKernelContext* context) override {}

  void ExpectOk(std::initializer_list<string> keys) {
    for (const auto& key_status : status) {
      // Only the status for keys in "keys" should be ok().
      bool in_keys = false;
      for (const string& key : keys) {
        if (key_status.first == key) {
          in_keys = true;
        }
      }
      EXPECT_EQ(in_keys, key_status.second.ok())
          << "key_status: " << key_status.first << ", " << key_status.second;
    }
  }

  string s;
  std::vector<string> s_list;
  int64 i;
  std::vector<int64> i_list;
  int32 i32;
  std::vector<int32> i32_list;
  float f;
  std::vector<float> f_list;
  bool b;
  std::vector<bool> b_list;
  DataType type;
  std::vector<DataType> type_list;
  DataTypeVector type_vector;
  TensorShapeProto shape_proto;
  std::vector<TensorShapeProto> shape_proto_list;
  TensorShape shape;
  std::vector<TensorShape> shape_list;
  std::vector<std::pair<string, Status>> status;
};

class GetAttrTest : public OpKernelBuilderTest {};

REGISTER_OP("GetAttrStringList")
    .Attr("attr_name: string")
    .Attr("a: list(string)");
REGISTER_KERNEL_BUILDER(Name("GetAttrStringList").Device(DEVICE_CPU),
                        GetAttrKernel);

TEST_F(GetAttrTest, StringList) {
  std::unique_ptr<OpKernel> op_kernel =
      ExpectSuccess("GetAttrStringList", DEVICE_CPU,
                    {"attr_name|string|'a'", "a|list(string)|['foo', 'bar']"});
  auto* get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"s_list"});
  EXPECT_EQ(std::vector<string>({"foo", "bar"}), get_attr_kernel->s_list);

  op_kernel = ExpectSuccess("GetAttrStringList", DEVICE_CPU,
                            {"attr_name|string|'b'", "a|list(string)|['baz']"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({});
  EXPECT_TRUE(get_attr_kernel->s_list.empty());
}

REGISTER_OP("GetAttrInt")
    .Attr("attr_name: string")
    .Attr("a: int")
    .Attr("b: list(int)");
REGISTER_KERNEL_BUILDER(Name("GetAttrInt").Device(DEVICE_CPU), GetAttrKernel);

TEST_F(GetAttrTest, Int) {
  std::unique_ptr<OpKernel> op_kernel = ExpectSuccess(
      "GetAttrInt", DEVICE_CPU,
      {"attr_name|string|'a'", "a|int|35", "b|list(int)|[-1, 2, -4]"});
  auto* get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"i", "i32"});
  EXPECT_EQ(35, get_attr_kernel->i);
  EXPECT_EQ(35, get_attr_kernel->i32);

  op_kernel = ExpectSuccess(
      "GetAttrInt", DEVICE_CPU,
      {"attr_name|string|'b'", "a|int|35", "b|list(int)|[-1, 2, -4]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"i_list", "i32_list"});
  EXPECT_EQ(std::vector<int64>({-1, 2, -4}), get_attr_kernel->i_list);
  EXPECT_EQ(std::vector<int32>({-1, 2, -4}), get_attr_kernel->i32_list);

  // 8589934592 == 2^33, too big to fit in an int32
  op_kernel = ExpectSuccess("GetAttrInt", DEVICE_CPU,
                            {"attr_name|string|'a'", "a|int|8589934592",
                             "b|list(int)|[-8589934592]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"i"});  // no i32
  EXPECT_EQ(8589934592ll, get_attr_kernel->i);
  for (const auto& key_status : get_attr_kernel->status) {
    if (key_status.first == "i32") {
      EXPECT_EQ(error::INVALID_ARGUMENT, key_status.second.code());
      EXPECT_EQ("Attr a has value 8589934592 out of range for an int32",
                key_status.second.error_message());
    }
  }

  op_kernel = ExpectSuccess("GetAttrInt", DEVICE_CPU,
                            {"attr_name|string|'b'", "a|int|8589934592",
                             "b|list(int)|[-8589934592]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"i_list"});  // no i32_list
  EXPECT_EQ(std::vector<int64>({-8589934592ll}), get_attr_kernel->i_list);
  for (const auto& key_status : get_attr_kernel->status) {
    if (key_status.first == "i32_list") {
      EXPECT_EQ(error::INVALID_ARGUMENT, key_status.second.code());
      EXPECT_EQ("Attr b has value -8589934592 out of range for an int32",
                key_status.second.error_message());
    }
  }
}

REGISTER_OP("GetAttrShape")
    .Attr("attr_name: string")
    .Attr("a: shape")
    .Attr("b: list(shape)");
REGISTER_KERNEL_BUILDER(Name("GetAttrShape").Device(DEVICE_CPU), GetAttrKernel);

TEST_F(GetAttrTest, Shape) {
  std::unique_ptr<OpKernel> op_kernel = ExpectSuccess(
      "GetAttrShape", DEVICE_CPU,
      {"attr_name|string|'a'", "a|shape|{ dim { size: 3 } }",
       "b|list(shape)|[{ dim { size:2 } }, { dim { size: 4 } }]"});
  auto* get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"shape", "shape_proto"});
  EXPECT_EQ(get_attr_kernel->shape_proto.ShortDebugString(), "dim { size: 3 }");
  EXPECT_EQ("[3]", get_attr_kernel->shape.DebugString());

  op_kernel = ExpectSuccess(
      "GetAttrShape", DEVICE_CPU,
      {"attr_name|string|'b'", "a|shape|{ dim { size: 3 } }",
       "b|list(shape)|[{ dim { size:2 } }, { dim { size: 4 } }]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"shape_list", "shape_proto_list"});
  ASSERT_EQ(2, get_attr_kernel->shape_proto_list.size());
  EXPECT_EQ(get_attr_kernel->shape_proto_list[0].ShortDebugString(),
            "dim { size: 2 }");
  EXPECT_EQ(get_attr_kernel->shape_proto_list[1].ShortDebugString(),
            "dim { size: 4 }");
  ASSERT_EQ(2, get_attr_kernel->shape_list.size());
  EXPECT_EQ("[2]", get_attr_kernel->shape_list[0].DebugString());
  EXPECT_EQ("[4]", get_attr_kernel->shape_list[1].DebugString());
}

REGISTER_OP("GetAttrType").Attr("attr_name: string").Attr("a: type");
REGISTER_KERNEL_BUILDER(Name("GetAttrType").Device(DEVICE_CPU), GetAttrKernel);

TEST_F(GetAttrTest, Type) {
  std::unique_ptr<OpKernel> op_kernel = ExpectSuccess(
      "GetAttrType", DEVICE_CPU, {"attr_name|string|'a'", "a|type|DT_FLOAT"});
  auto* get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"type"});
  EXPECT_EQ(DT_FLOAT, get_attr_kernel->type);
}

REGISTER_OP("GetAttrTypeList").Attr("attr_name: string").Attr("a: list(type)");
REGISTER_KERNEL_BUILDER(Name("GetAttrTypeList").Device(DEVICE_CPU),
                        GetAttrKernel);

TEST_F(GetAttrTest, TypeList) {
  std::unique_ptr<OpKernel> op_kernel = ExpectSuccess(
      "GetAttrTypeList", DEVICE_CPU,
      {"attr_name|string|'a'", "a|list(type)|[DT_INT32, DT_BOOL]"});
  auto* get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());

  get_attr_kernel->ExpectOk({"type_list", "type_vector"});
  ASSERT_EQ(2, get_attr_kernel->type_list.size());
  EXPECT_EQ(DT_INT32, get_attr_kernel->type_list[0]);
  EXPECT_EQ(DT_BOOL, get_attr_kernel->type_list[1]);
  ASSERT_EQ(2, get_attr_kernel->type_vector.size());
  EXPECT_EQ(DT_INT32, get_attr_kernel->type_vector[0]);
  EXPECT_EQ(DT_BOOL, get_attr_kernel->type_vector[1]);
}

class BaseKernel : public ::tensorflow::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(::tensorflow::OpKernelContext* context) override {}
  virtual int Which() const = 0;
};

template <int WHICH>
class LabeledKernel : public BaseKernel {
 public:
  using BaseKernel::BaseKernel;
  int Which() const override { return WHICH; }
};

class LabelTest : public OpKernelBuilderTest {};

REGISTER_OP("LabeledKernel");
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU),
                        LabeledKernel<0>);
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU).Label("one"),
                        LabeledKernel<1>);
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU).Label("dupe"),
                        LabeledKernel<2>);
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU).Label("dupe"),
                        LabeledKernel<3>);

TEST_F(LabelTest, Default) {
  std::unique_ptr<OpKernel> op_kernel =
      ExpectSuccess("LabeledKernel", DEVICE_CPU, {});
  auto* get_labeled_kernel = static_cast<BaseKernel*>(op_kernel.get());
  EXPECT_EQ(0, get_labeled_kernel->Which());

  EXPECT_EQ("LabeledKernel<0>",
            GetKernelClassName("LabeledKernel", DEVICE_CPU, {}));
}

TEST_F(LabelTest, Specified) {
  std::unique_ptr<OpKernel> op_kernel =
      ExpectSuccess("LabeledKernel", DEVICE_CPU, {"_kernel|string|'one'"});
  auto* get_labeled_kernel = static_cast<BaseKernel*>(op_kernel.get());
  EXPECT_EQ(1, get_labeled_kernel->Which());
  EXPECT_EQ("LabeledKernel<1>", GetKernelClassName("LabeledKernel", DEVICE_CPU,
                                                   {"_kernel|string|'one'"}));
}

TEST_F(LabelTest, Duplicate) {
  ExpectFailure("LabeledKernel", DEVICE_CPU, {"_kernel|string|'dupe'"},
                error::INVALID_ARGUMENT);
}

void BM_InputRangeHelper(int iters, const NodeDef& node_def,
                         const char* input_name, int expected_start,
                         int expected_stop) {
  Status status;
  std::unique_ptr<DummyDevice> device(new DummyDevice(Env::Default(), false));

  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    int start;
    int stop;
    TF_CHECK_OK(op->InputRange(input_name, &start, &stop));
    EXPECT_EQ(expected_start, start);
    EXPECT_EQ(expected_stop, stop);
  }
  testing::StopTiming();
}

REGISTER_KERNEL_BUILDER(Name("ConcatV2").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("Select").Device(DEVICE_CPU), DummyKernel);

void BM_ConcatInputRange(int iters) {
  testing::StopTiming();

  // Create a ConcatV2 NodeDef with 4 inputs (plus the axis).
  NodeDef node_def;
  node_def.set_name("concat-op");
  node_def.set_op("ConcatV2");
  AttrValue attr_N;
  attr_N.set_i(4);
  AttrValue attr_T;
  attr_T.set_type(DT_FLOAT);
  AttrValue attr_Tidx;
  attr_Tidx.set_type(DT_INT32);
  node_def.mutable_attr()->insert({"N", attr_N});
  node_def.mutable_attr()->insert({"T", attr_T});
  node_def.mutable_attr()->insert({"Tidx", attr_Tidx});
  for (size_t i = 0; i < 5; ++i) {
    node_def.add_input(strings::StrCat("a:", i));
  }

  BM_InputRangeHelper(iters, node_def, "values", 0, 4);
}

void BM_SelectInputRange(int iters) {
  testing::StopTiming();

  // Create a Select NodeDef with 3 inputs.
  NodeDef node_def;
  node_def.set_name("select-op");
  node_def.set_op("Select");
  AttrValue attr_T;
  attr_T.set_type(DT_FLOAT);
  node_def.mutable_attr()->insert({"T", attr_T});
  for (size_t i = 0; i < 3; ++i) {
    node_def.add_input(strings::StrCat("a:", i));
  }

  BM_InputRangeHelper(iters, node_def, "condition", 0, 1);
}

BENCHMARK(BM_ConcatInputRange);
BENCHMARK(BM_SelectInputRange);

TEST(RegisteredKernels, CanCallGetAllRegisteredKernels) {
  auto all_registered_kernels = GetAllRegisteredKernels();
  auto has_name_test1 = [](const KernelDef& k) { return k.op() == "Test1"; };

  // Verify we can find the "Test1" op registered above
  auto test1_it = std::find_if(all_registered_kernels.begin(),
                               all_registered_kernels.end(), has_name_test1);
  ASSERT_NE(test1_it, all_registered_kernels.end());
  EXPECT_EQ(test1_it->device_type(), "CPU");

  // Verify there was just one kernel
  ++test1_it;
  EXPECT_EQ(
      std::find_if(test1_it, all_registered_kernels.end(), has_name_test1),
      all_registered_kernels.end());
}

// Simple test just to check we can call LogAllRegisteredKernels
TEST(RegisteredKernels, CanLogAllRegisteredKernels) {
  tensorflow::LogAllRegisteredKernels();
}

}  // namespace
}  // namespace tensorflow
