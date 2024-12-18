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

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel_test_base.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/protobuf.h"

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
    absl::Status status = context->MatchSignature({::tensorflow::DT_INT32},
                                                  {::tensorflow::DT_INT32});
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

// Kernels with different priorities.
REGISTER_OP("Test5").Input("a: T").Input("b: T").Attr("T: type");

REGISTER_OP("OpWithoutKernel").Input("a: T").Input("b: T").Attr("T: type");

class TestOp5Cpu : public tensorflow::OpKernel {
 public:
  explicit TestOp5Cpu(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

REGISTER_KERNEL_BUILDER(Name("Test5").Device(DEVICE_CPU).Priority(2),
                        TestOp5Cpu);

class TestOp5Gpu : public tensorflow::OpKernel {
 public:
  explicit TestOp5Gpu(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

REGISTER_KERNEL_BUILDER(Name("Test5").Device(DEVICE_GPU).Priority(1),
                        TestOp5Gpu);

NodeDef StripNodeDef(OpKernelConstruction* ctx) {
  const NodeDef& original = ctx->def();
  NodeDef ret;
  ret.set_name(original.name());
  ret.set_op(original.op());
  return ret;
}

class StrippedNode : public tensorflow::OpKernel {
 public:
  explicit StrippedNode(OpKernelConstruction* context)
      : OpKernel(context, StripNodeDef(context), false) {}
  void Compute(OpKernelContext* context) override {}
};

REGISTER_OP("StrippedNode").Input("i: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("StrippedNode").Device(DEVICE_CPU), StrippedNode);

class RefInputs : public tensorflow::OpKernel {
 public:
  explicit RefInputs(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ctx->delete_ref_input(0, false);
    Tensor t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({}), &t));
    OP_REQUIRES_OK(ctx, ctx->replace_ref_input("b", t, false));
  }
};

REGISTER_OP("RefInputs")
    .Input("a: Ref(float)")
    .Input("b: Ref(float)")
    .Output("c: float");
REGISTER_KERNEL_BUILDER(Name("RefInputs").Device(DEVICE_CPU), RefInputs);

static std::vector<DeviceType> DeviceTypes() {
  return {DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)};
}

class OpKernelTest : public ::testing::Test {
 public:
  OpKernelTest() : device_(Env::Default()) {}

 protected:
  NodeDef CreateNodeDef(const string& op_type, const DataTypeVector& inputs,
                        const string& device = "") {
    NodeDefBuilder builder(op_type + "-op", op_type);
    for (DataType dt : inputs) {
      builder.Input(FakeInput(dt));
    }
    builder.Device(device);
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
    absl::Status status;
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
    absl::Status status;
    std::unique_ptr<OpKernel> op(
        CreateOpKernel(std::move(device_type), &device_, cpu_allocator(),
                       node_def, TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.message();
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

TEST_F(OpKernelTest, SuccessStrippedNode) {
  ExpectSuccess("StrippedNode", DEVICE_CPU, {DT_FLOAT}, {DT_FLOAT});
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
  PrioritizedDeviceTypeVector devs;
  TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
  EXPECT_EQ(1, devs.size());
  EXPECT_EQ(DeviceType(DEVICE_CPU), devs[0].first);
}

TEST_F(OpKernelTest, KernelNotRegistered) {
  const string& local_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  const string& remote_device = "/job:worker/replica:0/task:0/device";
  {
    // Try a node def of an op which does not have kernel. And the requested
    // device in NodeDef is on a different address space than the local device.
    NodeDef ndef =
        CreateNodeDef("OpWithoutKernel", {DT_STRING, DT_STRING}, remote_device);
    PrioritizedDeviceTypeVector devs;
    DeviceNameUtils::ParsedName local_device_name;
    DeviceNameUtils::ParseFullName(local_device, &local_device_name);
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs,
                                             &local_device_name));
    EXPECT_EQ(2, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[0].first);
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[1].first);
  }

  {
    // Try a node def of an op which does not have kernel. And the requested
    // device in NodeDef is on the same address space as the local device.
    NodeDef ndef =
        CreateNodeDef("OpWithoutKernel", {DT_STRING, DT_STRING}, local_device);
    PrioritizedDeviceTypeVector devs;
    DeviceNameUtils::ParsedName local_device_name;
    DeviceNameUtils::ParseFullName(local_device, &local_device_name);
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs,
                                             &local_device_name));
    EXPECT_EQ(0, devs.size());
  }
}

TEST_F(OpKernelTest, CpuAndGpuTypeRegistered) {
  {
    // Try a node def of an op that is registered for a specific type
    // only on CPU.
    NodeDef ndef = CreateNodeDef("Test3", {DT_INT8, DT_INT8});
    PrioritizedDeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(1, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[0].first);
  }
  {
    // Try a node def of an op that is registered for a specific type
    // only on GPU.
    NodeDef ndef = CreateNodeDef("Test3", {DT_FLOAT, DT_FLOAT});
    PrioritizedDeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(1, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[0].first);
  }
  {
    // Try a node def of an op that is only registered for other types.
    NodeDef ndef = CreateNodeDef("Test3", {DT_STRING, DT_STRING});
    PrioritizedDeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(0, devs.size());
  }

  {
    // Try a node def of an op that is registered for both.
    NodeDef ndef = CreateNodeDef("Test4", {DT_FLOAT});
    PrioritizedDeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(2, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[0].first);
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[1].first);
  }

  {
    // Try a node def of an op where kernels have priorities.
    NodeDef ndef = CreateNodeDef("Test5", {DT_STRING, DT_STRING});
    PrioritizedDeviceTypeVector devs;
    TF_ASSERT_OK(SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs));
    EXPECT_EQ(2, devs.size());
    EXPECT_EQ(DeviceType(DEVICE_CPU), devs[0].first);
    EXPECT_EQ(2, devs[0].second);
    EXPECT_EQ(DeviceType(DEVICE_GPU), devs[1].first);
    EXPECT_EQ(1, devs[1].second);
  }
}

TEST_F(OpKernelTest, NotFound) {
  const auto not_found = error::NOT_FOUND;
  // Something with that op type name exists, but only with a
  // different DeviceType.
  ExpectFailure(tsl::LegacyUnredactedDebugString(
                    CreateNodeDef("Test1", {DT_FLOAT, DT_INT32})),
                DEVICE_GPU, not_found);
  ExpectFailure(tsl::LegacyUnredactedDebugString(
                    CreateNodeDef("Test3", {DT_INT8, DT_INT8})),
                DEVICE_GPU, not_found);
  ExpectFailure(tsl::LegacyUnredactedDebugString(
                    CreateNodeDef("Test3", {DT_FLOAT, DT_FLOAT})),
                DEVICE_CPU, not_found);

  // No kernel with that signature registered.
  ExpectFailure(tsl::LegacyUnredactedDebugString(
                    CreateNodeDef("Test3", {DT_INT32, DT_INT32})),
                DEVICE_GPU, not_found);

  // Nothing with that op type name exists.
  ExpectFailure("name: 'NF' op: 'Testnotfound'", DEVICE_CPU, not_found);
  ExpectFailure("name: 'NF' op: 'Testnotfound'", DEVICE_GPU, not_found);
}

TEST_F(OpKernelTest, TooFewInputs) {
  const auto invalid = error::INVALID_ARGUMENT;
  NodeDef node_def = CreateNodeDef("Test1", {DT_FLOAT, DT_INT32});
  node_def.clear_input();
  ExpectFailure(tsl::LegacyUnredactedDebugString(node_def), DEVICE_CPU,
                invalid);
  node_def.add_input("a");
  ExpectFailure(tsl::LegacyUnredactedDebugString(node_def), DEVICE_CPU,
                invalid);
}

TEST_F(OpKernelTest, TooManyInputs) {
  const auto invalid = error::INVALID_ARGUMENT;
  NodeDef node_def = CreateNodeDef("Test1", {DT_FLOAT, DT_INT32});
  node_def.add_input("c");
  ExpectFailure(tsl::LegacyUnredactedDebugString(node_def), DEVICE_CPU,
                invalid);
}

TEST_F(OpKernelTest, MatchSignatureFails) {
  const auto invalid = error::INVALID_ARGUMENT;
  foo::match_signature_ = true;
  ExpectFailure(
      tsl::LegacyUnredactedDebugString(CreateNodeDef("Test2", {DT_FLOAT})),
      DEVICE_GPU, invalid);
  EXPECT_FALSE(foo::match_signature_);
}

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }
};

TEST_F(OpKernelTest, InputDtype) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  DummyDevice device(env);
  params.device = &device;
  absl::Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({}));
  Tensor b(DT_INT32, TensorShape({}));
  Tensor c(DT_UINT8, TensorShape({}));
  absl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b),
                                             TensorValue(&c)};
  params.inputs = inputs;
  auto ctx = std::make_unique<OpKernelContext>(&params);

  DataType dtype;
  EXPECT_FALSE(ctx->input_dtype("non_existent_input", &dtype).ok());
  ASSERT_TRUE(ctx->input_dtype("a", &dtype).ok());
  EXPECT_EQ(dtype, DT_FLOAT);
  ASSERT_TRUE(ctx->input_dtype("b", &dtype).ok());
  EXPECT_EQ(dtype, DT_INT32);
}

TEST_F(OpKernelTest, InputOnly) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  DummyDevice device(env);
  params.device = &device;
  absl::Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({}));
  absl::InlinedVector<TensorValue, 2> inputs{TensorValue(&a)};
  params.inputs = inputs;
  auto ctx = std::make_unique<OpKernelContext>(&params);

  ASSERT_TRUE(ctx->get_input(0).ok());
  EXPECT_FALSE(ctx->get_input(1).ok());
  EXPECT_EQ(ctx->get_input(1).status(),
            absl::InvalidArgumentError(
                "Given index was 1, but index of input must be greater than 0, "
                "less than the number of inputs (1), and not a ref."));
}

TEST_F(OpKernelTest, RefInputs) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  DummyDevice device(env);
  params.device = &device;
  absl::Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("RefInputs", {DT_FLOAT_REF, DT_FLOAT_REF}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  Tensor* a = new Tensor(DT_FLOAT, TensorShape({}));
  Tensor* b = new Tensor(DT_FLOAT, TensorShape({2}));
  mutex mu_a, mu_b;
  absl::InlinedVector<TensorValue, 4> inputs{TensorValue(&mu_a, a),
                                             TensorValue(&mu_b, b)};
  params.inputs = inputs;
  auto ctx = std::make_unique<OpKernelContext>(&params);

  op->Compute(ctx.get());
  // We mutate the second input to have a {} Shape in the kernel. Asserting that
  // here.
  EXPECT_NE(ctx->mutable_input(1, false).shape(), TensorShape({2}));
  // a doesn't need deletion since the kernel deletes it.
  delete b;
}

TEST_F(OpKernelTest, AllocateOutput) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  DummyDevice device(env);
  params.device = &device;
  absl::Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                     CreateNodeDef("Test1", {DT_FLOAT, DT_INT32}),
                     TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({}));
  Tensor b(DT_INT32, TensorShape({}));
  absl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b)};
  params.inputs = inputs;
  auto ctx = std::make_unique<OpKernelContext>(&params);
  Tensor* output = nullptr;

  // Allocating to index -1 should fail (Only 0 should work).
  absl::Status s = ctx->allocate_output(-1, TensorShape({}), &output);
  EXPECT_THAT(s, tensorflow::testing::StatusIs(error::INTERNAL));
  EXPECT_THAT(s.message(), ::testing::ContainsRegex("bad index=-1"));

  // Allocating to index 1 should fail (Only 0 should work).
  s = ctx->allocate_output(1, TensorShape({}), &output);
  EXPECT_THAT(s, tensorflow::testing::StatusIs(error::INTERNAL));
  EXPECT_THAT(s.message(), ::testing::ContainsRegex("bad index=1"));

  // Testing allocate_output when allocator attributes are set.
  AllocatorAttributes attrs;
  attrs.set_on_host(true);
  TF_ASSERT_OK(ctx->allocate_output("o", TensorShape({}), &output, attrs));

  // Index -1 should fail as only 1 output for the op.
  s = ctx->allocate_output(-1, TensorShape({}), &output, attrs);
  EXPECT_THAT(s, tensorflow::testing::StatusIs(error::INTERNAL));
  EXPECT_THAT(s.message(), ::testing::ContainsRegex("bad index=-1"));

  // Index 1 should fail as only 1 output for the op.
  s = ctx->allocate_output(1, TensorShape({}), &output, attrs);
  EXPECT_THAT(s, tensorflow::testing::StatusIs(error::INTERNAL));
  EXPECT_THAT(s.message(), ::testing::ContainsRegex("bad index=1"));
}

// A mock device that mimics the behavior of scoped allocator upon calling
// GetAllocator with a positive scope_id.
class ScopedAllocatorDevice : public DeviceBase {
 public:
  explicit ScopedAllocatorDevice(Env* env)
      : DeviceBase(env),
        scope_allocated_(false),
        num_allocations_(0),
        num_scoped_allocations_(0) {}

  Allocator* GetAllocator(AllocatorAttributes attrs) override {
    CHECK_LE(attrs.scope_id, 0);
    num_allocations_++;
    return cpu_allocator();
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attrs,
                                int64_t /*step_id*/) override {
    CHECK_GT(attrs.scope_id, 0);
    num_scoped_allocations_++;
    if (scope_allocated_) {
      return nullptr;
    } else {
      scope_allocated_ = true;
      return cpu_allocator();
    }
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
    CHECK(input_tensor->NumElements() == output_tensor->NumElements());
    tensor::DeepCopy(*input_tensor, output_tensor);
    done(absl::OkStatus());
  }

  // Return the count of calls to GetAllocator or GetScopedAllocator, depending
  // on when scoped is false or true respectively.  For testing purposes.
  int num_allocations(bool scoped) {
    if (scoped) {
      return num_scoped_allocations_;
    } else {
      return num_allocations_;
    }
  }

 private:
  bool scope_allocated_;
  int num_allocations_;
  int num_scoped_allocations_;
};

// Test that a kernel which has an output marked for allocation via
// ScopedAllocator, which calls allocate_temp and set_output, does the right
// thing.  In this case, the expected behavior is for allocate_temp to return
// a temporary buffer, and set_output to copy the contents of this temp buffer
// into the ScopedAllocator slice.
TEST_F(OpKernelTest, ScopedAllocationTest) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  auto sa_device = std::make_unique<ScopedAllocatorDevice>(env);
  params.device = sa_device.get();
  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(
      DEVICE_CPU, params.device, cpu_allocator(),
      CreateNodeDef("Test4", {DT_FLOAT}), TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok());
  params.op_kernel = op.get();
  AllocatorAttributes alloc_attrs;
  alloc_attrs.scope_id = 1;
  std::vector<AllocatorAttributes> output_alloc_attrs({alloc_attrs});
  params.output_attr_array = output_alloc_attrs.data();
  std::vector<int> forward_from({OpKernelContext::Params::kNeverForward});
  params.forward_from_array = forward_from.data();
  auto ctx = std::make_unique<OpKernelContext>(&params);

  EXPECT_EQ(sa_device->num_allocations(false), 0);
  EXPECT_EQ(sa_device->num_allocations(true), 0);
  Tensor temp1;
  TF_EXPECT_OK(
      ctx->allocate_temp(DT_FLOAT, TensorShape({8}), &temp1, alloc_attrs));
  EXPECT_EQ(sa_device->num_allocations(false), 1);
  EXPECT_EQ(sa_device->num_allocations(true), 0);
  Tensor temp2;
  alloc_attrs.scope_id = -1;
  TF_EXPECT_OK(
      ctx->allocate_temp(DT_FLOAT, TensorShape({4}), &temp2, alloc_attrs));
  EXPECT_EQ(sa_device->num_allocations(false), 2);
  EXPECT_EQ(sa_device->num_allocations(true), 0);
  ctx->set_output(0, temp1);
  EXPECT_EQ(sa_device->num_allocations(false), 2);
  EXPECT_EQ(sa_device->num_allocations(true), 1);
}

TEST_F(OpKernelTest, TraceString) {
  Env* env = Env::Default();
  OpKernelContext::Params params;
  DummyDevice device(env);
  params.device = &device;

  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(
      DEVICE_CPU, params.device, cpu_allocator(),
      CreateNodeDef("Test4", {DT_FLOAT}), TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);

  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({4, 8}));
  absl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  params.inputs = inputs;

  params.op_kernel = op.get();
  OpKernelContext ctx(&params);

  EXPECT_THAT(op->TraceString(ctx, true),
              ::testing::ContainsRegex("Test4-op:Test4#shape"));
}

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
  PrioritizedDeviceTypeVector devs;
  absl::Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Multiple OpKernel registrations match NodeDef"));

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
  PrioritizedDeviceTypeVector devs;
  absl::Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Multiple OpKernel registrations match NodeDef"));

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
  PrioritizedDeviceTypeVector devs;
  absl::Status status = SupportedDeviceTypesForNode(DeviceTypes(), ndef, &devs);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.message(),
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
  DummyDevice device(env);
  params.device = &device;
  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(
      DEVICE_CPU, params.device, cpu_allocator(),
      CreateNodeDef("ListOut", {"T|list(type)|[DT_FLOAT, DT_INT32]"}),
      TF_GRAPH_DEF_VERSION, &status));
  EXPECT_TRUE(status.ok()) << status.ToString();
  params.op_kernel = op.get();
  auto ctx = std::make_unique<OpKernelContext>(&params);

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
  int64_t i;
  std::vector<int64_t> i_list;
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
  std::vector<std::pair<string, absl::Status>> status;
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
  EXPECT_EQ(std::vector<int64_t>({-1, 2, -4}), get_attr_kernel->i_list);
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
                key_status.second.message());
    }
  }

  op_kernel = ExpectSuccess("GetAttrInt", DEVICE_CPU,
                            {"attr_name|string|'b'", "a|int|8589934592",
                             "b|list(int)|[-8589934592]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"i_list"});  // no i32_list
  EXPECT_EQ(std::vector<int64_t>({-8589934592ll}), get_attr_kernel->i_list);
  for (const auto& key_status : get_attr_kernel->status) {
    if (key_status.first == "i32_list") {
      EXPECT_EQ(error::INVALID_ARGUMENT, key_status.second.code());
      EXPECT_EQ("Attr b has value -8589934592 out of range for an int32",
                key_status.second.message());
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
  TensorShapeProto expected_shape_proto;
  protobuf::TextFormat::ParseFromString("dim { size: 3 }",
                                        &expected_shape_proto);
  EXPECT_EQ(get_attr_kernel->shape_proto.ShortDebugString(),
            expected_shape_proto.ShortDebugString());
  EXPECT_EQ("[3]", get_attr_kernel->shape.DebugString());

  op_kernel = ExpectSuccess(
      "GetAttrShape", DEVICE_CPU,
      {"attr_name|string|'b'", "a|shape|{ dim { size: 3 } }",
       "b|list(shape)|[{ dim { size:2 } }, { dim { size: 4 } }]"});
  get_attr_kernel = static_cast<GetAttrKernel*>(op_kernel.get());
  get_attr_kernel->ExpectOk({"shape_list", "shape_proto_list"});
  ASSERT_EQ(2, get_attr_kernel->shape_proto_list.size());
  protobuf::TextFormat::ParseFromString("dim { size: 2 }",
                                        &expected_shape_proto);
  EXPECT_EQ(get_attr_kernel->shape_proto_list[0].ShortDebugString(),
            expected_shape_proto.ShortDebugString());
  protobuf::TextFormat::ParseFromString("dim { size: 4 }",
                                        &expected_shape_proto);
  EXPECT_EQ(get_attr_kernel->shape_proto_list[1].ShortDebugString(),
            expected_shape_proto.ShortDebugString());
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

REGISTER_OP("JitKernel");
REGISTER_KERNEL_BUILDER(
    Name("JitKernel").Device(DEVICE_CPU).Label(kJitKernelLabel),
    LabeledKernel<4>);

TEST_F(LabelTest, Filter) {
  ExpectSuccess("JitKernel", DEVICE_CPU, {absl::StrCat("_kernel|string|''")});
  ExpectFailure("JitKernel", DEVICE_CPU,
                {absl::StrCat("_kernel|string|'", kJitKernelLabel, "'")},
                error::NOT_FOUND);
}

void BM_InputRangeHelper(::testing::benchmark::State& state,
                         const NodeDef& node_def, const char* input_name,
                         int expected_start, int expected_stop) {
  absl::Status status;
  auto device = std::make_unique<DummyDevice>(Env::Default());

  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);

  for (auto s : state) {
    int start;
    int stop;
    TF_CHECK_OK(op->InputRange(input_name, &start, &stop));
    EXPECT_EQ(expected_start, start);
    EXPECT_EQ(expected_stop, stop);
  }
}

REGISTER_KERNEL_BUILDER(Name("ConcatV2").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("Select").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("MatMul").Device(DEVICE_CPU), DummyKernel);

void BM_ConcatInputRange(::testing::benchmark::State& state) {
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

  BM_InputRangeHelper(state, node_def, "values", 0, 4);
}

void BM_SelectInputRange(::testing::benchmark::State& state) {
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

  BM_InputRangeHelper(state, node_def, "condition", 0, 1);
}

void BM_TraceString(::testing::benchmark::State& state) {
  const int verbose = state.range(0);

  // Create a MatMul NodeDef with 2 inputs.
  NodeDef node_def;
  node_def.set_name("gradient_tape/model_1/dense_1/MatMul_1");
  node_def.set_op("MatMul");
  AttrValue transpose_a, transpose_b, attr_t;
  attr_t.set_type(DT_FLOAT);
  node_def.mutable_attr()->insert({"T", attr_t});
  transpose_a.set_b(true);
  node_def.mutable_attr()->insert({"transpose_a", transpose_a});
  transpose_b.set_b(true);
  node_def.mutable_attr()->insert({"transpose_b", transpose_b});
  for (size_t i = 0; i < 2; ++i) {
    node_def.add_input(strings::StrCat("a:", i));
  }

  // Build OpKernel and OpKernelContext
  absl::Status status;
  auto device = std::make_unique<DummyDevice>(Env::Default());
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);

  OpKernelContext::Params params;
  params.device = device.get();
  params.op_kernel = op.get();
  Tensor a(DT_FLOAT, TensorShape({99000, 256}));
  Tensor b(DT_FLOAT, TensorShape({256, 256}));
  absl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b)};
  params.inputs = inputs;
  auto ctx = std::make_unique<OpKernelContext>(&params);

  for (auto s : state) {
    auto trace = op->TraceString(*ctx, verbose);
  }
}

BENCHMARK(BM_ConcatInputRange);
BENCHMARK(BM_SelectInputRange);
BENCHMARK(BM_TraceString)->Arg(1)->Arg(0);

TEST(RegisteredKernels, CanCallGetAllRegisteredKernels) {
  auto kernel_list = GetAllRegisteredKernels();
  auto all_registered_kernels = kernel_list.kernel();
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

TEST(RegisteredKernels, GetFilteredRegisteredKernels) {
  auto has_name_test1 = [](const KernelDef& k) { return k.op() == "Test1"; };
  auto kernel_list = GetFilteredRegisteredKernels(has_name_test1);
  ASSERT_EQ(kernel_list.kernel_size(), 1);
  EXPECT_EQ(kernel_list.kernel(0).op(), "Test1");
  EXPECT_EQ(kernel_list.kernel(0).device_type(), "CPU");
}

TEST(RegisteredKernels, GetRegisteredKernelsForOp) {
  auto kernel_list = GetRegisteredKernelsForOp("Test1");
  ASSERT_EQ(kernel_list.kernel_size(), 1);
  EXPECT_EQ(kernel_list.kernel(0).op(), "Test1");
  EXPECT_EQ(kernel_list.kernel(0).device_type(), "CPU");
}

// EXTRACT_KERNEL_NAME_TO_STRING wraps TF_EXTRACT_KERNEL_NAME for testing
// (it involves quite a bit of macro-magic).
#define EXTRACT_KERNEL_NAME_TO_STRING_IMPL(name, kernel_builder, ...) name
#define EXTRACT_KERNEL_NAME_TO_STRING(kernel_builder) \
  TF_EXTRACT_KERNEL_NAME(EXTRACT_KERNEL_NAME_TO_STRING_IMPL, kernel_builder)

TEST(RegisterKernelMacro, ExtractName) {
  static constexpr char const* kName = "Foo";
  static constexpr char const* kExtractedName =
      EXTRACT_KERNEL_NAME_TO_STRING(Name(kName).Label("Label"));
  EXPECT_THAT(kExtractedName, ::testing::StrEq(kName));
}

}  // namespace
}  // namespace tensorflow
