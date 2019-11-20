/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

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

void TestBitcastOp(Tensor* input_tensor, DataType out_type,
                   TensorShape expected_shape, error::Code expected_code) {
  Status status;
  NodeDef def;
  def.set_op("Bitcast");
  def.set_device(DEVICE_CPU);

  AttrValue typeAttr;
  SetAttrValue(input_tensor->dtype(), &typeAttr);

  AttrValue outTypeAttr;
  SetAttrValue(out_type, &outTypeAttr);

  (*def.mutable_attr())["T"] = typeAttr;
  (*def.mutable_attr())["type"] = outTypeAttr;

  def.add_input(
      strings::StrCat("input1: ", DataTypeString(input_tensor->dtype())));

  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_CPU), nullptr, nullptr, def, 1, &status);
  ASSERT_TRUE(status.ok()) << status.ToString();

  OpKernelContext::Params params;
  DummyDevice dummy_device(nullptr, false);
  params.device = &dummy_device;
  params.op_kernel = kernel.get();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.emplace_back(input_tensor);
  params.inputs = &inputs;

  OpKernelContext ctx(&params);
  kernel->Compute(&ctx);
  ASSERT_EQ(expected_code, ctx.status().code());
  if (expected_code == error::OK) {
    ASSERT_EQ(expected_shape, ctx.mutable_output(0)->shape())
        << ctx.mutable_output(0)->shape().DebugString();
  }
}

TEST(BitcastOpTest, TestUpcast) {
  Tensor int8_input(DT_UINT8, {8});
  for (int i = 0; i < 8; i++) {
    int8_input.vec<uint8>()(i) = static_cast<uint8>(1);
  }
  TestBitcastOp(&int8_input, DT_UINT64, TensorShape(), error::OK);
}

TEST(BitcastOpTest, TestDowncast) {
  Tensor int64_input(static_cast<uint64>(1));
  TestBitcastOp(&int64_input, DT_UINT8, TensorShape({8}), error::OK);
}

TEST(BitcastOpTest, TestCastToSameSize) {
  Tensor int32_input(DT_UINT32, {4, 6});
  TestBitcastOp(&int32_input, DT_UINT8, TensorShape({4, 6, 4}), error::OK);
}

TEST(BitcastOpTest, TestImpossibleCast) {
  Tensor int8_input(DT_UINT8, {1});
  TestBitcastOp(&int8_input, DT_UINT32, TensorShape(), error::INVALID_ARGUMENT);
}

PartialTensorShape S(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

TEST(BitcastOpTest, TestShapeInference_LargerShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT8)
                  .Attr("T", DT_INT64)
                  .Input(FakeInput(DT_INT64))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4,8]", c.DebugString(c.output(0)));
}

TEST(BitcastOpTest, TestShapeInference_SmallerShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT64)
                  .Attr("T", DT_INT8)
                  .Input(FakeInput(DT_INT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4, 8})}, {}, {},
                                      {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4,8]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

TEST(BitcastOpTest, TestShapeInference_SameShape) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Bitcast", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("type", DT_INT32)
                  .Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

}  // namespace
}  // namespace tensorflow
