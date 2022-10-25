/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

namespace {

static int* GetCopyCPUToGPUCounter() {
  static int* counter = new int(0);
  return counter;
}

static int* GetCopyGPUToCPUCounter() {
  static int* counter = new int(0);
  return counter;
}

static int* GetCopyGPUToGPUCounter() {
  static int* counter = new int(0);
  return counter;
}

struct StoredTensorValue {
  Tensor stored;
  string TypeName() const { return "StoredTensorValue"; }
  void Encode(VariantTensorData* data) const { data->tensors_ = {stored}; }
  bool Decode(const VariantTensorData& data) {
    CHECK_EQ(1, data.tensors_.size());
    stored = data.tensors_[0];
    return true;
  }
  static Status CopyCPUToGPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
    ++*GetCopyCPUToGPUCounter();
    return copy(from.stored, &(to->stored));
  }
  static Status CopyGPUToCPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
    ++*GetCopyGPUToCPUCounter();
    return copy(from.stored, &(to->stored));
  }
  static Status CopyGPUToGPU(
      const StoredTensorValue& from, StoredTensorValue* to,
      const std::function<Status(const Tensor&, Tensor*)>& copy) {
    ++*GetCopyGPUToGPUCounter();
    return copy(from.stored, &(to->stored));
  }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(StoredTensorValue, "StoredTensorValue");

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::HOST_TO_DEVICE,
    StoredTensorValue::CopyCPUToGPU);

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::DEVICE_TO_HOST,
    StoredTensorValue::CopyGPUToCPU);

INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(
    StoredTensorValue, VariantDeviceCopyDirection::DEVICE_TO_DEVICE,
    StoredTensorValue::CopyGPUToGPU);

REGISTER_OP("CreateTestVariant")
    .Input("input: T")
    .Attr("T: type")
    .Output("output: variant")
    .SetShapeFn(shape_inference::UnknownShape);

class CreateTestVariantOp : public OpKernel {
 public:
  explicit CreateTestVariantOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    // Take the scalar tensor fed as input, and emit a Tensor
    // containing 10 Variants (StoredTensorValues), both containing
    // the input tensor.
    const Tensor& stored_t = c->input(0);
    Tensor* out;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({10}), &out));
    StoredTensorValue store{stored_t};
    auto t = out->flat<Variant>();
    for (int i = 0; i < 10; ++i) {
      t(i) = store;
    }
    CHECK_EQ("StoredTensorValue", t(0).TypeName());
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateTestVariant").Device(DEVICE_CPU),
                        CreateTestVariantOp);

class CreateTestVariant {
 public:
  explicit CreateTestVariant(const ::tensorflow::Scope& scope,
                             const Input& value) {
    if (!scope.ok()) return;
    auto _value = ops::AsNodeOut(scope, value);
    if (!scope.ok()) return;
    ::tensorflow::Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("CreateTestVariant");
    auto builder = ::tensorflow::NodeBuilder(unique_name, "CreateTestVariant")
                       .Input(_value);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    if (!scope.ok()) return;
    this->output_ = Output(ret, 0);
  }

  // Intentionally not marked as explicit.
  // NOLINTNEXTLINE google-explicit-constructor
  operator ::tensorflow::Output() const { return output_; }
  // Intentionally not marked as explicit.
  // NOLINTNEXTLINE google-explicit-constructor
  operator ::tensorflow::Input() const { return output_; }

  ::tensorflow::Node* node() const { return output_.node(); }

  ::tensorflow::Output output_;
};

}  // end namespace

TEST(VariantOpCopyTest, CreateConstOnCPU) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_INT64, TensorShape({}));
  from.stored.scalar<int64_t>()() = 0xdeadbeef;
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_const}, &outputs));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_VARIANT, outputs[0].dtype());
  EXPECT_EQ(0, outputs[0].dims());
  const Variant& variant = outputs[0].scalar<Variant>()();
  EXPECT_EQ("StoredTensorValue", variant.TypeName());
  const StoredTensorValue* to = variant.get<StoredTensorValue>();
  EXPECT_EQ(to->stored.dtype(), DT_INT64);
  EXPECT_EQ(0xdeadbeef, to->stored.scalar<int64_t>()());
}

TEST(VariantOpCopyTest, CreateConstOnGPU) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/gpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_INT64, TensorShape({}));
  from.stored.scalar<int64_t>()() = 0xdeadbeef;
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;

  int copy_to_gpu_before = *GetCopyCPUToGPUCounter();
  int copy_to_cpu_before = *GetCopyGPUToCPUCounter();
  TF_CHECK_OK(session.Run({create_const}, &outputs));
  int copy_to_cpu_after = *GetCopyGPUToCPUCounter();
  int copy_to_gpu_after = *GetCopyCPUToGPUCounter();

  EXPECT_GT(copy_to_cpu_after - copy_to_cpu_before, 0);
  EXPECT_GT(copy_to_gpu_after - copy_to_gpu_before, 0);

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_VARIANT, outputs[0].dtype());
  EXPECT_EQ(0, outputs[0].dims());
  const Variant& variant = outputs[0].scalar<Variant>()();
  EXPECT_EQ("StoredTensorValue", variant.TypeName());
  const StoredTensorValue* to = variant.get<StoredTensorValue>();
  EXPECT_EQ(to->stored.dtype(), DT_INT64);
  EXPECT_EQ(0xdeadbeef, to->stored.scalar<int64_t>()());
}

TEST(VariantOpCopyTest, CreateConstOnGPUFailsGracefully) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/gpu:0");

  // Create the input StoredTensorValue and serialize it.
  StoredTensorValue from;
  from.stored = Tensor(DT_STRING, TensorShape({}));
  from.stored.scalar<tstring>()() = "hi";
  VariantTensorData data;
  data.set_type_name(from.TypeName());
  from.Encode(&data);

  TensorProto variant_proto;
  variant_proto.set_dtype(DT_VARIANT);
  TensorShape scalar_shape({});
  scalar_shape.AsProto(variant_proto.mutable_tensor_shape());
  data.ToProto(variant_proto.add_variant_val());

  Output create_const = ops::ConstFromProto(root, variant_proto);
  TF_ASSERT_OK(root.status());
  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status s = session.Run({create_const}, &outputs);
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "GPU copy from non-DMA string tensor"))
      << s.ToString();
}

TEST(VariantOpCopyTest, CreateCopyCPUToCPU) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Tensor t_42(DT_INT32, TensorShape({}));
  t_42.flat<int32>()(0) = 42;
  Output create_op = CreateTestVariant(root, t_42);
  Output identity = ops::Identity(root, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ(42, v1->stored.scalar<int32>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToCPUString) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Tensor t_str(DT_STRING, TensorShape({}));
  t_str.scalar<tstring>()() = "hi";
  Output create_op = CreateTestVariant(root, t_str);
  Output identity = ops::Identity(root, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ("hi", v1->stored.scalar<tstring>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToGPU) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Scope with_gpu = root.WithDevice("/gpu:0");
  Tensor t_42(DT_INT32, TensorShape({}));
  t_42.scalar<int32>()() = 42;
  Output create_op = CreateTestVariant(root, t_42);
  Output identity = ops::Identity(with_gpu, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  int copy_to_gpu_before = *GetCopyCPUToGPUCounter();
  int copy_to_cpu_before = *GetCopyGPUToCPUCounter();
  // Force the identity to run on GPU, and then the data to be copied
  // back to CPU for the final output.
  TF_CHECK_OK(session.Run({create_op, identity}, &outputs));
  int copy_to_cpu_after = *GetCopyGPUToCPUCounter();
  int copy_to_gpu_after = *GetCopyCPUToGPUCounter();

  EXPECT_GT(copy_to_cpu_after - copy_to_cpu_before, 0);
  EXPECT_GT(copy_to_gpu_after - copy_to_gpu_before, 0);

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(10, outputs[1].dim_size(0));
  auto output = outputs[1].flat<Variant>();
  for (int i = 0; i < 10; ++i) {
    const Variant& r1 = output(i);
    EXPECT_EQ("StoredTensorValue", r1.TypeName());
    const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
    EXPECT_NE(v1, nullptr);
    EXPECT_EQ(42, v1->stored.scalar<int32>()());
  }
}

TEST(VariantOpCopyTest, CreateCopyCPUToGPUStringFailsSafely) {
  if (!IsGoogleCudaEnabled()) return;

  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Scope with_gpu = root.WithDevice("/gpu:0");
  Tensor t_str(DT_STRING, TensorShape({}));
  t_str.scalar<tstring>()() = "hi";
  Output create_op = CreateTestVariant(root, t_str);
  Output identity = ops::Identity(with_gpu, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  Status err = session.Run({create_op, identity}, &outputs);
  EXPECT_TRUE(errors::IsInvalidArgument(err));
  EXPECT_TRUE(
      absl::StrContains(err.error_message(),
                        "During Variant Host->Device Copy: non-DMA-copy "
                        "attempted of tensor type: string"))
      << err.error_message();
}

// TODO(ebrevdo): Identify a way to create two virtual GPUs within a
// single session, so that we can test the Device <-> Device copy
// branch.

}  // end namespace tensorflow
