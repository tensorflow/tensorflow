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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

struct StoredTensorValue {
  Tensor stored;
  string TypeName() const { return "StoredTensorValue"; }
  void Encode(VariantTensorData* data) const { data->tensors_ = {stored}; }
  bool Decode(const VariantTensorData& data) {
    CHECK_EQ(1, data.tensors_.size());
    stored = data.tensors_[0];
    return true;
  }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(StoredTensorValue, "StoredTensorValue");

REGISTER_OP("CreateTestVariant")
    .Output("output: variant")
    .SetShapeFn(shape_inference::UnknownShape);

class CreateTestVariantOp : public OpKernel {
 public:
  explicit CreateTestVariantOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    Tensor* out;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &out));
    PersistentTensor stored_pt;
    Tensor* stored_t;
    OP_REQUIRES_OK(c, c->allocate_persistent(DT_INT32, TensorShape({}),
                                             &stored_pt, &stored_t));
    auto stored = stored_t->scalar<int32>();
    stored() = 42;
    StoredTensorValue store{*stored_t};
    auto t = out->flat<Variant>();
    t(0) = store;
    CHECK_EQ("StoredTensorValue", t(0).TypeName());
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateTestVariant").Device(DEVICE_CPU),
                        CreateTestVariantOp);

class CreateTestVariant {
 public:
  explicit CreateTestVariant(const ::tensorflow::Scope& scope) {
    if (!scope.ok()) return;
    ::tensorflow::Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("CreateTestVariant");
    auto builder = ::tensorflow::NodeBuilder(unique_name, "CreateTestVariant");
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

TEST(VariantOpCopyTest, CreateCopyCPUToCPU) {
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");
  Output create_op = CreateTestVariant(root);
  Output identity = ops::Identity(root, create_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({create_op, identity}, &outputs));
  EXPECT_EQ(2, outputs.size());
  const Variant& r1 = outputs[1].scalar<Variant>()();

  EXPECT_EQ("StoredTensorValue", r1.TypeName());
  const StoredTensorValue* v1 = r1.get<StoredTensorValue>();
  EXPECT_NE(v1, nullptr);
  EXPECT_EQ(42, v1->stored.scalar<int32>()());
}

}  // end namespace tensorflow
