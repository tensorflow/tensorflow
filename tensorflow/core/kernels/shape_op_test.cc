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

#include <functional>
#include <memory>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class ShapeOpTest : public OpsTestBase {};

struct NoKnownShape {
  string TypeName() const { return "NO KNOWN SHAPE"; }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(NoKnownShape, "NO KNOWN SHAPE");

struct KnownVecSize {
  KnownVecSize() : shape_value(0) {}
  explicit KnownVecSize(int value) : shape_value(value) {}
  string TypeName() const { return "KNOWN VECTOR SIZE TYPE"; }
  bool Decode(const VariantTensorData& d) {
    return d.get_metadata(&shape_value);
  }
  void Encode(VariantTensorData* d) const { d->set_metadata(shape_value); }
  int shape_value;
};

Status GetShapeFromKnownVecSize(const KnownVecSize& ks, TensorShape* s) {
  *s = TensorShape({ks.shape_value});
  return Status::OK();
}

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(KnownVecSize, "KNOWN VECTOR SIZE TYPE");

REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(KnownVecSize, GetShapeFromKnownVecSize);

static void ExpectHasError(const Status& s, StringPiece substr) {
  EXPECT_TRUE(str_util::StrContains(s.ToString(), substr))
      << ">>" << s << "<<, expected substring >>" << substr << "<<";
}

TEST_F(ShapeOpTest, Simple) {
  // Ensure the ops run on CPU, as we have no device copy registration
  // for NoKnownShape and KnownVecSize objects.
  Scope root = Scope::NewRootScope().WithDevice("/cpu:0");

  // Use a placeholder so the graph optimizer doesn't optimize away
  // the shape function.
  auto input = ops::Placeholder(root, DT_VARIANT);
  auto shape_output = ops::Shape(root, input);
  auto rank_output = ops::Rank(root, input);
  auto size_output = ops::Size(root, input);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);

  std::vector<Tensor> outputs;

  {
    // Test no shape registered.
    Tensor variant_tensor(DT_VARIANT, TensorShape({}));
    Variant& v = variant_tensor.scalar<Variant>()();
    v = NoKnownShape();
    Status s = session.Run({{input, variant_tensor}}, {shape_output}, &outputs);
    EXPECT_FALSE(s.ok());
    ExpectHasError(
        s, strings::StrCat(
               "No unary variant shape function found for Variant type_index: ",
               port::MaybeAbiDemangle(MakeTypeIndex<NoKnownShape>().name())));
  }

  {
    // Test non-scalar variant.
    Tensor variant_tensor(DT_VARIANT, TensorShape({1}));
    Status s = session.Run({{input, variant_tensor}}, {shape_output}, &outputs);
    EXPECT_FALSE(s.ok());
    ExpectHasError(s, "Shape of non-unary Variant not supported.");
  }

  {
    // Test registered variant.
    Tensor variant_tensor(DT_VARIANT, TensorShape({}));
    const int vec_dim_value = -0xdeadbeef;  // must be non-negative.
    Variant& v = variant_tensor.scalar<Variant>()();
    v = KnownVecSize(vec_dim_value);
    TF_EXPECT_OK(session.Run({{input, variant_tensor}},
                             {shape_output, rank_output, size_output},
                             &outputs));
    EXPECT_EQ(outputs[0].dims(), 1);  // shape
    EXPECT_EQ(vec_dim_value, outputs[0].vec<int32>()(0));
    EXPECT_EQ(outputs[1].dims(), 0);  // rank
    EXPECT_EQ(1, outputs[1].scalar<int32>()());
    EXPECT_EQ(outputs[2].dims(), 0);  // size
    EXPECT_EQ(vec_dim_value, outputs[0].scalar<int32>()());
  }
}

}  // namespace
}  // namespace tensorflow
