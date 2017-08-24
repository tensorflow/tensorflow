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

#include <memory>

#include "tensorflow/core/framework/variant_op_registry.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

struct VariantValue {
  string TypeName() const { return "TEST VariantValue"; }
  static Status ShapeFn(const VariantValue& v, TensorShape* s) {
    if (v.early_exit) {
      return errors::InvalidArgument("early exit!");
    }
    *s = TensorShape({-0xdeadbeef});
    return Status::OK();
  }
  bool early_exit;
};

REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(VariantValue, "TEST VariantValue",
                                      VariantValue::ShapeFn);

}  // namespace

TEST(VariantOpRegistryTest, TestBasic) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetShapeFn("YOU SHALL NOT PASS"),
            nullptr);

  auto* shape_fn =
      UnaryVariantOpRegistry::Global()->GetShapeFn("TEST VariantValue");
  EXPECT_NE(shape_fn, nullptr);
  TensorShape shape;

  VariantValue vv_early_exit{true /* early_exit */};
  Variant v = vv_early_exit;
  Status s0 = (*shape_fn)(&v, &shape);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(StringPiece(s0.error_message()).contains("early exit!"));

  VariantValue vv_ok{false /* early_exit */};
  v = vv_ok;
  TF_EXPECT_OK((*shape_fn)(&v, &shape));
  EXPECT_EQ(shape, TensorShape({-0xdeadbeef}));
}

TEST(OpRegistrationTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantShapeFn f;
  registry.RegisterShapeFn("fjfjfj", f);
  EXPECT_DEATH(registry.RegisterShapeFn("fjfjfj", f),
               "fjfjfj already registered");
}

}  // namespace tensorflow
