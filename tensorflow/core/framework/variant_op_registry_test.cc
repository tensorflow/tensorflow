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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/variant_op_registry.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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
  static Status CPUZerosLikeFn(OpKernelContext* ctx, const VariantValue& v,
                               VariantValue* v_out) {
    if (v.early_exit) {
      return errors::InvalidArgument("early exit zeros_like!");
    }
    v_out->zeros_like_set = 1;  // CPU
    return Status::OK();
  }
  static Status GPUZerosLikeFn(OpKernelContext* ctx, const VariantValue& v,
                               VariantValue* v_out) {
    if (v.early_exit) {
      return errors::InvalidArgument("early exit zeros_like!");
    }
    v_out->zeros_like_set = 2;  // GPU
    return Status::OK();
  }
  bool early_exit;
  int zeros_like_set;
};

REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(VariantValue, "TEST VariantValue",
                                      VariantValue::ShapeFn);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VariantValue, "TEST VariantValue");

REGISTER_UNARY_VARIANT_ZEROS_LIKE_FUNCTION(DEVICE_CPU, VariantValue,
                                           "TEST VariantValue",
                                           VariantValue::CPUZerosLikeFn);

REGISTER_UNARY_VARIANT_ZEROS_LIKE_FUNCTION(DEVICE_GPU, VariantValue,
                                           "TEST VariantValue",
                                           VariantValue::GPUZerosLikeFn);

}  // namespace

TEST(VariantOpShapeRegistryTest, TestBasic) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetShapeFn("YOU SHALL NOT PASS"),
            nullptr);

  auto* shape_fn =
      UnaryVariantOpRegistry::Global()->GetShapeFn("TEST VariantValue");
  EXPECT_NE(shape_fn, nullptr);
  TensorShape shape;

  VariantValue vv_early_exit{true /* early_exit */};
  Variant v = vv_early_exit;
  Status s0 = (*shape_fn)(v, &shape);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(StringPiece(s0.error_message()).contains("early exit!"));

  VariantValue vv_ok{false /* early_exit */};
  v = vv_ok;
  TF_EXPECT_OK((*shape_fn)(v, &shape));
  EXPECT_EQ(shape, TensorShape({-0xdeadbeef}));
}

TEST(VariantOpShapeRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantShapeFn f;
  registry.RegisterShapeFn("fjfjfj", f);
  EXPECT_DEATH(registry.RegisterShapeFn("fjfjfj", f),
               "fjfjfj already registered");
}

TEST(VariantOpDecodeRegistryTest, TestBasic) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetDecodeFn("YOU SHALL NOT PASS"),
            nullptr);

  auto* decode_fn =
      UnaryVariantOpRegistry::Global()->GetDecodeFn("TEST VariantValue");
  EXPECT_NE(decode_fn, nullptr);

  VariantValue vv{true /* early_exit */};
  Variant v = vv;
  VariantTensorData data;
  v.Encode(&data);
  VariantTensorDataProto proto;
  data.ToProto(&proto);
  Variant encoded = proto;
  EXPECT_TRUE((*decode_fn)(&encoded));
  VariantValue* decoded = encoded.get<VariantValue>();
  EXPECT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->early_exit, true);
}

TEST(VariantOpDecodeRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantDecodeFn f;
  registry.RegisterDecodeFn("fjfjfj", f);
  EXPECT_DEATH(registry.RegisterDecodeFn("fjfjfj", f),
               "fjfjfj already registered");
}

TEST(VariantOpZerosLikeRegistryTest, TestBasicCPU) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetZerosLikeFn(
                DEVICE_CPU, "YOU SHALL NOT PASS"),
            nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 0 /* zeros_like_set */};
  Variant v = vv_early_exit;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 =
      CreateZerosLikeVariant<CPUDevice>(null_context_pointer, v, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(
      StringPiece(s0.error_message()).contains("early exit zeros_like"));

  VariantValue vv_ok{false /* early_exit */, 0 /* zeros_like_set */};
  v = vv_ok;
  TF_EXPECT_OK(
      CreateZerosLikeVariant<CPUDevice>(null_context_pointer, v, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->zeros_like_set, 1);  // CPU
}

#if GOOGLE_CUDA
TEST(VariantOpZerosLikeRegistryTest, TestBasicGPU) {
  EXPECT_EQ(UnaryVariantOpRegistry::Global()->GetZerosLikeFn(
                DEVICE_GPU, "YOU SHALL NOT PASS"),
            nullptr);

  VariantValue vv_early_exit{true /* early_exit */, 0 /* zeros_like_set */};
  Variant v = vv_early_exit;
  Variant v_out = VariantValue();

  OpKernelContext* null_context_pointer = nullptr;
  Status s0 =
      CreateZerosLikeVariant<GPUDevice>(null_context_pointer, v, &v_out);
  EXPECT_FALSE(s0.ok());
  EXPECT_TRUE(
      StringPiece(s0.error_message()).contains("early exit zeros_like"));

  VariantValue vv_ok{false /* early_exit */, 0 /* zeros_like_set */};
  v = vv_ok;
  TF_EXPECT_OK(
      CreateZerosLikeVariant<GPUDevice>(null_context_pointer, v, &v_out));
  VariantValue* vv_out = CHECK_NOTNULL(v_out.get<VariantValue>());
  EXPECT_EQ(vv_out->zeros_like_set, 2);  // GPU
}
#endif  // GOOGLE_CUDA

TEST(VariantOpZerosLikeRegistryTest, TestDuplicate) {
  UnaryVariantOpRegistry registry;
  UnaryVariantOpRegistry::VariantZerosLikeFn f;

  registry.RegisterZerosLikeFn(DEVICE_CPU, "fjfjfj", f);
  EXPECT_DEATH(registry.RegisterZerosLikeFn(DEVICE_CPU, "fjfjfj", f),
               "fjfjfj already registered");

  registry.RegisterZerosLikeFn(DEVICE_GPU, "fjfjfj", f);
  EXPECT_DEATH(registry.RegisterZerosLikeFn(DEVICE_GPU, "fjfjfj", f),
               "fjfjfj already registered");
}

}  // namespace tensorflow
