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

#ifdef INTEL_MKL

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

class EagerOpRewriteTest : public ::testing::Test {
 public:
  EagerOpRewriteTest() : eager_ctx_(nullptr) {}
  ~EagerOpRewriteTest() {
    if (eager_ctx_) {
      eager_ctx_->Unref();
    }
  }

  // Creates a new op to be used as input to MKL eager rewrite.
  std::unique_ptr<tensorflow::EagerOperation> CreateOp(const string op_name) {
    std::unique_ptr<DeviceMgr> device_mgr =
        absl::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
            "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
    bool async = false;
    tensorflow::Rendezvous* rendezvous =
        new tensorflow::IntraProcessRendezvous(device_mgr.get());
    eager_ctx_ = new tensorflow::EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        async, device_mgr.get(), false, rendezvous);

    EagerExecutor executor_(false);
    std::unique_ptr<tensorflow::EagerOperation> op(
        new tensorflow::EagerOperation(eager_ctx_));
    EXPECT_EQ(Status::OK(),
              op.get()->Reset(op_name.c_str(), nullptr, false, &executor_));
    return op;
  }

  // Validates the result of MKL eager rewrite.
  void CheckRewrite(EagerOperation* orig_op, string expected_op_name) {
    std::unique_ptr<tensorflow::EagerOperation> out_op;
    EXPECT_EQ(Status::OK(),
              EagerOpRewriteRegistry::Global()->RunRewrite(
                  EagerOpRewriteRegistry::PRE_EXECUTION, orig_op, &out_op));

    // actual_op_name is same as original op name if rewrite didn't happen.
    string actual_op_name = orig_op->Name();
    if (out_op) {
      actual_op_name = out_op->Name();
    }

    EXPECT_EQ(actual_op_name, expected_op_name);
  }

 protected:
  tensorflow::EagerContext* eager_ctx_;
};

#define CONV_FORWARD_OPS "Conv2D", "Conv3D", "DepthwiseConv2dNative"

#define CONV_BACKWARD_OPS                                                  \
  "Conv2DBackpropInput", "Conv2DBackpropFilter", "Conv3DBackpropFilterV2", \
      "Conv3DBackpropInputV2", "DepthwiseConv2dNativeBackpropFilter",      \
      "DepthwiseConv2dNativeBackpropInput"

#define CONV_OPS CONV_FORWARD_OPS, CONV_BACKWARD_OPS

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> conv_ops = {CONV_OPS};                        \
    for (int i = 0; i < conv_ops.size(); ++i) {                       \
      auto orig_op = CreateOp(conv_ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("padding", "VALID");               \
      CheckRewrite(orig_op.get(),                                     \
                   mkl_op_registry::GetMklNativeOpName(conv_ops[i])); \
    }                                                                 \
  }
REGISTER_TEST_ALL_TYPES(ConvOps_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> conv_ops = {CONV_FORWARD_OPS};                \
    for (int i = 0; i < conv_ops.size(); ++i) {                       \
      auto orig_op = CreateOp(conv_ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("padding", "EXPLICIT");            \
      CheckRewrite(orig_op.get(),                                     \
                   mkl_op_registry::GetMklNativeOpName(conv_ops[i])); \
    }                                                                 \
  }
REGISTER_TEST_ALL_TYPES(ConvOpsExplicitPadding_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                      \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                 \
    std::vector<string> conv_ops = {CONV_BACKWARD_OPS};    \
    for (int i = 0; i < conv_ops.size(); ++i) {            \
      auto orig_op = CreateOp(conv_ops[i]);                \
      orig_op->MutableAttrs()->Set("T", T);                \
      orig_op->MutableAttrs()->Set("padding", "EXPLICIT"); \
      CheckRewrite(orig_op.get(), conv_ops[i]);            \
    }                                                      \
  }
REGISTER_TEST_ALL_TYPES(ConvOpsExplicitPadding_Negative);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                            \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                       \
    std::vector<string> ops = {"AvgPool",                        \
                               "AvgPoolGrad",                    \
                               "AvgPool3D",                      \
                               "AvgPool3DGrad",                  \
                               "BatchMatMul",                    \
                               "FusedBatchNorm",                 \
                               "FusedBatchNormV2",               \
                               "FusedBatchNormV3",               \
                               "FusedBatchNormGrad",             \
                               "FusedBatchNormGradV2",           \
                               "FusedBatchNormGradV3",           \
                               "MatMul"};                        \
    for (int i = 0; i < ops.size(); ++i) {                       \
      auto orig_op = CreateOp(ops[i]);                           \
      orig_op->MutableAttrs()->Set("T", T);                      \
      CheckRewrite(orig_op.get(),                                \
                   mkl_op_registry::GetMklNativeOpName(ops[i])); \
    }                                                            \
  }
REGISTER_TEST_ALL_TYPES(MostOps_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                 \
  TEST_F(EagerOpRewriteTest, NAME##_##T) {                            \
    std::vector<string> Fused_BN_ops = {"FusedBatchNormV3",           \
                                        "FusedBatchNormGradV3"};      \
    for (int i = 0; i < Fused_BN_ops.size(); ++i) {                   \
      auto orig_op = CreateOp(Fused_BN_ops[i]);                       \
      orig_op->MutableAttrs()->Set("T", T);                           \
      orig_op->MutableAttrs()->Set("data_format", "" DATA_FORMAT ""); \
      CheckRewrite(orig_op.get(), Fused_BN_ops[i]);                   \
    }                                                                 \
  }
#define DATA_FORMAT "NCDHW"
REGISTER_TEST_ALL_TYPES(FusedBatchNormV3_5D_Negative_1);
#undef DATA_FORMAT

#define DATA_FORMAT "NDHWC"
REGISTER_TEST_ALL_TYPES(FusedBatchNormV3_5D_Negative_2);
#undef DATA_FORMAT

#undef REGISTER_TEST

}  // namespace tensorflow

#endif  // INTEL_MKL
