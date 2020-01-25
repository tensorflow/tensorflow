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

namespace tensorflow {

class EagerOpRewriteTest {
 public:
  EagerOpRewriteTest() {}

  // Creates a new op to be used as input to MKL eager rewrite.
  static std::unique_ptr<tensorflow::EagerOperation> CreateOp(
      const string op_name) {
    std::unique_ptr<DeviceMgr> device_mgr =
        absl::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
            "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
    bool async = false;
    bool lazy_remote_tensor_copy = false;
    tensorflow::Rendezvous* rendezvous =
        new tensorflow::IntraProcessRendezvous(device_mgr.get());
    std::unique_ptr<tensorflow::EagerContext> eager_ctx =
        std::unique_ptr<tensorflow::EagerContext>(new tensorflow::EagerContext(
            SessionOptions(),
            tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
            tensorflow::ContextMirroringPolicy::MIRRORING_NONE, async,
            lazy_remote_tensor_copy, device_mgr.get(), false, rendezvous,
            GetDefaultCustomKernelCreator()));

    EagerExecutor executor_(false);
    std::unique_ptr<tensorflow::EagerOperation> op(
        new tensorflow::EagerOperation(eager_ctx.get()));
    EXPECT_EQ(Status::OK(),
              op.get()->Reset(op_name.c_str(), nullptr, false, &executor_));
    return op;
  }

  // Validates the result of MKL eager rewrite.
  static void CheckRewrite(EagerOperation* orig_op, string expected_op_name) {
    std::unique_ptr<tensorflow::EagerOperation> out_op;
    EagerOpRewriteRegistry::Global()->RunRewrite(
        EagerOpRewriteRegistry::PRE_EXECUTION, orig_op, &out_op);

    // actual_op_name is same as original op name if rewrite didn't happen.
    string actual_op_name = orig_op->Name();
    if (out_op) {
      actual_op_name = out_op->Name();
    }

    EXPECT_EQ(actual_op_name, expected_op_name);
  }
};

TEST(EagerOpRewriteTest, Conv2D) {
  const string orig_op_name = "Conv2D";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);
  orig_op->MutableAttrs()->Set("padding", "VALID");

  EagerOpRewriteTest::CheckRewrite(orig_op.get(), "_MklEagerConv2D");
}

TEST(EagerOpRewriteTest, Conv2D_Explicit_Padding) {
  const string orig_op_name = "Conv2D";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);
  orig_op->MutableAttrs()->Set("padding", "EXPLICIT");

  EagerOpRewriteTest::CheckRewrite(orig_op.get(), "Conv2D");
}

TEST(EagerOpRewriteTest, Conv2DBackpropInput) {
  const string orig_op_name = "Conv2DBackpropInput";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);
  orig_op->MutableAttrs()->Set("padding", "VALID");

  EagerOpRewriteTest::CheckRewrite(orig_op.get(),
                                   "_MklEagerConv2DBackpropInput");
}

TEST(EagerOpRewriteTest, Conv2DBackpropFilter) {
  const string orig_op_name = "Conv2DBackpropFilter";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);
  orig_op->MutableAttrs()->Set("padding", "VALID");

  EagerOpRewriteTest::CheckRewrite(orig_op.get(),
                                   "_MklEagerConv2DBackpropFilter");
}

TEST(EagerOpRewriteTest, BatchMatMul) {
  const string orig_op_name = "BatchMatMul";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);

  EagerOpRewriteTest::CheckRewrite(orig_op.get(), "_MklBatchMatMul");
}

TEST(EagerOpRewriteTest, MatMul) {
  const string orig_op_name = "MatMul";
  std::unique_ptr<tensorflow::EagerOperation> orig_op =
      EagerOpRewriteTest::CreateOp(orig_op_name);

  orig_op->MutableAttrs()->Set("T", DT_FLOAT);

  EagerOpRewriteTest::CheckRewrite(orig_op.get(), "_MklMatMul");
}

}  // namespace tensorflow

#endif  // INTEL_MKL
