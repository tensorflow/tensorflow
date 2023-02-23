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

#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestEagerOpRewrite : public EagerOpRewrite {
 public:
  TestEagerOpRewrite(string name, string file, string line)
      : EagerOpRewrite(name, file, line),
        executor_(/*async=*/false, /*enable_streaming_enqueue=*/true) {}
  static int count_;
  EagerExecutor executor_;
  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op) override {
    ++count_;
    // Create a new NoOp Eager operation.
    tensorflow::EagerOperation* op =
        new tensorflow::EagerOperation(&orig_op->EagerContext());
    TF_RETURN_IF_ERROR(op->Reset("NoOp", nullptr, false, &executor_));
    out_op->reset(op);
    return OkStatus();
  }
};

int TestEagerOpRewrite::count_ = 0;

// Register two rewriter passes during the PRE_EXECUTION phase
REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, 10000,
                 TestEagerOpRewrite);
REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, 10001,
                 TestEagerOpRewrite);

TEST(EagerOpRewriteRegistryTest, RegisterRewritePass) {
  EXPECT_EQ(0, TestEagerOpRewrite::count_);
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  tensorflow::EagerContext* ctx = new tensorflow::EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true);
  EagerOperation orig_op(ctx);
  std::unique_ptr<tensorflow::EagerOperation> out_op;
  EXPECT_EQ(OkStatus(),
            EagerOpRewriteRegistry::Global()->RunRewrite(
                EagerOpRewriteRegistry::PRE_EXECUTION, &orig_op, &out_op));
  EXPECT_EQ(2, TestEagerOpRewrite::count_);
  EXPECT_EQ("NoOp", out_op->Name());
  ctx->Unref();
}

}  // namespace tensorflow
