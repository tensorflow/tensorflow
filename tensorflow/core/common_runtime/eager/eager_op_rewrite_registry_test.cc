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
      : EagerOpRewrite(name, file, line), executor_(/*async=*/false) {}
  static int count_;
  EagerExecutor executor_;
  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op) override {
    ++count_;
    const tensorflow::AttrTypeMap* types;
    bool is_function = false;
    const string kNewOp = "NoOp";
    TF_RETURN_IF_ERROR(
        tensorflow::AttrTypeMapForOp(kNewOp.c_str(), &types, &is_function));
    // Create a new NoOp Eager operation.
    out_op->reset(new tensorflow::EagerOperation(
        nullptr, kNewOp.c_str(), is_function, types, &executor_));
    return Status::OK();
  }
};

int TestEagerOpRewrite::count_ = 0;

REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, TestEagerOpRewrite);

TEST(EagerOpRewriteRegistryTest, RegisterRewritePass) {
  EXPECT_EQ(0, TestEagerOpRewrite::count_);
  EagerOperation* orig_op = nullptr;
  std::unique_ptr<tensorflow::EagerOperation> out_op;
  EXPECT_EQ(Status::OK(),
            EagerOpRewriteRegistry::Global()->RunRewrite(
                EagerOpRewriteRegistry::PRE_EXECUTION, orig_op, &out_op));
  EXPECT_EQ(1, TestEagerOpRewrite::count_);
  EXPECT_EQ("NoOp", out_op->Name());
}

}  // namespace tensorflow
