/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

void Register(const string& op_name, OpRegistry* registry) {
  registry->Register(
      [op_name](OpRegistrationData* op_reg_data) -> absl::Status {
        op_reg_data->op_def.set_name(op_name);
        return absl::OkStatus();
      });
}

}  // namespace

TEST(OpRegistrationTest, TestBasic) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("Foo", registry.get());
  OpList op_list;
  registry->Export(true, &op_list);
  EXPECT_EQ(op_list.op().size(), 1);
  EXPECT_EQ(op_list.op(0).name(), "Foo");
}

TEST(OpRegistrationTest, TestDuplicate) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("Foo", registry.get());
  absl::Status s = registry->ProcessRegistrations();
  EXPECT_TRUE(s.ok());

  TF_EXPECT_OK(registry->SetWatcher(
      [](const absl::Status& s, const OpDef& op_def) -> absl::Status {
        EXPECT_TRUE(absl::IsAlreadyExists(s));
        return absl::OkStatus();
      }));
  Register("Foo", registry.get());
  s = registry->ProcessRegistrations();
  EXPECT_TRUE(s.ok());
}

}  // namespace tensorflow
