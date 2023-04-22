/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/ops/compat/op_compatibility_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(BackwardsCompatibilityTest, IsCompatible) {
  OpCompatibilityLib compatibility("tensorflow/core/ops",
                                   strings::StrCat("v", TF_MAJOR_VERSION),
                                   nullptr);

  Env* env = Env::Default();
  int changed_ops = 0;
  int added_ops = 0;
  TF_ASSERT_OK(
      compatibility.ValidateCompatible(env, &changed_ops, &added_ops, nullptr));
  printf("%d changed ops\n%d added ops\n", changed_ops, added_ops);
}

}  // namespace
}  // namespace tensorflow
