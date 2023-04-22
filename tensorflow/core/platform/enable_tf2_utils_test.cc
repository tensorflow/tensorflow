/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Testing TF2 enablement.

#include "tensorflow/core/platform/enable_tf2_utils.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

TEST(TF2EnabledTest, enabled_behavior) {
  string tf2_env;
  TF_CHECK_OK(ReadStringFromEnvVar("TF2_BEHAVIOR", "0", &tf2_env));
  bool expected = (tf2_env != "0");
  EXPECT_EQ(tensorflow::tf2_execution_enabled(), expected);
  tensorflow::set_tf2_execution(true);
  EXPECT_TRUE(tensorflow::tf2_execution_enabled());
  tensorflow::set_tf2_execution(false);
  EXPECT_FALSE(tensorflow::tf2_execution_enabled());
}

}  // namespace tensorflow
