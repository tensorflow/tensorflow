/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gce_env_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/fake_env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

TEST(GcpEnvUtils, IsRunningOnGce) {
  {
    test::FakeEnv env(test::FakeEnv::kGoogle);
    bool is_running_on_gcp = false;
    TF_EXPECT_OK(IsRunningOnGce(&env, &is_running_on_gcp));
    EXPECT_TRUE(is_running_on_gcp);
  }
  {
    test::FakeEnv env(test::FakeEnv::kGce);
    bool is_running_on_gcp = false;
    TF_EXPECT_OK(IsRunningOnGce(&env, &is_running_on_gcp));
    EXPECT_TRUE(is_running_on_gcp);
  }
  {
    test::FakeEnv env(test::FakeEnv::kLocal);
    bool is_running_on_gcp = false;
    TF_EXPECT_OK(IsRunningOnGce(&env, &is_running_on_gcp));
    EXPECT_FALSE(is_running_on_gcp);
  }
  {
    test::FakeEnv env(test::FakeEnv::kBad);
    bool is_running_on_gcp = false;
    EXPECT_TRUE(errors::IsInternal(IsRunningOnGce(&env, &is_running_on_gcp)));
  }
}

}  // namespace
}  // namespace tensorflow
