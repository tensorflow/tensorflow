/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/acceleration/configuration/stable_delegate_registry.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"

namespace {

using tflite::delegates::StableDelegateRegistry;

TfLiteStableDelegate CreateTestStableDelegate() {
  TfLiteStableDelegate stable_delegate = {TFL_STABLE_DELEGATE_ABI_VERSION,
                                          "test_delegate", "V1.0.0", nullptr};
  return stable_delegate;
}

class StableDelegateRegistryTest : public testing::Test {
 public:
  void SetUp() override {
    stable_delegate_ = CreateTestStableDelegate();
    StableDelegateRegistry::RegisterStableDelegate(&stable_delegate_);
  }

 protected:
  TfLiteStableDelegate stable_delegate_;
};

TEST_F(StableDelegateRegistryTest, TestRetrieval) {
  EXPECT_EQ(StableDelegateRegistry::RetrieveStableDelegate("test_delegate"),
            &stable_delegate_);
}

TEST_F(StableDelegateRegistryTest, NoRegistrationFound) {
  EXPECT_EQ(
      StableDelegateRegistry::RetrieveStableDelegate("not_valid_delegate"),
      nullptr);
}

}  // namespace
