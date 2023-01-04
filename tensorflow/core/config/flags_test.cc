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

#include "tensorflow/core/config/flags.h"

#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TFFlags, ReadFlagValue) {
  EXPECT_TRUE(flags::Global().test_only_experiment_1.value());
  EXPECT_FALSE(flags::Global().test_only_experiment_2.value());
}

TEST(TFFlags, ResetFlagValue) {
  EXPECT_TRUE(flags::Global().test_only_experiment_1.value());
  flags::Global().test_only_experiment_1.reset(false);
  EXPECT_FALSE(flags::Global().test_only_experiment_1.value());
}

}  // namespace
}  // namespace tensorflow
