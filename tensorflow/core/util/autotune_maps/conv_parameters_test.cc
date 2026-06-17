/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/autotune_maps/conv_parameters.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using ::testing::StrEq;

TEST(DeviceIdentifierForAutotuning, RoundsUpRamToGb) {
  EXPECT_THAT(
      DeviceIdentifierForAutotuning(
          "sm_7.0 with 98765432100B RAM, 97 cores, 777KHz clock, 204KHz "
          "mem clock, 888B L2$"),
      StrEq("sm_7.0 with 98.8GB RAM, 97 cores, 777KHz clock, 204KHz mem clock, "
            "888B L2$"));
}

TEST(DeviceIdentifierForAutotuning, NoChangeIfRegexDoesNotMatch) {
  EXPECT_THAT(DeviceIdentifierForAutotuning("expect no change 123456B RAM"),
              StrEq("expect no change 123456B RAM"));
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace

}  // namespace tensorflow
