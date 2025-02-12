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

#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {

TEST(Winograd, CorrectAttributesFor4x4To6x6) {
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 2);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(1, 3, 3, 1);
  EXPECT_TRUE(IsSuitableForWinograd4x4To6x6(attr));
}

TEST(Winograd, IncorrectAttributesFor4x4To6x6) {
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 2);
  attr.padding.appended = HW(0, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(1, 2, 3, 1);
  EXPECT_FALSE(IsSuitableForWinograd4x4To6x6(attr));
}

}  // namespace gpu
}  // namespace tflite
