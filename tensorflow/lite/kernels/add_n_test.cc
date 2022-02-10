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
#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/add_n_test_common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(FloatAddNOpModel, AddMultipleTensors) {
  FloatAddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}}},
                     {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  m.PopulateTensor<float>(m.input(2), {0.5, 0.1, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.4, 0.5, 1.1, 1.5}));
}

TEST(IntegerAddNOpModel, AddMultipleTensors) {
  IntegerAddNOpModel m({{TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input(0), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input(1), {1, 2, 3, 5});
  m.PopulateTensor<int32_t>(m.input(2), {10, -5, 1, -2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-9, -1, 11, 11}));
}

}  // namespace
}  // namespace tflite
