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
#include "tensorflow/lite/tools/optimize/model_utils.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace tflite {
namespace optimize {
namespace utils {
namespace {

TEST(ModelUtilsTest, QuantizationParametersExist) {
  TensorT tensor;
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.quantization->scale.push_back(0.5);
  tensor.quantization->scale.push_back(1.5);
  EXPECT_FALSE(QuantizationParametersExist(&tensor));
  tensor.quantization->zero_point.push_back(1);
  tensor.quantization->zero_point.push_back(-1);
  EXPECT_TRUE(QuantizationParametersExist(&tensor));
}

TEST(ModelUtilsTest, HasBuffer) {
  tflite::ModelT model;
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<tflite::TensorT>();
  tensor->buffer = 0;
  subgraph->tensors.push_back(std::move(tensor));
  model.subgraphs.push_back(std::move(subgraph));
  auto buffer = std::make_unique<tflite::BufferT>();
  model.buffers.push_back(std::move(buffer));
  EXPECT_FALSE(HasBuffer(&model, model.subgraphs[0].get(), 0));
  model.buffers[0]->data = {0, 1, 2, 3};
  EXPECT_TRUE(HasBuffer(&model, model.subgraphs[0].get(), 0));
}

TEST(ModelUtilsTest, HasMinMax) {
  TensorT tensor;
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.quantization->min.push_back(0.5);
  EXPECT_FALSE(HasMinMax(&tensor));
  tensor.quantization->max.push_back(1.5);
  EXPECT_TRUE(HasMinMax(&tensor));
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite
