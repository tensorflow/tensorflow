// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_

#include <cstdint>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

constexpr const char* kModelFileName = "simple_model.tflite";
constexpr const char* kQualcommModelFileName = "simple_model_qualcomm.bin";
constexpr const char* kPixelModelFileName = "simple_model_pixel.bin";

constexpr const int32_t kTestInput0Dimensions[] = {2};
constexpr const int32_t kNumTestInput0Dimensions =
    sizeof(kTestInput0Dimensions) / sizeof(kTestInput0Dimensions[0]);
constexpr const int32_t kTestInput1Dimensions[] = {2};
constexpr const int32_t kNumTestInput1Dimensions =
    sizeof(kTestInput1Dimensions) / sizeof(kTestInput1Dimensions[0]);
constexpr const int32_t kTestOutputDimensions[] = {2};
constexpr const int32_t kNumTestOutputDimensions =
    sizeof(kTestOutputDimensions) / sizeof(kTestOutputDimensions[0]);

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const float kTestOutputTensor[] = {11, 22};

constexpr const LrtRankedTensorType kInput0TensorType = {
    /*.element_type=*/kLrtElementTypeFloat32,
    /*.layout=*/{
        /*.rank=*/kNumTestInput0Dimensions,
        /*.dimensions=*/kTestInput0Dimensions,
    }};

constexpr const LrtRankedTensorType kInput1TensorType = {
    /*.element_type=*/kLrtElementTypeFloat32,
    /*.layout=*/{
        /*.rank=*/kNumTestInput1Dimensions,
        /*.dimensions=*/kTestInput1Dimensions,
    }};

constexpr const LrtRankedTensorType kOutputTensorType = {
    /*.element_type=*/kLrtElementTypeFloat32,
    /*.layout=*/{
        /*.rank=*/kNumTestOutputDimensions,
        /*.dimensions=*/kTestOutputDimensions,
    }};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_TESTDATA_SIMPLE_MODEL_TEST_VECTORS_H_
