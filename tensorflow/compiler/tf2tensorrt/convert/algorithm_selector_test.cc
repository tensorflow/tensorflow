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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h"

#include <memory>

#include <gtest/gtest.h>
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

TEST(TestAlgorithmSelector, TensorRT7_1) {
  // Verify that the algorithm selector for TRT 7.1 is not required.
  AlgorithmSelectorImpl sel71({7, 1, 3, 4});
  ASSERT_FALSE(sel71.IsAlgorithmSelectorRequired());
}

TEST(TestAlgorithmSelector, TensorRT7_2) {
  // Verify that the algorithm selector for TRT 7.2 is required.
  AlgorithmSelectorImpl sel72({7, 2, 0, 0});
  ASSERT_TRUE(sel72.IsAlgorithmSelectorRequired());

  // Check that the correct tactics are banned.
  auto turing_tactics = AlgorithmSelectorImpl::GetBannedTRT72TuringTactics();

  for (auto id : turing_tactics) {
    EXPECT_TRUE(sel72.IsBannedTactic(id));
  }

  // Check that a bad shuffle format is banned.
  EXPECT_FALSE(sel72.AllowShuffleAlgorithm(0, nvinfer1::DataType::kFLOAT,
                                           nvinfer1::TensorFormat::kCHW32));

  // Check that other formats are not banned.
  EXPECT_TRUE(sel72.AllowShuffleAlgorithm(0, nvinfer1::DataType::kHALF,
                                          nvinfer1::TensorFormat::kCHW32));
  EXPECT_TRUE(sel72.AllowShuffleAlgorithm(0, nvinfer1::DataType::kINT32,
                                          nvinfer1::TensorFormat::kCHW32));
  EXPECT_TRUE(sel72.AllowShuffleAlgorithm(0, nvinfer1::DataType::kFLOAT,
                                          nvinfer1::TensorFormat::kCHW16));
}

TEST(TestAlgorithmSelector, TensorRT8_0) {
  // Verify that the algorithm selector for TRT 8.0 is required.
  AlgorithmSelectorImpl sel80({8, 0, 1, 6});
  ASSERT_TRUE(sel80.IsAlgorithmSelectorRequired());

  // Check that the turing 7.2 tactics are not banned.
  auto turing_tactics = AlgorithmSelectorImpl::GetBannedTRT72TuringTactics();
  for (auto id : turing_tactics) {
    EXPECT_FALSE(sel80.IsBannedTactic(id));
  }

  // Check that a bad shuffle format is banned.
  EXPECT_FALSE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kINT8,
                                           nvinfer1::TensorFormat::kLINEAR));

  // Check that other formats are not banned.
  EXPECT_TRUE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kHALF,
                                          nvinfer1::TensorFormat::kLINEAR));
  EXPECT_TRUE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kINT32,
                                          nvinfer1::TensorFormat::kLINEAR));
  EXPECT_TRUE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kFLOAT,
                                          nvinfer1::TensorFormat::kLINEAR));
  EXPECT_TRUE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kINT8,
                                          nvinfer1::TensorFormat::kCHW16));
  EXPECT_TRUE(sel80.AllowShuffleAlgorithm(0, nvinfer1::DataType::kINT8,
                                          nvinfer1::TensorFormat::kCHW32));
}

TEST(TestAlgorithmSelector, TensorRT8_2) {
  // Verify that the algorithm selector for TRT 8.0 is required.
  AlgorithmSelectorImpl sel({8, 2, 0, 0});
  ASSERT_FALSE(sel.IsAlgorithmSelectorRequired());
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
