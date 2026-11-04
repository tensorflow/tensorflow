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
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/model_building.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace {

class XnnpackDelegateWeightCacheTest : public testing::Test {
 public:
  TfLiteXNNPackDelegateOptions GetXNNPackOptions() {
    TfLiteXNNPackDelegateOptions xnnpack_options =
        TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.flags |=
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
    xnnpack_options.flags |=
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
    return xnnpack_options;
  }

  model_builder::ModelBuilder BuildFullyConnectedGraph() {
    model_builder::ModelBuilder builder;
    model_builder::Quantization weights_quantization =
        model_builder::AffineQuantization{/*zero_points=*/{0},
                                          /*scales=*/{1.45},
                                          /*axis=*/1};
    auto weights = NewConstantBuffer(builder);
    Assign<kTfLiteInt8>(
        weights, {3, 4},
        std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        std::move(weights_quantization));
    auto graph = NewGraph(builder);
    auto inputs = NewInput(graph, kTfLiteFloat32);
    SetShape(inputs, {2, 4});
    auto out = FullyConnected(inputs, weights);
    model_builder::MarkOutputs({out});
    return builder;
  }
};

TEST_F(XnnpackDelegateWeightCacheTest,
       ReuseImplicitlyLoadedCacheAcrossDelegates) {
  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  xnnpack::MMapWeightCacheProvider shared_weight_cache_provider;
  xnnpack_options.weight_cache_provider = &shared_weight_cache_provider;
  xnnpack_options.weight_cache_file_path =
      TfLiteXNNPackDelegateInMemoryFilePath();

  for (int i = 0; i < 2; ++i) {
    model_builder::ModelBuilder builder = BuildFullyConnectedGraph();

    tflite::Interpreter interpreter;
    builder.Build(interpreter);

    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
        TfLiteXNNPackDelegateCreate(&xnnpack_options),
        TfLiteXNNPackDelegateDelete);

    interpreter.ModifyGraphWithDelegate(std::move(xnnpack_delegate));

    EXPECT_TRUE(shared_weight_cache_provider.IsActive());

    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input_tensor(0);
    for (int i = 0; i < 8; ++i) {
      input->data.f[i] = i;
    }

    ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  }
}

TEST_F(XnnpackDelegateWeightCacheTest,
       ReuseExplicitlyLoadedCacheAcrossDelegates) {
  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  xnnpack::MMapWeightCacheProvider shared_weight_cache_provider;
  xnnpack_options.weight_cache_provider = &shared_weight_cache_provider;
  ASSERT_TRUE(shared_weight_cache_provider.LoadOrStartBuild(
      TfLiteXNNPackDelegateInMemoryFilePath()));

  for (int i = 0; i < 2; ++i) {
    model_builder::ModelBuilder builder = BuildFullyConnectedGraph();

    tflite::Interpreter interpreter;
    builder.Build(interpreter);

    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
        TfLiteXNNPackDelegateCreate(&xnnpack_options),
        TfLiteXNNPackDelegateDelete);

    interpreter.ModifyGraphWithDelegate(std::move(xnnpack_delegate));
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input_tensor(0);
    for (int i = 0; i < 8; ++i) {
      input->data.f[i] = i;
    }

    ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  }
}

TEST_F(XnnpackDelegateWeightCacheTest,
       ImplicitlyLoadedCacheFailsIfNoPathOrFileDescriptorIsProvided) {
  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  xnnpack::MMapWeightCacheProvider shared_weight_cache_provider;
  xnnpack_options.weight_cache_provider = &shared_weight_cache_provider;

  model_builder::ModelBuilder builder = BuildFullyConnectedGraph();

  tflite::Interpreter interpreter;
  builder.Build(interpreter);

  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
      TfLiteXNNPackDelegateCreate(&xnnpack_options),
      TfLiteXNNPackDelegateDelete);

  EXPECT_EQ(interpreter.ModifyGraphWithDelegate(std::move(xnnpack_delegate)),
            kTfLiteOk);
  EXPECT_FALSE(shared_weight_cache_provider.IsActive());
}

TEST_F(XnnpackDelegateWeightCacheTest,
       ReuseCacheAcrossMultiplePartitionsInSubgraph) {
  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  xnnpack::MMapWeightCacheProvider shared_weight_cache_provider;
  xnnpack_options.weight_cache_provider = &shared_weight_cache_provider;
  const std::string cache_path = testing::TempDir() + "/partition_cache";
  std::remove(cache_path.c_str());
  xnnpack_options.weight_cache_file_path = cache_path.c_str();

  std::vector<float> first_run_output;
  for (int run = 0; run < 2; ++run) {
    // Define quantized weights for the two FullyConnected partitions.
    model_builder::ModelBuilder builder;
    auto weights1 = NewConstantBuffer(builder);
    Assign<kTfLiteInt8>(weights1, {3, 4},
                        std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                        model_builder::AffineQuantization{/*zero_points=*/{0},
                                                          /*scales=*/{1.45},
                                                          /*axis=*/1});
    auto weights2 = NewConstantBuffer(builder);
    Assign<kTfLiteInt8>(weights2, {2, 3}, std::vector<int>{2, -1, 3, 1, 4, -2},
                        model_builder::AffineQuantization{/*zero_points=*/{0},
                                                          /*scales=*/{0.5},
                                                          /*axis=*/1});

    // Weightless decomposition subgraph referenced by the composite op below;
    // exercises an empty build step in `DelegatePrepare`.
    auto main_graph = NewGraph(builder);
    auto fallback_graph = NewGraph(builder);
    auto fallback_in = NewInput(fallback_graph, kTfLiteFloat32);
    SetShape(fallback_in, {2, 3});
    MarkOutput(Add(fallback_in, fallback_in));

    // Main subgraph: `FullyConnected -> StableHLOComposite -> FullyConnected`.
    // The unsupported composite op splits `main_graph` into two XNNPACK
    // partitions that share one build step in `DelegatePrepare`.
    auto input = NewInput(main_graph, kTfLiteFloat32);
    SetShape(input, {2, 4});
    auto fc1 = FullyConnected(input, weights1);
    auto mid = StableHLOComposite("custom_fallback", fallback_graph, {fc1});
    SetShape(mid[0], {2, 3});
    MarkOutput(FullyConnected(mid[0], weights2));

    // Delegate the interpreter; `run == 0` lets `DelegatePrepare` scope a
    // single build step across both XNNPACK partitions, while `run == 1`
    // starts an outer build step and verifies that partition creation does not
    // prematurely stop it.
    tflite::Interpreter interpreter;
    builder.Build(interpreter);

    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
        TfLiteXNNPackDelegateCreate(&xnnpack_options),
        TfLiteXNNPackDelegateDelete);

    if (run == 1) {
      ASSERT_TRUE(shared_weight_cache_provider.StartBuildStep());
    }
    ASSERT_EQ(interpreter.ModifyGraphWithDelegate(std::move(xnnpack_delegate)),
              kTfLiteOk);
    EXPECT_EQ(interpreter.execution_plan().size(), 3);
    EXPECT_TRUE(shared_weight_cache_provider.IsActive());
    if (run == 1) {
      EXPECT_TRUE(shared_weight_cache_provider.IsBuilding());
      ASSERT_TRUE(shared_weight_cache_provider.StopBuildStep());
    }
    EXPECT_FALSE(shared_weight_cache_provider.IsBuilding());

    // Run inference and verify that the cache-hit run matches the cache-build
    // run.
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    TfLiteTensor* input_tensor = interpreter.input_tensor(0);
    for (int i = 0; i < 8; ++i) {
      input_tensor->data.f[i] = static_cast<float>(i);
    }
    ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

    const TfLiteTensor* output_tensor = interpreter.output_tensor(0);
    std::vector<float> current_output(output_tensor->data.f,
                                      output_tensor->data.f + 4);
    if (run == 0) {
      first_run_output = std::move(current_output);
    } else {
      EXPECT_EQ(current_output, first_run_output);
    }
  }
}

}  // namespace
}  // namespace tflite
