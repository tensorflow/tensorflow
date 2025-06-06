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
#include <memory>
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

}  // namespace
}  // namespace tflite
