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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/model_building.h"
#include "tensorflow/lite/core/subgraph.h"
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

  model_builder::ModelBuilder BuildFullyConnectedGraph(
      int num_subgraphs = 1, std::vector<int8_t>* external_weights = nullptr) {
    model_builder::ModelBuilder builder;
    model_builder::Quantization weights_quantization =
        model_builder::AffineQuantization{/*zero_points=*/{0},
                                          /*scales=*/{1.45},
                                          /*axis=*/1};
    auto weights = NewConstantBuffer(builder);
    if (external_weights != nullptr) {
      model_builder::Assign(
          weights, kTfLiteInt8, {3, 4},
          reinterpret_cast<char*>(external_weights->data()), +[](char*) {},
          external_weights->size(), std::move(weights_quantization));
    } else {
      Assign<kTfLiteInt8>(
          weights, {3, 4},
          std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
          std::move(weights_quantization));
    }
    for (int s = 0; s < num_subgraphs; ++s) {
      auto graph = NewGraph(builder);
      auto inputs = NewInput(graph, kTfLiteFloat32);
      SetShape(inputs, {2, 4});
      auto out = FullyConnected(inputs, weights);
      model_builder::MarkOutputs({out});
    }
    return builder;
  }

  void InvokeFullyConnectedSubgraph(Subgraph* subgraph) {
    ASSERT_NE(subgraph, nullptr);
    ASSERT_EQ(subgraph->AllocateTensors(), kTfLiteOk);
    TfLiteTensor* input = subgraph->tensor(subgraph->inputs()[0]);
    for (int i = 0; i < 8; ++i) {
      input->data.f[i] = static_cast<float>(i);
    }
    ASSERT_EQ(subgraph->Invoke(), kTfLiteOk);
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
    InvokeFullyConnectedSubgraph(interpreter.subgraph(0));
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
    InvokeFullyConnectedSubgraph(interpreter.subgraph(0));
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
       DefaultInMemoryCacheAutomaticallyDeduplicatesMultiSubgraphWeights) {
  model_builder::ModelBuilder builder =
      BuildFullyConnectedGraph(/*num_subgraphs=*/3);
  tflite::Interpreter interpreter;
  builder.Build(interpreter);
  ASSERT_GE(interpreter.subgraphs_size(), 3);

  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> xnnpack_delegate(
      TfLiteXNNPackDelegateCreate(&xnnpack_options),
      TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(std::move(xnnpack_delegate)),
            kTfLiteOk);

  for (int s = 0; s < 3; ++s) {
    InvokeFullyConnectedSubgraph(interpreter.subgraph(s));
  }

  const TfLiteTensor* out0 =
      interpreter.subgraph(0)->tensor(interpreter.subgraph(0)->outputs()[0]);
  const TfLiteTensor* out1 =
      interpreter.subgraph(1)->tensor(interpreter.subgraph(1)->outputs()[0]);
  const TfLiteTensor* out2 =
      interpreter.subgraph(2)->tensor(interpreter.subgraph(2)->outputs()[0]);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(out0->data.f[i], out1->data.f[i]);
    EXPECT_FLOAT_EQ(out0->data.f[i], out2->data.f[i]);
  }
}

TEST_F(XnnpackDelegateWeightCacheTest,
       DefaultInMemoryCacheIsIsolatedPerDelegateUnlessExplicitlyShared) {
  std::vector<int8_t> shared_weight_bytes = {1, 2, 3, 4,  5,  6,
                                             7, 8, 9, 10, 11, 12};
  model_builder::ModelBuilder builder0 =
      BuildFullyConnectedGraph(/*num_subgraphs=*/2, &shared_weight_bytes);
  tflite::Interpreter interpreter0;
  builder0.Build(interpreter0);
  ASSERT_GE(interpreter0.subgraphs_size(), 2);

  model_builder::ModelBuilder builder1 =
      BuildFullyConnectedGraph(/*num_subgraphs=*/2, &shared_weight_bytes);
  tflite::Interpreter interpreter1;
  builder1.Build(interpreter1);
  ASSERT_GE(interpreter1.subgraphs_size(), 2);

  TfLiteXNNPackDelegateOptions xnnpack_options = GetXNNPackOptions();
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate0(
      TfLiteXNNPackDelegateCreate(&xnnpack_options),
      TfLiteXNNPackDelegateDelete);
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate1(
      TfLiteXNNPackDelegateCreate(&xnnpack_options),
      TfLiteXNNPackDelegateDelete);

  ASSERT_EQ(interpreter0.ModifyGraphWithDelegate(delegate0.get()), kTfLiteOk);
  ASSERT_EQ(interpreter1.ModifyGraphWithDelegate(delegate1.get()), kTfLiteOk);

  if (TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider()) {
    const TfLiteXNNPackDelegateOptions* opts0 =
        TfLiteXNNPackDelegateGetOptions(delegate0.get());
    const TfLiteXNNPackDelegateOptions* opts1 =
        TfLiteXNNPackDelegateGetOptions(delegate1.get());
    ASSERT_NE(opts0, nullptr);
    ASSERT_NE(opts1, nullptr);
    EXPECT_NE(opts0->weights_cache, nullptr);
    EXPECT_NE(opts1->weights_cache, nullptr);
    EXPECT_NE(opts0->weights_cache, opts1->weights_cache);
  }

  Subgraph* subgraph0 = interpreter0.subgraph(0);
  Subgraph* subgraph1 = interpreter1.subgraph(1);
  InvokeFullyConnectedSubgraph(subgraph0);
  InvokeFullyConnectedSubgraph(subgraph1);

  const TfLiteTensor* out0 = subgraph0->tensor(subgraph0->outputs()[0]);
  const TfLiteTensor* out1 = subgraph1->tensor(subgraph1->outputs()[0]);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(out0->data.f[i], out1->data.f[i]);
  }
}

}  // namespace
}  // namespace tflite
