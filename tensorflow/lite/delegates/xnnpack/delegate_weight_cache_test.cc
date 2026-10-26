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
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/model_building.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"
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

using DelegatePtr = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

// Tests the in-memory weight cache that the delegate enables by default for
// models with more than one subgraph.
//
// The cache is observed through the size of its `[anon:xnnpack-weights-cache]`
// mappings in `/proc/self/maps`.
class XnnpackDelegateDefaultWeightCacheTest
    : public XnnpackDelegateWeightCacheTest {
 public:
  struct Model {
    int num_subgraphs = 2;
    // Whether the subgraphs use the same weight buffer.
    bool share_weights = true;
    // First weight value.
    int weights = 1;
  };

  void SetUp() override {
    if (!TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider()) {
      GTEST_SKIP() << "In-memory weight caches aren't supported.";
    }
    xnnpack::MMapHandle probe;
    ASSERT_TRUE(probe.MapAnonymous(1, xnnpack::kInMemoryCacheVmaName));
    if (CacheBytes() == 0) {
      GTEST_SKIP() << "Anonymous mapping names aren't supported.";
    }
  }

  // Returns the size of the in-memory weight cache mappings.
  static size_t CacheBytes() {
    const std::string vma_name =
        std::string("[anon:") + xnnpack::kInMemoryCacheVmaName + "]";
    size_t bytes = 0;
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
      uintptr_t start = 0;
      uintptr_t end = 0;
      if (line.find(vma_name) != std::string::npos &&
          std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &start, &end) ==
              2) {
        bytes += end - start;
      }
    }
    return bytes;
  }

  // Builds an interpreter with a FULLY_CONNECTED subgraph per signature.
  std::unique_ptr<Interpreter> Build(const Model& model) {
    // The builder owns the weights.
    model_builder::ModelBuilder& builder = *builders_.emplace_back(
        std::make_unique<model_builder::ModelBuilder>());
    auto new_weights = [&] {
      model_builder::Buffer buffer = NewConstantBuffer(builder);
      std::vector<float> values(12);
      std::iota(values.begin(), values.end(), model.weights);
      Assign<kTfLiteFloat32>(buffer, {3, 4}, values,
                             model_builder::NoQuantization{});
      return buffer;
    };
    const model_builder::Buffer shared_weights = new_weights();
    for (int i = 0; i < model.num_subgraphs; ++i) {
      auto graph = NewGraph(builder);
      auto inputs = NewInput(graph, kTfLiteFloat32);
      SetShape(inputs, {2, 4});
      auto out = FullyConnected(
          inputs, model.share_weights ? shared_weights : new_weights());
      model_builder::MarkOutputs({out});
    }
    auto interpreter = std::make_unique<Interpreter>();
    builder.Build(*interpreter);
    return interpreter;
  }

  DelegatePtr CreateDelegate(uint32_t extra_flags = 0) {
    TfLiteXNNPackDelegateOptions options = GetXNNPackOptions();
    options.flags |= extra_flags;
    return DelegatePtr(TfLiteXNNPackDelegateCreate(&options),
                       TfLiteXNNPackDelegateDelete);
  }

  // Runs subgraph `index` on a fixed input and returns its output.
  static std::vector<float> Run(Interpreter& interpreter, int index) {
    Subgraph* subgraph = interpreter.subgraph(index);
    EXPECT_EQ(subgraph->AllocateTensors(), kTfLiteOk);
    float* input = subgraph->tensor(subgraph->inputs()[0])->data.f;
    std::iota(input, input + 8, 0.0f);
    EXPECT_EQ(subgraph->Invoke(), kTfLiteOk);
    const float* output = subgraph->tensor(subgraph->outputs()[0])->data.f;
    return std::vector<float>(output, output + 6);
  }

  // Runs subgraph `index` of `model` without weight cache.
  std::vector<float> RunWithoutCache(const Model& model, int index) {
    std::unique_ptr<Interpreter> interpreter = Build(model);
    DelegatePtr delegate = CreateDelegate(
        TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_DEFAULT_WEIGHT_CACHE);
    EXPECT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
    return Run(*interpreter, index);
  }

 private:
  std::vector<std::unique_ptr<model_builder::ModelBuilder>> builders_;
};

TEST_F(XnnpackDelegateDefaultWeightCacheTest, DeduplicatesSharedWeights) {
  const Model shared_model{/*num_subgraphs=*/3, /*share_weights=*/true};
  const Model distinct_model{/*num_subgraphs=*/3, /*share_weights=*/false};
  const size_t initial_bytes = CacheBytes();

  std::unique_ptr<Interpreter> shared = Build(shared_model);
  DelegatePtr shared_delegate = CreateDelegate();
  ASSERT_EQ(shared->ModifyGraphWithDelegate(shared_delegate.get()), kTfLiteOk);
  const size_t shared_bytes = CacheBytes() - initial_bytes;

  std::unique_ptr<Interpreter> distinct = Build(distinct_model);
  DelegatePtr distinct_delegate = CreateDelegate();
  ASSERT_EQ(distinct->ModifyGraphWithDelegate(distinct_delegate.get()),
            kTfLiteOk);
  const size_t distinct_bytes = CacheBytes() - initial_bytes - shared_bytes;

  EXPECT_GT(shared_bytes, 0);
  EXPECT_LT(shared_bytes, distinct_bytes);
  // Delegates created from copied options must not share the cache.
  EXPECT_EQ(
      TfLiteXNNPackDelegateGetOptions(shared_delegate.get())->weights_cache,
      nullptr);
  for (int i = 0; i < shared_model.num_subgraphs; ++i) {
    EXPECT_EQ(Run(*shared, i), RunWithoutCache(shared_model, i));
  }
}

TEST_F(XnnpackDelegateDefaultWeightCacheTest, IsNotEnabledForOneSubgraph) {
  const size_t initial_bytes = CacheBytes();
  std::unique_ptr<Interpreter> interpreter = Build({/*num_subgraphs=*/1});
  DelegatePtr delegate = CreateDelegate();
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
  EXPECT_EQ(CacheBytes(), initial_bytes);
}

TEST_F(XnnpackDelegateDefaultWeightCacheTest, CanBeDisabled) {
  const size_t initial_bytes = CacheBytes();
  std::unique_ptr<Interpreter> interpreter = Build({/*num_subgraphs=*/3});
  DelegatePtr delegate =
      CreateDelegate(TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_DEFAULT_WEIGHT_CACHE);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
  EXPECT_EQ(CacheBytes(), initial_bytes);
}

TEST_F(XnnpackDelegateDefaultWeightCacheTest, IsNotReusedForAnotherModel) {
  // Both models have the same buffer identifiers but different weights.
  const Model model_1{/*num_subgraphs=*/2, /*share_weights=*/true,
                      /*weights=*/1};
  const Model model_2{/*num_subgraphs=*/2, /*share_weights=*/true,
                      /*weights=*/-12};
  const std::vector<float> expected_1 = RunWithoutCache(model_1, 1);
  const std::vector<float> expected_2 = RunWithoutCache(model_2, 1);
  ASSERT_NE(expected_1, expected_2);
  const size_t initial_bytes = CacheBytes();

  DelegatePtr delegate = CreateDelegate();
  {
    // One model after the other: the second model may get the address of the
    // first one.
    std::unique_ptr<Interpreter> interpreter_1 = Build(model_1);
    ASSERT_EQ(interpreter_1->ModifyGraphWithDelegate(delegate.get()),
              kTfLiteOk);
    EXPECT_GT(CacheBytes(), initial_bytes);
    EXPECT_EQ(Run(*interpreter_1, 1), expected_1);
    interpreter_1.reset();
    // The cache is freed with its model.
    EXPECT_EQ(CacheBytes(), initial_bytes);

    std::unique_ptr<Interpreter> interpreter_2 = Build(model_2);
    ASSERT_EQ(interpreter_2->ModifyGraphWithDelegate(delegate.get()),
              kTfLiteOk);
    EXPECT_EQ(Run(*interpreter_2, 1), expected_2);
  }
  {
    // Both models at the same time.
    std::unique_ptr<Interpreter> interpreter_1 = Build(model_1);
    ASSERT_EQ(interpreter_1->ModifyGraphWithDelegate(delegate.get()),
              kTfLiteOk);
    std::unique_ptr<Interpreter> interpreter_2 = Build(model_2);
    ASSERT_EQ(interpreter_2->ModifyGraphWithDelegate(delegate.get()),
              kTfLiteOk);
    EXPECT_EQ(Run(*interpreter_2, 1), expected_2);
    EXPECT_EQ(Run(*interpreter_1, 1), expected_1);
  }
  EXPECT_EQ(CacheBytes(), initial_bytes);
}

TEST_F(XnnpackDelegateDefaultWeightCacheTest, ExplicitCacheIsShared) {
  xnnpack::MMapWeightCacheProvider shared_weight_cache_provider;
  ASSERT_TRUE(shared_weight_cache_provider.LoadOrStartBuild(
      TfLiteXNNPackDelegateInMemoryFilePath()));
  TfLiteXNNPackDelegateOptions options = GetXNNPackOptions();
  options.weight_cache_provider = &shared_weight_cache_provider;
  DelegatePtr delegate_1(TfLiteXNNPackDelegateCreate(&options),
                         TfLiteXNNPackDelegateDelete);
  DelegatePtr delegate_2(TfLiteXNNPackDelegateCreate(&options),
                         TfLiteXNNPackDelegateDelete);
  const size_t initial_bytes = CacheBytes();

  std::unique_ptr<Interpreter> interpreter_1 = Build({});
  ASSERT_EQ(interpreter_1->ModifyGraphWithDelegate(delegate_1.get()),
            kTfLiteOk);
  const size_t bytes = CacheBytes();
  EXPECT_GT(bytes, initial_bytes);

  // The weights of the second model are already in the shared cache.
  std::unique_ptr<Interpreter> interpreter_2 = Build({});
  ASSERT_EQ(interpreter_2->ModifyGraphWithDelegate(delegate_2.get()),
            kTfLiteOk);
  EXPECT_EQ(CacheBytes(), bytes);
  EXPECT_EQ(Run(*interpreter_2, 1), Run(*interpreter_1, 1));
}

}  // namespace
}  // namespace tflite
