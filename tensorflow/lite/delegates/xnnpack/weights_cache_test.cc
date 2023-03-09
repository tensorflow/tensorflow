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

#include <memory>  // For std::unique_ptr.
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/delegates/xnnpack/conv_2d_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace xnnpack {

TEST(XNNPACK_WEIGHTS_CACHE, WithSize) {
  std::vector<char> buffer = Conv2DTester().CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter));
  ASSERT_EQ(kTfLiteOk, interpreter->AllocateTensors());

  size_t four_mb = 4194304;
  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreateWithSize(four_mb),
                    TfLiteXNNPackDelegateWeightsCacheDelete);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.weights_cache = weights_cache.get();

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  ASSERT_EQ(kTfLiteOk, interpreter->ModifyGraphWithDelegate(delegate.get()));

  ASSERT_TRUE(
      TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache.get()));

  ASSERT_EQ(kTfLiteOk, interpreter->Invoke());
}

TEST(XNNPACK_WEIGHTS_CACHE, InvokeBeforeFinalization) {
  std::vector<char> buffer = Conv2DTester().CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter));
  ASSERT_EQ(kTfLiteOk, interpreter->AllocateTensors());

  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.weights_cache = weights_cache.get();

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  ASSERT_EQ(kTfLiteOk, interpreter->ModifyGraphWithDelegate(delegate.get()));

  // Invoking before finalization fails.
  ASSERT_NE(kTfLiteOk, interpreter->Invoke());
}

TEST(XNNPACK_WEIGHTS_CACHE, HardFinalization) {
  std::vector<char> buffer = Conv2DTester().CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<Interpreter> interpreter1;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter1));
  ASSERT_EQ(kTfLiteOk, interpreter1->AllocateTensors());

  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.weights_cache = weights_cache.get();

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate1(TfLiteXNNPackDelegateCreate(&delegate_options),
                TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(kTfLiteOk, interpreter1->ModifyGraphWithDelegate(delegate1.get()));
  ASSERT_TRUE(
      TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache.get()));

  ASSERT_EQ(kTfLiteOk, interpreter1->Invoke());

  // We cannot create new instances using the same weights cache after hard
  // finalization.
  std::unique_ptr<Interpreter> interpreter2;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter2));
  ASSERT_EQ(kTfLiteOk, interpreter2->AllocateTensors());
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate2(TfLiteXNNPackDelegateCreate(&delegate_options),
                TfLiteXNNPackDelegateDelete);
  ASSERT_NE(kTfLiteOk, interpreter2->ModifyGraphWithDelegate(delegate2.get()));
}

TEST(XNNPACK_WEIGHTS_CACHE, SoftFinalization) {
  std::vector<char> buffer = Conv2DTester().CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.weights_cache = weights_cache.get();

  std::unique_ptr<Interpreter> interpreter1;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter1));
  ASSERT_EQ(kTfLiteOk, interpreter1->AllocateTensors());
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate1(TfLiteXNNPackDelegateCreate(&delegate_options),
                TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(kTfLiteOk, interpreter1->ModifyGraphWithDelegate(delegate1.get()));

  ASSERT_TRUE(
      TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(weights_cache.get()));

  ASSERT_EQ(kTfLiteOk, interpreter1->Invoke());

  // Build a second interpreter, it should work after soft finalization.
  std::unique_ptr<Interpreter> interpreter2;
  ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter2));
  ASSERT_EQ(kTfLiteOk, interpreter2->AllocateTensors());
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate2(TfLiteXNNPackDelegateCreate(&delegate_options),
                TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(kTfLiteOk, interpreter2->ModifyGraphWithDelegate(delegate2.get()));
  ASSERT_EQ(kTfLiteOk, interpreter2->Invoke());
}

// Dummy class to use with parameterized test.
class WeightsCacheTest : public testing::TestWithParam<size_t> {};

TEST_P(WeightsCacheTest, SoftFinalizationMultithreaded) {
  std::vector<char> buffer = Conv2DTester().CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.weights_cache = weights_cache.get();

  // Create the first interpreter and finalize it.
  std::unique_ptr<Interpreter> initial_interpreter;
  ASSERT_EQ(kTfLiteOk,
            InterpreterBuilder(model, resolver)(&initial_interpreter));
  ASSERT_EQ(kTfLiteOk, initial_interpreter->AllocateTensors());
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      initial_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(kTfLiteOk, initial_interpreter->ModifyGraphWithDelegate(
                           initial_delegate.get()));

  ASSERT_TRUE(
      TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(weights_cache.get()));

  ASSERT_EQ(kTfLiteOk, initial_interpreter->Invoke());

  // Create multiple interpreters afterwards.
  const size_t num_threads = GetParam();
  if (num_threads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(std::thread([&] {
      std::unique_ptr<Interpreter> interpreter;
      ASSERT_EQ(kTfLiteOk, InterpreterBuilder(model, resolver)(&interpreter));
      ASSERT_EQ(kTfLiteOk, interpreter->AllocateTensors());

      std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
          delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                   TfLiteXNNPackDelegateDelete);

      ASSERT_EQ(kTfLiteOk,
                interpreter->ModifyGraphWithDelegate(delegate.get()));
      ASSERT_EQ(kTfLiteOk, interpreter->Invoke());
    }));
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}

INSTANTIATE_TEST_SUITE_P(WeightsCacheTest, WeightsCacheTest,
                         testing::Values(2, 4),
                         testing::PrintToStringParamName());

}  // namespace xnnpack
}  // namespace tflite
