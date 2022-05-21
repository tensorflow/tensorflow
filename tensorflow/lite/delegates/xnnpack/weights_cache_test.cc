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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/conv_2d_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace xnnpack {

TEST(XNNPACK_WEIGHTS_CACHE, invoke_before_finalization) {
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

  // Invoking before finalization fails.
  ASSERT_NE(kTfLiteOk, interpreter1->Invoke());
}

TEST(XNNPACK_WEIGHTS_CACHE, hard_finalization) {
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

TEST(XNNPACK_WEIGHTS_CACHE, soft_finalization) {
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

}  // namespace xnnpack
}  // namespace tflite
