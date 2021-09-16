/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace xnnpack {

TEST(Delegate, CreateWithoutParams) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithDefaultParams) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithNumThreadsParam) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, GetThreadPool) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(xnnpack_delegate.get()));
  ASSERT_TRUE(threadpool);
  ASSERT_EQ(2, pthreadpool_get_threads_count(threadpool));
}

TEST(Delegate, CreateWithStaticUnpackNodes) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

  interpreter->AddTensors(2);
  interpreter->SetInputs({0});
  interpreter->SetOutputs({1});

  TfLiteQuantization quant = {
      .type = kTfLiteAffineQuantization,
      .params = malloc(sizeof(TfLiteAffineQuantization))};
  TfLiteAffineQuantization* aq =
      reinterpret_cast<TfLiteAffineQuantization*>(quant.params);
  aq->scale = TfLiteFloatArrayCreate(1);
  aq->scale->data[0] = 0.5;
  aq->zero_point = TfLiteIntArrayCreate(1);
  aq->zero_point->data[0] = 127;
  aq->quantized_dimension = 0;

  uint16_t src;
  float target;

  TfLiteQuantization no_quant = {kTfLiteNoQuantization, 0};
  interpreter->SetTensorParametersReadOnly(0, kTfLiteFloat16, "src", {1}, quant,
                                           (char*)&src, sizeof(src));
  interpreter->SetTensorParametersReadOnly(1, kTfLiteFloat32, "target", {1},
                                           no_quant, (char*)&target,
                                           sizeof(target));

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* deq_op =
      resolver.FindOp(tflite::BuiltinOperator::BuiltinOperator_DEQUANTIZE, 1);
  interpreter->AddNodeWithParameters({0}, {1}, nullptr, 0, 0, deq_op);

  ASSERT_EQ(kTfLiteOk,
            interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()));
}

}  // namespace xnnpack
}  // namespace tflite
