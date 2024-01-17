/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/ml_adjacent/tflite/extern_call.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/ml_adjacent/algo/crop.h"
#include "tensorflow/lite/experimental/ml_adjacent/algo/resize.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/experimental/ml_adjacent/tflite/tfl_tensor_ref.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace extern_call {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::MutableTflTensorRef;
using ::ml_adj::data::TflTensorRef;

// UniquePtr wrapper around vectors that hold the inputs/outputs to
// library's `Algo`s.
template <typename PackType>
struct PackDeleter {
  void operator()(PackType* pack) {
    if (pack == nullptr) return;
    for (auto* d : *pack) {
      if (d == nullptr) continue;
      delete d;
    }
    delete pack;
  }
};
template <typename PackType>
using UniquePack = std::unique_ptr<PackType, PackDeleter<PackType>>;

constexpr uint8_t kNumFuncs = 2;
static const Algo* const kReg[kNumFuncs] = {ml_adj::crop::Impl_CenterCrop(),
                                            ml_adj::resize::Impl_Resize()};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    SetTensorToDynamic(output);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  UniquePack<InputPack> lib_inputs(new InputPack());
  for (int i = 0; i < NumInputs(node); ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));

    lib_inputs->push_back(new TflTensorRef(input));
  }

  UniquePack<OutputPack> lib_outputs(new OutputPack());
  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));

    lib_outputs->push_back(new MutableTflTensorRef(output, context));
  }

  TF_LITE_ENSURE_EQ(context, node->custom_initial_data_size,
                    sizeof(ExternCallOptions));
  const auto* const options =
      reinterpret_cast<const ExternCallOptions*>(node->custom_initial_data);
  TF_LITE_ENSURE(context,
                 options->func_id >= 0 && options->func_id < kNumFuncs);

  const Algo* const algo = kReg[options->func_id];

  algo->process(*lib_inputs, *lib_outputs);

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_EXTERN_CALL() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace extern_call
}  // namespace tflite
