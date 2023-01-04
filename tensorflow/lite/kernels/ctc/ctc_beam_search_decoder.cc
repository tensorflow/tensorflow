/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/ctc/ctc_beam_search.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace custom {
namespace ctc_beam_search_decoder {

constexpr int kInputsTensor = 0;
constexpr int kSequenceLengthTensor = 1;

typedef struct {
  int beam_width;
  int top_paths;
  bool merge_repeated;
} CTCBeamSearchDecoderParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_CHECK(buffer != nullptr);
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  CTCBeamSearchDecoderParams* option = new CTCBeamSearchDecoderParams;
  option->beam_width = m["beam_width"].AsInt32();
  option->top_paths = m["top_paths"].AsInt32();
  option->merge_repeated = m["merge_repeated"].AsBool();

  return option;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<CTCBeamSearchDecoderParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const CTCBeamSearchDecoderParams* option =
      reinterpret_cast<CTCBeamSearchDecoderParams*>(node->user_data);
  const int top_paths = option->top_paths;
  TF_LITE_ENSURE(context, option->beam_width >= top_paths);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  // The outputs should be top_paths * 3 + 1.
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3 * top_paths + 1);

  const TfLiteTensor* inputs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &inputs));
  TF_LITE_ENSURE_EQ(context, NumDimensions(inputs), 3);
  // TensorFlow only supports float.
  TF_LITE_ENSURE_EQ(context, inputs->type, kTfLiteFloat32);
  const int batch_size = SizeOfDimension(inputs, 1);

  const TfLiteTensor* sequence_length;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSequenceLengthTensor,
                                          &sequence_length));
  TF_LITE_ENSURE_EQ(context, NumDimensions(sequence_length), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(sequence_length), batch_size);
  // TensorFlow only supports int32.
  TF_LITE_ENSURE_EQ(context, sequence_length->type, kTfLiteInt32);

  // Resize decoded outputs.
  // Do not resize indices & values cause we don't know the values yet.
  for (int i = 0; i < top_paths; ++i) {
    TfLiteTensor* indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &indices));
    SetTensorToDynamic(indices);
    TfLiteTensor* values;
    TF_LITE_ENSURE_OK(context,
                      GetOutputSafe(context, node, i + top_paths, &values));
    SetTensorToDynamic(values);
    TfLiteTensor* output_shape;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i + 2 * top_paths,
                                             &output_shape));
    SetTensorToDynamic(output_shape);
  }

  // Resize log probability outputs.
  TfLiteTensor* log_probability_output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, top_paths * 3,
                                           &log_probability_output));
  TfLiteIntArray* log_probability_output_shape_array = TfLiteIntArrayCreate(2);
  log_probability_output_shape_array->data[0] = batch_size;
  log_probability_output_shape_array->data[1] = top_paths;
  return context->ResizeTensor(context, log_probability_output,
                               log_probability_output_shape_array);
}

TfLiteStatus Resize(TfLiteContext* context,
                    std::initializer_list<int32_t> output_shape,
                    TfLiteTensor* output) {
  const int dimensions = output_shape.size();
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(dimensions);
  int i = 0;
  for (const int v : output_shape) {
    output_shape_array->data[i++] = v;
  }
  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus StoreAllDecodedSequences(
    TfLiteContext* context,
    const std::vector<std::vector<std::vector<int>>>& sequences,
    TfLiteNode* node, int top_paths) {
  const int32_t batch_size = sequences.size();
  std::vector<int32_t> num_entries(top_paths, 0);

  // Calculate num_entries per path
  for (const auto& batch_s : sequences) {
    TF_LITE_ENSURE_EQ(context, batch_s.size(), top_paths);
    for (int p = 0; p < top_paths; ++p) {
      num_entries[p] += batch_s[p].size();
    }
  }

  for (int p = 0; p < top_paths; ++p) {
    const int32_t p_num = num_entries[p];

    // Resize the decoded outputs.
    TfLiteTensor* indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, p, &indices));
    TF_LITE_ENSURE_OK(context, Resize(context, {p_num, 2}, indices));

    TfLiteTensor* values;
    TF_LITE_ENSURE_OK(context,
                      GetOutputSafe(context, node, p + top_paths, &values));
    TF_LITE_ENSURE_OK(context, Resize(context, {p_num}, values));

    TfLiteTensor* decoded_shape;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, p + 2 * top_paths,
                                             &decoded_shape));
    TF_LITE_ENSURE_OK(context, Resize(context, {2}, decoded_shape));

    int32_t max_decoded = 0;
    int32_t offset = 0;

    int32_t* indices_data = GetTensorData<int32_t>(indices);
    int32_t* values_data = GetTensorData<int32_t>(values);
    int32_t* decoded_shape_data = GetTensorData<int32_t>(decoded_shape);
    for (int b = 0; b < batch_size; ++b) {
      auto& p_batch = sequences[b][p];
      int32_t num_decoded = p_batch.size();
      max_decoded = std::max(max_decoded, num_decoded);

      std::copy_n(p_batch.begin(), num_decoded, values_data + offset);
      for (int32_t t = 0; t < num_decoded; ++t, ++offset) {
        indices_data[offset * 2] = b;
        indices_data[offset * 2 + 1] = t;
      }
    }

    decoded_shape_data[0] = batch_size;
    decoded_shape_data[1] = max_decoded;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* inputs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &inputs));
  const TfLiteTensor* sequence_length;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSequenceLengthTensor,
                                          &sequence_length));
  const CTCBeamSearchDecoderParams* option =
      reinterpret_cast<CTCBeamSearchDecoderParams*>(node->user_data);

  const int max_time = SizeOfDimension(inputs, 0);
  const int batch_size = SizeOfDimension(inputs, 1);
  const int num_classes = SizeOfDimension(inputs, 2);

  const int beam_width = option->beam_width;
  const int top_paths = option->top_paths;
  const bool merge_repeated = option->merge_repeated;

  // Validate sequence length is less or equal than max time.
  for (int i = 0; i < batch_size; ++i) {
    TF_LITE_ENSURE(context,
                   max_time >= GetTensorData<int32_t>(sequence_length)[i]);
  }

  // The following logic is implemented like
  // tensorflow/core/kernels/ctc_decoder_ops.cc
  std::vector<optimized_ops::TTypes<float>::UnalignedConstMatrix> input_list_t;

  for (std::size_t t = 0; t < max_time; ++t) {
    input_list_t.emplace_back(
        GetTensorData<float>(inputs) + t * batch_size * num_classes, batch_size,
        num_classes);
  }

  ::tflite::custom::ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer beam_scorer;
  ::tflite::custom::ctc::CTCBeamSearchDecoder<> beam_search(
      num_classes, beam_width, &beam_scorer, 1 /* batch_size */,
      merge_repeated);

  // Allocate temporary memory for holding chip operation data.
  float* input_chip_t_data =
      static_cast<float*>(malloc(num_classes * sizeof(float)));
  Eigen::array<Eigen::DenseIndex, 1> dims;
  dims[0] = num_classes;
  optimized_ops::TTypes<float>::Flat input_chip_t(input_chip_t_data, dims);

  std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
  std::vector<float> log_probs;

  TfLiteTensor* log_probabilities;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, 3 * top_paths, &log_probabilities));
  float* log_probabilities_output = GetTensorData<float>(log_probabilities);

  // Assumption: the blank index is num_classes - 1
  for (int b = 0; b < batch_size; ++b) {
    auto& best_paths_b = best_paths[b];
    best_paths_b.resize(top_paths);
    for (int t = 0; t < GetTensorData<int32_t>(sequence_length)[b]; ++t) {
      input_chip_t = input_list_t[t].chip(b, 0);
      auto input_bi =
          Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
      beam_search.Step(input_bi);
    }
    TF_LITE_ENSURE(context, beam_search.TopPaths(top_paths, &best_paths_b,
                                                 &log_probs, merge_repeated));
    beam_search.Reset();

    // Fill in log_probabilities output.
    for (int bp = 0; bp < top_paths; ++bp) {
      log_probabilities_output[b * top_paths + bp] = log_probs[bp];
    }
  }

  free(input_chip_t_data);
  return StoreAllDecodedSequences(context, best_paths, node, top_paths);
}

}  // namespace ctc_beam_search_decoder

TfLiteRegistration* Register_CTC_BEAM_SEARCH_DECODER() {
  static TfLiteRegistration r = {
      ctc_beam_search_decoder::Init, ctc_beam_search_decoder::Free,
      ctc_beam_search_decoder::Prepare, ctc_beam_search_decoder::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
