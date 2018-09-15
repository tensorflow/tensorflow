/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/models/smartreply/predictor.h"

#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/string_util.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

namespace tflite {
namespace custom {
namespace smartreply {

// Split sentence into segments (using punctuation).
std::vector<std::string> SplitSentence(const std::string& input) {
  string result(input);

  RE2::GlobalReplace(&result, "([?.!,])+", " \\1");
  RE2::GlobalReplace(&result, "([?.!,])+\\s+", "\\1\t");
  RE2::GlobalReplace(&result, "[ ]+", " ");
  RE2::GlobalReplace(&result, "\t+$", "");

  return absl::StrSplit(result, '\t');
}

// Predict with TfLite model.
void ExecuteTfLite(const std::string& sentence,
                   ::tflite::Interpreter* interpreter,
                   std::map<std::string, float>* response_map) {
  {
    TfLiteTensor* input = interpreter->tensor(interpreter->inputs()[0]);
    tflite::DynamicBuffer buf;
    buf.AddString(sentence.data(), sentence.length());
    buf.WriteToTensor(input);
    interpreter->AllocateTensors();

    interpreter->Invoke();

    TfLiteTensor* messages = interpreter->tensor(interpreter->outputs()[0]);
    TfLiteTensor* confidence = interpreter->tensor(interpreter->outputs()[1]);

    for (int i = 0; i < confidence->dims->data[0]; i++) {
      float weight = confidence->data.f[i];
      auto response_text = tflite::GetString(messages, i);
      if (response_text.len > 0) {
        (*response_map)[string(response_text.str, response_text.len)] += weight;
      }
    }
  }
}

void GetSegmentPredictions(
    const std::vector<std::string>& input,
    const ::tflite::FlatBufferModel& model, const SmartReplyConfig& config,
    std::vector<PredictorResponse>* predictor_responses) {
  // Initialize interpreter
  std::unique_ptr<::tflite::Interpreter> interpreter;
  ::tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
  ::tflite::InterpreterBuilder(model, resolver)(&interpreter);

  if (!model.initialized()) {
    fprintf(stderr, "Failed to mmap model \n");
    return;
  }

  // Execute Tflite Model
  std::map<std::string, float> response_map;
  std::vector<std::string> sentences;
  for (const std::string& str : input) {
    std::vector<std::string> splitted_str = SplitSentence(str);
    sentences.insert(sentences.end(), splitted_str.begin(), splitted_str.end());
  }
  for (const auto& sentence : sentences) {
    ExecuteTfLite(sentence, interpreter.get(), &response_map);
  }

  // Generate the result.
  for (const auto& iter : response_map) {
    PredictorResponse prediction(iter.first, iter.second);
    predictor_responses->emplace_back(prediction);
  }
  std::sort(predictor_responses->begin(), predictor_responses->end(),
            [](const PredictorResponse& a, const PredictorResponse& b) {
              return a.GetScore() > b.GetScore();
            });

  // Add backoff response.
  for (const auto& backoff : config.backoff_responses) {
    if (predictor_responses->size() >= config.num_response) {
      break;
    }
    predictor_responses->emplace_back(backoff, config.backoff_confidence);
  }
}

}  // namespace smartreply
}  // namespace custom
}  // namespace tflite
