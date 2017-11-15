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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_SMARTREPLY_PREDICTOR_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_SMARTREPLY_PREDICTOR_H_

#include <string>
#include <vector>

#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace custom {
namespace smartreply {

const int kDefaultNumResponse = 10;
const float kDefaultBackoffConfidence = 1e-4;

class PredictorResponse;
struct SmartReplyConfig;

// With a given string as input, predict the response with a Tflite model.
// When config.backoff_response is not empty, predictor_responses will be filled
// with messagees from backoff response.
void GetSegmentPredictions(const std::vector<string>& input,
                           const ::tflite::FlatBufferModel& model,
                           const SmartReplyConfig& config,
                           std::vector<PredictorResponse>* predictor_responses);

// Data object used to hold a single predictor response.
// It includes messages, and confidence.
class PredictorResponse {
 public:
  PredictorResponse(const string& response_text, float score) {
    response_text_ = response_text;
    prediction_score_ = score;
  }

  // Accessor methods.
  const string& GetText() const { return response_text_; }
  float GetScore() const { return prediction_score_; }

 private:
  string response_text_ = "";
  float prediction_score_ = 0.0;
};

// Configurations for SmartReply.
struct SmartReplyConfig {
  // Maximum responses to return.
  int num_response;
  // Default confidence for backoff responses.
  float backoff_confidence;
  // Backoff responses are used when predicted responses cannot fulfill the
  // list.
  const std::vector<string>& backoff_responses;

  SmartReplyConfig(std::vector<string> backoff_responses)
      : num_response(kDefaultNumResponse),
        backoff_confidence(kDefaultBackoffConfidence),
        backoff_responses(backoff_responses) {}
};

}  // namespace smartreply
}  // namespace custom
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_MODELS_SMARTREPLY_PREDICTOR_H_
