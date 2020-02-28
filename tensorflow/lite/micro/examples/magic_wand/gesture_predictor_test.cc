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

#include "tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h"

#include "tensorflow/lite/micro/examples/magic_wand/constants.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SuccessfulPrediction) {
  // Use the threshold from the 0th gesture.
  float probabilities[kGestureCount] = {kDetectionThreshold, 0.0, 0.0, 0.0};
  int prediction;
  // Loop just too few times to trigger a prediction.
  for (int i = 0; i < kPredictionHistoryLength - 1; i++) {
    prediction = PredictGesture(probabilities);
    TF_LITE_MICRO_EXPECT_EQ(prediction, kNoGesture);
  }
  // Call once more, triggering a prediction
  // for category 0.
  prediction = PredictGesture(probabilities);
  TF_LITE_MICRO_EXPECT_EQ(prediction, 0);
}

TF_LITE_MICRO_TEST(FailPartWayThere) {
  // Use the threshold from the 0th gesture.
  float probabilities[kGestureCount] = {kDetectionThreshold, 0.0, 0.0, 0.0};
  int prediction;
  // Loop just too few times to trigger a prediction.
  for (int i = 0; i <= kPredictionHistoryLength - 1; i++) {
    prediction = PredictGesture(probabilities);
    TF_LITE_MICRO_EXPECT_EQ(prediction, kNoGesture);
  }
  // Call with a different prediction, triggering a failure.
  probabilities[0] = 0.0;
  probabilities[2] = 1.0;
  prediction = PredictGesture(probabilities);
  TF_LITE_MICRO_EXPECT_EQ(prediction, kNoGesture);
}

TF_LITE_MICRO_TEST(InsufficientProbability) {
  // Just below the detection threshold.
  float probabilities[kGestureCount] = {kDetectionThreshold - 0.1f, 0.0, 0.0,
                                        0.0};
  int prediction;
  // Loop the exact right number of times
  for (int i = 0; i <= kPredictionHistoryLength; i++) {
    prediction = PredictGesture(probabilities);
    TF_LITE_MICRO_EXPECT_EQ(prediction, kNoGesture);
  }
}

TF_LITE_MICRO_TESTS_END
