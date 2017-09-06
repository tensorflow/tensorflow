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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_RECOGNIZE_COMMANDS_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_RECOGNIZE_COMMANDS_H_

#include <deque>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// This class is designed to apply a very primitive decoding model on top of the
// instantaneous results from running an audio recognition model on a single
// window of samples. It applies smoothing over time so that noisy individual
// label scores are averaged, increasing the confidence that apparent matches
// are real.
// To use it, you should create a class object with the configuration you
// want, and then feed results from running a TensorFlow model into the
// processing method. The timestamp for each subsequent call should be
// increasing from the previous, since the class is designed to process a stream
// of data over time.
class RecognizeCommands {
 public:
  // labels should be a list of the strings associated with each one-hot score.
  // The window duration controls the smoothing. Longer durations will give a
  // higher confidence that the results are correct, but may miss some commands.
  // The detection threshold has a similar effect, with high values increasing
  // the precision at the cost of recall. The minimum count controls how many
  // results need to be in the averaging window before it's seen as a reliable
  // average. This prevents erroneous results when the averaging window is
  // initially being populated for example. The suppression argument disables
  // further recognitions for a set time after one has been triggered, which can
  // help reduce spurious recognitions.
  explicit RecognizeCommands(const std::vector<string>& labels,
                             int32 average_window_duration_ms = 1000,
                             float detection_threshold = 0.2,
                             int32 suppression_ms = 500,
                             int32 minimum_count = 3);

  // Call this with the results of running a model on sample data.
  Status ProcessLatestResults(const Tensor& latest_results,
                              const int64 current_time_ms,
                              string* found_command, float* score,
                              bool* is_new_command);

 private:
  // Configuration
  std::vector<string> labels_;
  int32 average_window_duration_ms_;
  float detection_threshold_;
  int32 suppression_ms_;
  int32 minimum_count_;

  // Working variables
  std::deque<std::pair<int64, Tensor>> previous_results_;
  string previous_top_label_;
  int64 labels_count_;
  int64 previous_top_label_time_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_RECOGNIZE_COMMANDS_H_
