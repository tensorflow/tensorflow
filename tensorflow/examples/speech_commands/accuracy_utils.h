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

#ifndef TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_ACCURACY_UTILS_H_
#define TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_ACCURACY_UTILS_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

struct StreamingAccuracyStats {
  StreamingAccuracyStats()
      : how_many_ground_truth_words(0),
        how_many_ground_truth_matched(0),
        how_many_false_positives(0),
        how_many_correct_words(0),
        how_many_wrong_words(0) {}
  int32 how_many_ground_truth_words;
  int32 how_many_ground_truth_matched;
  int32 how_many_false_positives;
  int32 how_many_correct_words;
  int32 how_many_wrong_words;
};

// Takes a file name, and loads a list of expected word labels and times from
// it, as comma-separated variables.
Status ReadGroundTruthFile(const string& file_name,
                           std::vector<std::pair<string, int64_t>>* result);

// Given ground truth labels and corresponding predictions found by a model,
// figure out how many were correct. Takes a time limit, so that only
// predictions up to a point in time are considered, in case we're evaluating
// accuracy when the model has only been run on part of the stream.
void CalculateAccuracyStats(
    const std::vector<std::pair<string, int64_t>>& ground_truth_list,
    const std::vector<std::pair<string, int64_t>>& found_words,
    int64_t up_to_time_ms, int64_t time_tolerance_ms,
    StreamingAccuracyStats* stats);

// Writes a human-readable description of the statistics to stdout.
void PrintAccuracyStats(const StreamingAccuracyStats& stats);

}  // namespace tensorflow

#endif  // TENSORFLOW_EXAMPLES_SPEECH_COMMANDS_ACCURACY_UTILS_H_
