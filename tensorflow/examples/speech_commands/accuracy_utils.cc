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

#include "tensorflow/examples/speech_commands/accuracy_utils.h"

#include <fstream>
#include <iomanip>
#include <unordered_set>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

Status ReadGroundTruthFile(const string& file_name,
                           std::vector<std::pair<string, int64_t>>* result) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Ground truth file '", file_name,
                                        "' not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    std::vector<string> pieces = tensorflow::str_util::Split(line, ',');
    if (pieces.size() != 2) {
      continue;
    }
    float timestamp;
    if (!tensorflow::strings::safe_strtof(pieces[1].c_str(), &timestamp)) {
      return tensorflow::errors::InvalidArgument(
          "Wrong number format at line: ", line);
    }
    string label = pieces[0];
    auto timestamp_int64 = static_cast<int64_t>(timestamp);
    result->push_back({label, timestamp_int64});
  }
  std::sort(result->begin(), result->end(),
            [](const std::pair<string, int64>& left,
               const std::pair<string, int64>& right) {
              return left.second < right.second;
            });
  return Status::OK();
}

void CalculateAccuracyStats(
    const std::vector<std::pair<string, int64_t>>& ground_truth_list,
    const std::vector<std::pair<string, int64_t>>& found_words,
    int64_t up_to_time_ms, int64_t time_tolerance_ms,
    StreamingAccuracyStats* stats) {
  int64_t latest_possible_time;
  if (up_to_time_ms == -1) {
    latest_possible_time = std::numeric_limits<int64_t>::max();
  } else {
    latest_possible_time = up_to_time_ms + time_tolerance_ms;
  }
  stats->how_many_ground_truth_words = 0;
  for (const std::pair<string, int64_t>& ground_truth : ground_truth_list) {
    const int64_t ground_truth_time = ground_truth.second;
    if (ground_truth_time > latest_possible_time) {
      break;
    }
    ++stats->how_many_ground_truth_words;
  }

  stats->how_many_false_positives = 0;
  stats->how_many_correct_words = 0;
  stats->how_many_wrong_words = 0;
  std::unordered_set<int64_t> has_ground_truth_been_matched;
  for (const std::pair<string, int64_t>& found_word : found_words) {
    const string& found_label = found_word.first;
    const int64_t found_time = found_word.second;
    const int64_t earliest_time = found_time - time_tolerance_ms;
    const int64_t latest_time = found_time + time_tolerance_ms;
    bool has_match_been_found = false;
    for (const std::pair<string, int64_t>& ground_truth : ground_truth_list) {
      const int64_t ground_truth_time = ground_truth.second;
      if ((ground_truth_time > latest_time) ||
          (ground_truth_time > latest_possible_time)) {
        break;
      }
      if (ground_truth_time < earliest_time) {
        continue;
      }
      const string& ground_truth_label = ground_truth.first;
      if ((ground_truth_label == found_label) &&
          (has_ground_truth_been_matched.count(ground_truth_time) == 0)) {
        ++stats->how_many_correct_words;
      } else {
        ++stats->how_many_wrong_words;
      }
      has_ground_truth_been_matched.insert(ground_truth_time);
      has_match_been_found = true;
      break;
    }
    if (!has_match_been_found) {
      ++stats->how_many_false_positives;
    }
  }
  stats->how_many_ground_truth_matched = has_ground_truth_been_matched.size();
}

void PrintAccuracyStats(const StreamingAccuracyStats& stats) {
  if (stats.how_many_ground_truth_words == 0) {
    LOG(INFO) << "No ground truth yet, " << stats.how_many_false_positives
              << " false positives";
  } else {
    float any_match_percentage =
        (stats.how_many_ground_truth_matched * 100.0f) /
        stats.how_many_ground_truth_words;
    float correct_match_percentage = (stats.how_many_correct_words * 100.0f) /
                                     stats.how_many_ground_truth_words;
    float wrong_match_percentage = (stats.how_many_wrong_words * 100.0f) /
                                   stats.how_many_ground_truth_words;
    float false_positive_percentage =
        (stats.how_many_false_positives * 100.0f) /
        stats.how_many_ground_truth_words;

    LOG(INFO) << std::setprecision(1) << std::fixed << any_match_percentage
              << "% matched, " << correct_match_percentage << "% correctly, "
              << wrong_match_percentage << "% wrongly, "
              << false_positive_percentage << "% false positives ";
  }
}

}  // namespace tensorflow
