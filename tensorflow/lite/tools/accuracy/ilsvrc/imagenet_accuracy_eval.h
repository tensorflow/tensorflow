/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_IMAGENET_ACCURACY_EVAL_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_IMAGENET_ACCURACY_EVAL_H_

#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <ostream>
#include <string>

#include "tensorflow/lite/tools/accuracy/csv_writer.h"
#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"

namespace tensorflow {
namespace metrics {

// Writes topK accuracy results to a CSV file & logs progress to standard output
// with `kLogDelayUs` microseconds.
class ResultsWriter : public ImagenetModelEvaluator::Observer {
 public:
  ResultsWriter(int top_k, const std::string& output_file_path);

  bool IsValid() const { return writer_ != nullptr; }

  void OnEvaluationStart(const std::unordered_map<uint64_t, int>&
                             shard_id_image_count_map) override;

  void OnSingleImageEvaluationComplete(
      uint64_t shard_id,
      const tflite::evaluation::TopkAccuracyEvalMetrics& metrics,
      const std::string& image) override;

  tflite::evaluation::TopkAccuracyEvalMetrics AggregatedMetrics();

  void OutputEvalMetriccProto(const std::string& proto_output_file);

 private:
  void AggregateAccuraciesAndNumImages(std::vector<double>* accuracies,
                                       int* num_done_images);

  int top_k_ = 0;
  std::unordered_map<uint64_t, tflite::evaluation::TopkAccuracyEvalMetrics>
      shard_id_accuracy_metrics_map_;
  std::unordered_map<uint64_t, int> shard_id_done_image_count_map_;

  // TODO(b/146988222): Refactor CSVWriter to take the memory ownership of
  // 'output_stream_'.
  std::unique_ptr<std::ofstream> output_stream_;
  std::unique_ptr<CSVWriter> writer_;

  // For logging to stdout.
  uint64_t last_logged_time_us_ = 0;
  int total_num_images_ = 0;
  static constexpr int kLogDelayUs = 500 * 1000;

  std::mutex mu_;
};

// Create an evaluator by parsing command line arguments.
// Note argc and argv will be updated accordingly as matching arguments will
// be removed argv.
std::unique_ptr<ImagenetModelEvaluator> CreateImagenetModelEvaluator(
    int* argc, char* argv[],
    int num_threads = 1  // the number of threads used for evaluation.
);

std::unique_ptr<ResultsWriter> CreateImagenetEvalResultsWriter(
    int top_k, const std::string& output_file_path);

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_IMAGENET_ACCURACY_EVAL_H_
