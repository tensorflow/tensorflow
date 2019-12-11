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

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <ostream>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/accuracy/csv_writer.h"
#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tensorflow {
namespace metrics {

namespace {

using ::tflite::evaluation::TopkAccuracyEvalMetrics;

constexpr char kNumThreadsFlag[] = "num_threads";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kProtoOutputFilePathFlag[] = "proto_output_file_path";

// TODO(b/130823599): Move to tools/evaluation/stages/topk_accuracy_eval_stage.
// Computes total number of images processed & aggregates Top-K accuracies
// into 'accuracies'.
void AggregateAccuraciesAndNumImages(
    int k,
    const std::unordered_map<uint64_t, TopkAccuracyEvalMetrics>&
        shard_id_accuracy_metrics_map,
    const std::unordered_map<uint64_t, int>& shard_id_done_image_count_map,
    std::vector<double>* accuracies, int* num_done_images) {
  // Total images done.
  *num_done_images = 0;
  for (auto iter = shard_id_done_image_count_map.begin();
       iter != shard_id_done_image_count_map.end(); ++iter) {
    *num_done_images += iter->second;
  }

  // Aggregated accuracies.
  for (int i = 0; i < k; ++i) {
    double correct_inferences = 0;
    double total_inferences = 0;
    for (auto iter = shard_id_done_image_count_map.begin();
         iter != shard_id_done_image_count_map.end(); ++iter) {
      const uint64_t shard_id = iter->first;
      const TopkAccuracyEvalMetrics& accuracy_metrics =
          shard_id_accuracy_metrics_map.at(shard_id);
      const int num_images = iter->second;
      correct_inferences += num_images * accuracy_metrics.topk_accuracies(i);
      total_inferences += num_images;
    }
    // Convert to percentage.
    accuracies->push_back(100.0 * correct_inferences / total_inferences);
  }
}

}  // namespace

// Writes results to a CSV file & logs progress to standard output with
// `kLogDelayUs` microseconds.
class ResultsWriter : public ImagenetModelEvaluator::Observer {
 public:
  explicit ResultsWriter(int k, std::unique_ptr<CSVWriter> writer)
      : k_(k), writer_(std::move(writer)) {}

  void OnEvaluationStart(const std::unordered_map<uint64_t, int>&
                             shard_id_image_count_map) override;

  void OnSingleImageEvaluationComplete(uint64_t shard_id,
                                       const TopkAccuracyEvalMetrics& metrics,
                                       const string& image) override;

  TopkAccuracyEvalMetrics AggregatedMetrics();

 private:
  // For writing to CSV.
  int k_;
  std::unordered_map<uint64_t, TopkAccuracyEvalMetrics>
      shard_id_accuracy_metrics_map_;
  std::unordered_map<uint64_t, int> shard_id_done_image_count_map_;
  std::unique_ptr<CSVWriter> writer_;

  // For logging to stdout.
  uint64_t last_logged_time_us_ = 0;
  int total_num_images_;
  static constexpr int kLogDelayUs = 500 * 1000;

  std::mutex mu_;
};

void ResultsWriter::OnEvaluationStart(
    const std::unordered_map<uint64_t, int>& shard_id_image_count_map) {
  int total_num_images = 0;
  for (const auto& kv : shard_id_image_count_map) {
    total_num_images += kv.second;
  }
  LOG(ERROR) << "Starting model evaluation: " << total_num_images;
  std::lock_guard<std::mutex> lock(mu_);
  total_num_images_ = total_num_images;
}

void ResultsWriter::OnSingleImageEvaluationComplete(
    uint64_t shard_id,
    const tflite::evaluation::TopkAccuracyEvalMetrics& metrics,
    const string& image) {
  std::lock_guard<std::mutex> lock(mu_);
  shard_id_done_image_count_map_[shard_id] += 1;
  shard_id_accuracy_metrics_map_[shard_id] = metrics;

  int num_evaluated;
  std::vector<double> total_accuracies;
  AggregateAccuraciesAndNumImages(k_, shard_id_accuracy_metrics_map_,
                                  shard_id_done_image_count_map_,
                                  &total_accuracies, &num_evaluated);
  if (writer_->WriteRow(total_accuracies) != kTfLiteOk) {
    LOG(ERROR) << "Could not write to file";
    return;
  }
  writer_->Flush();

  auto now_us = tflite::profiling::time::NowMicros();
  if ((now_us - last_logged_time_us_) >= kLogDelayUs) {
    last_logged_time_us_ = now_us;
    double current_percent = num_evaluated * 100.0 / total_num_images_;
    LOG(ERROR) << "Evaluated " << num_evaluated << "/" << total_num_images_
               << " images, " << std::setprecision(2) << std::fixed
               << current_percent << "%";
  }
}

TopkAccuracyEvalMetrics ResultsWriter::AggregatedMetrics() {
  std::lock_guard<std::mutex> lock(mu_);
  int num_evaluated;
  std::vector<double> total_accuracies;
  AggregateAccuraciesAndNumImages(k_, shard_id_accuracy_metrics_map_,
                                  shard_id_done_image_count_map_,
                                  &total_accuracies, &num_evaluated);
  TopkAccuracyEvalMetrics aggregated_metrics;
  for (auto accuracy : total_accuracies) {
    aggregated_metrics.add_topk_accuracies(accuracy);
  }
  return aggregated_metrics;
}

int Main(int argc, char* argv[]) {
  std::string output_file_path, proto_output_file_path;
  int num_threads = 4;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kNumThreadsFlag, &num_threads,
                               "Number of threads."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path,
                               "Path to output file."),
      tflite::Flag::CreateFlag(kProtoOutputFilePathFlag,
                               &proto_output_file_path,
                               "Path to proto output file."),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  std::unique_ptr<ImagenetModelEvaluator> evaluator;
  if (output_file_path.empty()) {
    LOG(ERROR) << "Invalid output file path.";
    return EXIT_FAILURE;
  }

  if (num_threads <= 0) {
    LOG(ERROR) << "Invalid number of threads.";
    return EXIT_FAILURE;
  }

  if (ImagenetModelEvaluator::Create(argc, argv, num_threads, &evaluator) !=
      kTfLiteOk)
    return EXIT_FAILURE;

  std::ofstream output_stream(output_file_path, std::ios::out);
  if (!output_stream) {
    LOG(ERROR) << "Unable to open output file path: '" << output_file_path
               << "'";
  }

  output_stream << std::setprecision(3) << std::fixed;
  std::vector<string> columns;
  columns.reserve(evaluator->params().num_ranks);
  for (int i = 0; i < evaluator->params().num_ranks; i++) {
    std::string column_name = "Top ";
    column_name = column_name + std::to_string(i + 1);
    columns.push_back(column_name);
  }

  ResultsWriter results_writer(
      evaluator->params().num_ranks,
      absl::make_unique<CSVWriter>(columns, &output_stream));
  evaluator->AddObserver(&results_writer);
  LOG(ERROR) << "Starting evaluation with: " << num_threads << " threads.";
  if (evaluator->EvaluateModel() != kTfLiteOk) {
    LOG(ERROR) << "Failed to evaluate the model!";
    return EXIT_FAILURE;
  }

  if (!proto_output_file_path.empty()) {
    std::ofstream proto_out_file(proto_output_file_path,
                                 std::ios::out | std::ios::binary);
    TopkAccuracyEvalMetrics metrics = results_writer.AggregatedMetrics();
    proto_out_file << metrics.SerializeAsString();
    proto_out_file.close();
  }

  return EXIT_SUCCESS;
}

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::metrics::Main(argc, argv);
}
