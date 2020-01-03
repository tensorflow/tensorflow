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

#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_accuracy_eval.h"

#include <cstdlib>
#include <iomanip>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/accuracy/csv_writer.h"
#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tensorflow {
namespace metrics {

using ::tflite::evaluation::TopkAccuracyEvalMetrics;

ResultsWriter::ResultsWriter(int top_k, const std::string& output_file_path)
    : top_k_(top_k) {
  if (output_file_path.empty()) {
    LOG(ERROR) << "Empty output file path.";
    return;
  }

  output_stream_.reset(new std::ofstream(output_file_path, std::ios::out));
  if (!output_stream_) {
    LOG(ERROR) << "Unable to open output file path: '" << output_file_path
               << "'";
  }

  (*output_stream_) << std::setprecision(3) << std::fixed;
  std::vector<string> columns;
  columns.reserve(top_k);
  for (int i = 0; i < top_k; i++) {
    std::string column_name = "Top ";
    column_name = column_name + std::to_string(i + 1);
    columns.push_back(column_name);
  }

  writer_.reset(new CSVWriter(columns, output_stream_.get()));
}

void ResultsWriter::AggregateAccuraciesAndNumImages(
    std::vector<double>* accuracies, int* num_done_images) {
  // Total images done.
  *num_done_images = 0;
  for (const auto entry : shard_id_done_image_count_map_) {
    *num_done_images += entry.second;
  }

  // Aggregated accuracies.
  for (int i = 0; i < top_k_; ++i) {
    double correct_inferences = 0;
    double total_inferences = 0;
    for (const auto entry : shard_id_done_image_count_map_) {
      const uint64_t shard_id = entry.first;
      const TopkAccuracyEvalMetrics& accuracy_metrics =
          shard_id_accuracy_metrics_map_.at(shard_id);
      const int num_images = entry.second;
      correct_inferences += num_images * accuracy_metrics.topk_accuracies(i);
      total_inferences += num_images;
    }
    // Convert to percentage.
    accuracies->push_back(100.0 * correct_inferences / total_inferences);
  }
}

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
  AggregateAccuraciesAndNumImages(&total_accuracies, &num_evaluated);
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
  AggregateAccuraciesAndNumImages(&total_accuracies, &num_evaluated);
  TopkAccuracyEvalMetrics aggregated_metrics;
  for (auto accuracy : total_accuracies) {
    aggregated_metrics.add_topk_accuracies(accuracy);
  }
  return aggregated_metrics;
}

void ResultsWriter::OutputEvalMetriccProto(
    const std::string& proto_output_file) {
  if (!proto_output_file.empty()) {
    std::ofstream proto_out_file(proto_output_file,
                                 std::ios::out | std::ios::binary);
    TopkAccuracyEvalMetrics metrics = AggregatedMetrics();
    proto_out_file << metrics.SerializeAsString();
    proto_out_file.close();
    LOG(INFO) << "The result metrics proto is written to " << proto_output_file;
  } else {
    LOG(INFO) << "Metrics proto output file path is not specified!";
  }
}

std::unique_ptr<ImagenetModelEvaluator> CreateImagenetModelEvaluator(
    int* argc, char* argv[], int num_threads) {
  std::unique_ptr<ImagenetModelEvaluator> evaluator;
  if (ImagenetModelEvaluator::Create(argc, argv, num_threads, &evaluator) !=
      kTfLiteOk) {
    evaluator.reset(nullptr);
  }

  return evaluator;
}

std::unique_ptr<ResultsWriter> CreateImagenetEvalResultsWriter(
    int top_k, const std::string& output_file_path) {
  std::unique_ptr<ResultsWriter> writer(
      new ResultsWriter(top_k, output_file_path));
  if (!writer->IsValid()) return nullptr;

  return writer;
}

}  // namespace metrics
}  // namespace tensorflow
