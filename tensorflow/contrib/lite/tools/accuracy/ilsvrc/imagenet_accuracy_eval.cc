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

#include <iomanip>
#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/contrib/lite/tools/accuracy/csv_writer.h"
#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"
#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_topk_eval.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace metrics {

namespace {

std::vector<double> GetAccuracies(
    const ImagenetTopKAccuracy::AccuracyStats& accuracy_stats) {
  std::vector<double> results;
  results.reserve(accuracy_stats.number_of_images);
  if (accuracy_stats.number_of_images > 0) {
    for (int n : accuracy_stats.topk_counts) {
      double accuracy = 0;
      if (accuracy_stats.number_of_images > 0) {
        accuracy = (n * 100.0) / accuracy_stats.number_of_images;
      }
      results.push_back(accuracy);
    }
  }
  return results;
}

}  // namespace

// Writes results to a CSV file.
class ResultsWriter : public ImagenetModelEvaluator::Observer {
 public:
  explicit ResultsWriter(std::unique_ptr<CSVWriter> writer)
      : writer_(std::move(writer)) {}

  void OnEvaluationStart(int total_number_of_images) override {}

  void OnSingleImageEvaluationComplete(
      const ImagenetTopKAccuracy::AccuracyStats& stats,
      const string& image) override;

 private:
  std::unique_ptr<CSVWriter> writer_;
};

void ResultsWriter::OnSingleImageEvaluationComplete(
    const ImagenetTopKAccuracy::AccuracyStats& stats, const string& image) {
  TF_CHECK_OK(writer_->WriteRow(GetAccuracies(stats)));
  writer_->Flush();
}

// Logs results to standard output with `kLogDelayUs` microseconds.
class ResultsLogger : public ImagenetModelEvaluator::Observer {
 public:
  void OnEvaluationStart(int total_number_of_images) override;

  void OnSingleImageEvaluationComplete(
      const ImagenetTopKAccuracy::AccuracyStats& stats,
      const string& image) override;

 private:
  int total_num_images_ = 0;
  uint64 last_logged_time_us_ = 0;
  static constexpr int kLogDelayUs = 500 * 1000;
};

void ResultsLogger::OnEvaluationStart(int total_number_of_images) {
  total_num_images_ = total_number_of_images;
  LOG(ERROR) << "Starting model evaluation: " << total_num_images_;
}

void ResultsLogger::OnSingleImageEvaluationComplete(
    const ImagenetTopKAccuracy::AccuracyStats& stats, const string& image) {
  int num_evaluated = stats.number_of_images;

  double current_percent = num_evaluated * 100.0 / total_num_images_;
  auto now_us = Env::Default()->NowMicros();

  if ((now_us - last_logged_time_us_) >= kLogDelayUs) {
    last_logged_time_us_ = now_us;

    LOG(ERROR) << "Evaluated " << num_evaluated << "/" << total_num_images_
               << " images, " << std::setprecision(2) << std::fixed
               << current_percent << "%";
  }
}

int Main(int argc, char* argv[]) {
  // TODO(shashishekhar): Make this binary configurable and model
  // agnostic.
  string output_file_path;
  std::vector<Flag> flag_list = {
      Flag("output_file_path", &output_file_path, "Path to output file."),
  };
  Flags::Parse(&argc, argv, flag_list);

  std::unique_ptr<ImagenetModelEvaluator> evaluator;
  CHECK(!output_file_path.empty()) << "Invalid output file path.";

  TF_CHECK_OK(ImagenetModelEvaluator::Create(argc, argv, &evaluator));

  std::ofstream output_stream(output_file_path, std::ios::out);
  CHECK(output_stream) << "Unable to open output file path: '"
                       << output_file_path << "'";

  output_stream << std::setprecision(3) << std::fixed;
  std::vector<string> columns;
  columns.reserve(evaluator->params().num_ranks);
  for (int i = 0; i < evaluator->params().num_ranks; i++) {
    string column_name = "Top ";
    tensorflow::strings::StrAppend(&column_name, i + 1);
    columns.push_back(column_name);
  }

  ResultsWriter results_writer(
      absl::make_unique<CSVWriter>(columns, &output_stream));
  ResultsLogger logger;
  evaluator->AddObserver(&results_writer);
  evaluator->AddObserver(&logger);
  TF_CHECK_OK(evaluator->EvaluateModel());
  return 0;
}

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::metrics::Main(argc, argv);
}
