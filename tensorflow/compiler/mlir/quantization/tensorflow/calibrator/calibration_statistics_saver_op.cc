/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_average_min_max.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_histogram.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics_collector_min_max.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/file_system.h"

namespace tensorflow {
namespace {

using ::stablehlo::quantization::CalibrationOptions;
using CalibrationMethod =
    ::stablehlo::quantization::CalibrationOptions_CalibrationMethod;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibrationStatisticsCollectorAverageMinMax;
using ::tensorflow::calibrator::CalibrationStatisticsCollectorBase;
using ::tensorflow::calibrator::CalibrationStatisticsCollectorHistogram;
using ::tensorflow::calibrator::CalibrationStatisticsCollectorMinMax;
using ::tensorflow::calibrator::CalibrationStatisticsMap;

}  // namespace

REGISTER_OP("CalibrationStatisticsSaver")
    .Input("args: Tin")
    .Attr("Tin: list(type) >= 0")
    .Attr("ids: list(string) >= 1")
    .Attr("calibration_methods: list(int) >= 1")
    .Attr("output_file_path: string")
    .SetIsStateful()
    .Doc(R"doc(
Aggregates and saves the calibration statistics data.

This op collects outputs of multiples CustomAggregator ops, which includes
`min`, `max` and `histogram`. Then it aggregates them according to the
calibration method and save the result to the given file path as a binary
proto file.)doc");

class CalibrationStatisticsSaverOp : public OpKernel {
 public:
  explicit CalibrationStatisticsSaverOp(
      absl::Nonnull<OpKernelConstruction*> context)
      : OpKernel(context) {
    std::string output_file_path;
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_file_path", &output_file_path));
    OP_REQUIRES_OK(context, context->env()->NewWritableFile(output_file_path,
                                                            &output_file_));

    OP_REQUIRES_OK(context, context->GetAttr("ids", &ids_));
    OP_REQUIRES_OK(context, context->GetAttr("calibration_methods",
                                             &calibration_methods_));
    OP_REQUIRES(
        context, ids_.size() == calibration_methods_.size(),
        absl::AbortedError(
            "The `ids` and `calibration_methods` must have the same size."));

    // Check the number and type of inputs.
    OP_REQUIRES(context, context->num_inputs() == ids_.size() * 3,
                absl::AbortedError("The number of inputs must be  three times "
                                   "the size of the `ids` list."));
    for (int i = 0; i < ids_.size(); ++i) {
      OP_REQUIRES(context, context->input_type(i * 3) == DT_FLOAT,
                  absl::AbortedError("The input `min` must have float type."));
      OP_REQUIRES(context, context->input_type(i * 3 + 1) == DT_FLOAT,
                  absl::AbortedError("The input `max` must have float type."));
      OP_REQUIRES(
          context, context->input_type(i * 3 + 2) == DT_INT64,
          absl::AbortedError("The input `histogram` must have int64 type."));
    }
  }

  ~CalibrationStatisticsSaverOp() override {
    // Save to file during destruction so we only save it once.
    // TODO - b/335044516 : Find a way to flush outside of the destructor.
    CalibrationStatisticsMap statistics_map;
    for (const auto& [id, collector] : id_to_collector_) {
      std::optional<CalibrationStatistics> statistics =
          collector->GetStatistics();
      if (!statistics.has_value()) continue;

      statistics_map.mutable_statistics()->emplace(id, std::move(*statistics));
    }

    if (auto status = output_file_->Append(statistics_map.SerializeAsString());
        !status.ok()) {
      LOG(ERROR) << "Failed to write calibration statistics: "
                 << status.message();
    }
    if (auto status = output_file_->Close(); !status.ok()) {
      LOG(ERROR) << "Failed to close calibration statistics file: "
                 << status.message();
    }
  }

  void Compute(absl::Nonnull<OpKernelContext*> context) override {
    for (int idx = 0; idx < ids_.size(); ++idx) {
      AssignIfNotExists(
          ids_[idx], static_cast<CalibrationMethod>(calibration_methods_[idx]));

      const Tensor& min_tensor = context->input(3 * idx);
      const Tensor& max_tensor = context->input(3 * idx + 1);
      const Tensor& histogram_tensor = context->input(3 * idx + 2);

      const float min_value = min_tensor.scalar<float>()();
      const float max_value = max_tensor.scalar<float>()();
      auto histogram_flat = histogram_tensor.flat<int64_t>();
      absl::Span<const int64_t> histogram_data =
          absl::MakeSpan(histogram_flat.data(), histogram_flat.size());
      id_to_collector_[ids_[idx]]->Collect(min_value, max_value,
                                           histogram_data);
    }
  }

 private:
  // The path to save calibration statistics data.
  std::unique_ptr<tsl::WritableFile> output_file_;
  // The id and calibration method of preceding CustomAggregator ops.
  std::vector<std::string> ids_;
  std::vector<int32_t> calibration_methods_;
  // Map from id to its collector instance.
  absl::flat_hash_map<std::string,
                      std::unique_ptr<CalibrationStatisticsCollectorBase>>
      id_to_collector_;

  void AssignIfNotExists(absl::string_view id,
                         const CalibrationMethod calibration_method) {
    std::unique_ptr<CalibrationStatisticsCollectorBase>& collector =
        id_to_collector_[id];

    if (collector != nullptr) return;

    switch (calibration_method) {
      case CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX:
        collector =
            std::make_unique<CalibrationStatisticsCollectorAverageMinMax>();
        break;
      case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE:
      case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE:
      case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC:
      case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY:
        collector = std::make_unique<CalibrationStatisticsCollectorHistogram>();
        break;
      case CalibrationOptions::CALIBRATION_METHOD_MIN_MAX:
      default:
        collector = std::make_unique<CalibrationStatisticsCollectorMinMax>();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CalibrationStatisticsSaver").Device(DEVICE_CPU),
                        CalibrationStatisticsSaverOp);

}  // namespace tensorflow
