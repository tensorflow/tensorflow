/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#define EIGEN_USE_THREADS
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/calibration_parameters.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace {

using ::stablehlo::quantization::CalculateBinIndexSafe;
using ::stablehlo::quantization::CalculateBinWidth;
using ::stablehlo::quantization::CalculateLowerBound;
using ::stablehlo::quantization::CalibrationOptions;
using ::stablehlo::quantization::GetNumBins;
using CPUDevice = ::Eigen::ThreadPoolDevice;
using CalibrationMethod =
    ::stablehlo::quantization::CalibrationOptions_CalibrationMethod;

}  // namespace

REGISTER_OP("CustomAggregator")
    .Input("input: float")
    .Output("output: float")
    .Output("min: float")
    .Output("max: float")
    .Output("histogram: int64")
    .Attr("id: string")
    .Attr("calibration_method: int = 0")
    .Attr("initial_num_bins: int = 0")
    .Attr("min_percentile: float = 0.0")
    .Attr("max_percentile: float = 0.0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());

      const tensorflow::AttrValue* calibration_method_attr;
      TF_RETURN_IF_ERROR(
          c->GetAttr("calibration_method", &calibration_method_attr));
      int32_t num_bins = GetNumBins(
          static_cast<CalibrationMethod>(calibration_method_attr->i()));
      c->set_output(3, c->MakeShape({num_bins}));

      return absl::OkStatus();
    });

class CustomAggregatorOp : public OpKernel {
 public:
  explicit CustomAggregatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));

    int calibration_method_value;
    int initial_num_bins;
    float min_percentile;
    float max_percentile;
    OP_REQUIRES_OK(context, context->GetAttr("calibration_method",
                                             &calibration_method_value));
    OP_REQUIRES_OK(context,
                   context->GetAttr("initial_num_bins", &initial_num_bins));
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_percentile", &min_percentile));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_percentile", &max_percentile));

    auto calibration_method =
        static_cast<CalibrationMethod>(calibration_method_value);
    OP_REQUIRES(
        context,
        calibration_method !=
            CalibrationOptions::CALIBRATION_METHOD_UNSPECIFIED,
        absl::AbortedError("The calibration method must be specified."));

    calib_opts_.set_calibration_method(calibration_method);
    calib_opts_.mutable_calibration_parameters()->set_initial_num_bins(
        initial_num_bins);
    calib_opts_.mutable_calibration_parameters()->set_min_percentile(
        min_percentile);
    calib_opts_.mutable_calibration_parameters()->set_max_percentile(
        max_percentile);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    // Use the same input for the first output.
    context->set_output(0, input_tensor);

    // Calculate min/max statistics.
    const auto input_flat = input_tensor.flat<float>();
    Tensor *min_output = nullptr, *max_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("min", {}, &min_output));
    OP_REQUIRES_OK(context, context->allocate_output("max", {}, &max_output));
    min_output->scalar<float>().device(
        context->template eigen_device<CPUDevice>()) = input_flat.minimum();
    max_output->scalar<float>().device(
        context->template eigen_device<CPUDevice>()) = input_flat.maximum();

    // Calculate histogram statistics.
    int32_t num_bins = GetNumBins(calib_opts_.calibration_method());
    Tensor* histogram_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("histogram", {num_bins},
                                                     &histogram_output));
    if (num_bins > 0) {
      const float min_value = min_output->scalar<float>()();
      const float max_value = max_output->scalar<float>()();
      CalculateHistogramStatistics(context, input_tensor, min_value, max_value,
                                   num_bins, histogram_output);
    }

    // By passing calib_opts_ and input_tensor to CalibratorSingleton,
    // CalibrationStatisticsCollector can calculate statistics for calibration.
    calibrator::CalibratorSingleton::Report(id_, *min_output, *max_output,
                                            *histogram_output, calib_opts_);
  }

 private:
  std::string id_;
  CalibrationOptions calib_opts_;

  void CalculateHistogramStatistics(OpKernelContext* context,
                                    const Tensor& input_tensor,
                                    const float min_value,
                                    const float max_value,
                                    const int32_t num_bins,
                                    Tensor* histogram_tensor) {
    const auto input_flat = input_tensor.flat<float>();
    auto histogram_flat = histogram_tensor->flat<int64_t>();
    histogram_flat.setZero();

    const float bin_width = CalculateBinWidth(min_value, max_value, num_bins);
    const float lower_bound = CalculateLowerBound(min_value, bin_width);
    for (int i = 0; i < input_flat.size(); ++i) {
      int32_t bin_index = CalculateBinIndexSafe(
          input_flat.data()[i], lower_bound, bin_width, num_bins);
      histogram_flat.data()[bin_index] += 1;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CustomAggregator").Device(DEVICE_CPU),
                        CustomAggregatorOp);
}  // namespace tensorflow
