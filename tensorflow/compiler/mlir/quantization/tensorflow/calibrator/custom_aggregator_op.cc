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
#include <string>

#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::quantization::CalibrationOptions;

REGISTER_OP("CustomAggregator")
    .Input("input: float")
    .Output("output: float")
    .Attr("id: string")
    .Attr("calibration_method: int = 0")
    .Attr("initial_num_bins: int = 0")
    .Attr("min_percentile: float = 0.0")
    .Attr("max_percentile: float = 0.0")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

class CustomAggregatorOp : public OpKernel {
 public:
  explicit CustomAggregatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    int initial_num_bins;
    int calibration_method;
    float min_percentile;
    float max_percentile;
    OP_REQUIRES_OK(
        context, context->GetAttr("calibration_method", (&calibration_method)));
    OP_REQUIRES_OK(context,
                   context->GetAttr("initial_num_bins", &initial_num_bins));
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_percentile", &min_percentile));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_percentile", &max_percentile));
    calib_opts_.set_calibration_method(
        static_cast<CalibrationOptions::CalibrationMethod>(calibration_method));
    calib_opts_.mutable_calibration_parameters()->set_initial_num_bins(
        initial_num_bins);
    calib_opts_.mutable_calibration_parameters()->set_min_percentile(
        min_percentile);
    calib_opts_.mutable_calibration_parameters()->set_max_percentile(
        max_percentile);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    auto input_flat = input_tensor.flat<float>();

    const int N = input_flat.size();
    if (N == 0) {
      // Use the same input for the output.
      context->set_output(0, input_tensor);
      return;
    }

    // By passing calib_opts_ and input_tensor to CalibratorSingleton,
    // CalibrationStatisticsCollector can calculate statistics for calibration.
    calibrator::CalibratorSingleton::Report(id_, input_tensor, calib_opts_);

    // Use the same input for the output.
    context->set_output(0, input_tensor);
  }

 private:
  std::string id_;
  CalibrationOptions calib_opts_;
};

REGISTER_KERNEL_BUILDER(Name("CustomAggregator").Device(DEVICE_CPU),
                        CustomAggregatorOp);
}  // namespace tensorflow
