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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("CustomAggregator")
    .Input("input: float")
    .Output("output: float")
    .Attr("id: string")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class CustomAggregatorOp : public OpKernel {
 public:
  explicit CustomAggregatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input_flat = input_tensor.flat<float>();

    const int N = input_flat.size();
    if (N == 0) {
      // Use the same input for the output.
      context->set_output(0, input_tensor);
      return;
    }

    const float* data = input_flat.data();
    auto minmax = std::minmax_element(data, data + input_flat.size());

    // Report the min/max values.
    calibrator::CalibratorSingleton::ReportMinMax(id_, *minmax.first,
                                                  *minmax.second);

    // Use the same input for the output.
    context->set_output(0, input_tensor);
  }

 private:
  std::string id_;
};

REGISTER_KERNEL_BUILDER(Name("CustomAggregator").Device(DEVICE_CPU),
                        CustomAggregatorOp);
}  // namespace tensorflow
