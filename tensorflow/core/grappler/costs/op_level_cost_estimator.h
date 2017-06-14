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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_OP_LEVEL_COST_ESTIMATOR_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_OP_LEVEL_COST_ESTIMATOR_H_

#include <functional>
#include <map>
#include <string>

#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {
namespace grappler {

class OpLevelCostEstimator {
 public:
  OpLevelCostEstimator();
  virtual ~OpLevelCostEstimator() {}

  virtual Costs PredictCosts(const OpInfo& op_features) const;

 protected:
  // Returns an estimate of device performance (in billions of operations
  // executed per second) and memory bandwidth (in GigaBytes/second) for the
  // specified device.
  virtual std::pair<double, double> GetDeviceInfo(
      const DeviceProperties& device) const;

  // For operations for which we haven't yet built estimates, returns a dummy
  // value based on input size.
  Costs DummyExecutionTime(const OpInfo& op_features) const;

  // Naive cost estimate based on operations divided by device ops/sec.
  Costs PredictOpCountBasedCost(double operations,
                                const OpInfo& op_features) const;

  // This family of routines counts the number of operations to perform the
  // specified TensorFlow Op.
  struct MatMulDimensions {
    int m;
    int n;
    int k;
  };
  struct ConvolutionDimensions {
    int64 batch;      // Batch size.
    int64 ix;         // Input size x.
    int64 iy;         // Input size y.
    int64 iz;         // Input depth.
    int64 kx;         // Kernel x.
    int64 ky;         // Kernel y.
    int64 oz;         // Output depth.
    int64 ox;         // Output size x.
    int64 oy;         // Output size y.
    int64 sx;         // Stride x.
    int64 sy;         // Stride y.
    Padding padding;  // SAME or VALID.
  };
  int64 CountConv2DOperations(const OpInfo& op_features,
                              bool* found_unknown_shapes) const;
  int64 CountConv2DOperations(const OpInfo& op_features,
                              ConvolutionDimensions* conv_info,
                              bool* found_unknown_shapes) const;
  int64 CountMatMulOperations(const OpInfo& op_features,
                              bool* found_unknown_shapes) const;
  int64 CountMatMulOperations(const OpInfo& op_features,
                              MatMulDimensions* mat_mul,
                              bool* found_unknown_shapes) const;
  int64 CountBatchMatMulOperations(const OpInfo& op_features,
                                   bool* found_unknown_shapes) const;
  int64 CountConv2DBackPropInputOperations(const OpInfo& op_features,
                                           ConvolutionDimensions* conv_info,
                                           bool* found_unknown_shapes) const;
  int64 CountConv2DBackPropFilterOperations(const OpInfo& op_features,
                                            ConvolutionDimensions* conv_info,
                                            bool* found_unknown_shapes) const;

  // Calculate the element count of an input/output tensor.
  int64 CalculateTensorElementCount(const OpInfo::TensorProperties& tensor,
                                    bool* found_unknown_shapes) const;

  // Calculate the total size in bytes of an input/output tensor.
  int64 CalculateTensorSize(const OpInfo::TensorProperties& tensor,
                            bool* found_unknown_shapes) const;

  // Calculate the element count of the largest
  // input of specified TensorFlow op.
  int64 CalculateLargestInputCount(const OpInfo& op_features,
                                   bool* found_unknown_shapes) const;

  // Calculate the total size in bytes of the all
  // the inputs of specified TensorFlow op.
  int64 CalculateInputSize(const OpInfo& op_features,
                           bool* found_unknown_shapes) const;

  // Calculate the total size in bytes of the all
  // the outputs of specified TensorFlow op.
  int64 CalculateOutputSize(const OpInfo& op_features,
                            bool* found_unknown_shapes) const;

  // This family of routines predicts the costs to
  // perform the specified TensorFlow Op on the
  // device represented by a subclass. The default
  // implementation just divides the operations to
  // perform the op (from the "Count" routines,
  // above) by the device peak operations per
  // second. Override to supply a better estimate.
  // Implementation of costs other than
  // execution_time is optional, depending on the
  // device.
  Costs PredictConv2D(const OpInfo& op_features) const;
  Costs PredictCwiseOp(const OpInfo& op_features) const;
  Costs PredictConv2DBackPropInput(const OpInfo& op_features) const;
  Costs PredictConv2DBackPropFilter(const OpInfo& op_features) const;
  Costs PredictMatMul(const OpInfo& op_features) const;
  Costs PredictNoOp(const OpInfo& op_features) const;
  Costs PredictBatchMatMul(const OpInfo& op_features) const;

  // Utility function for safe division. Returns 0
  // if rhs is 0 or negative.
  static double SafeDiv(const double lhs, const double rhs) {
    if (rhs > 0) {
      return lhs / rhs;
    } else {
      return 0.0;
    }
  }

  static ConvolutionDimensions ConvolutionDimensionsFromInputs(
      const TensorShapeProto& original_image_shape,
      const TensorShapeProto& original_filter_shape, const OpInfo& op_features,
      bool* found_unknown_shapes);

 protected:
  std::map<string, int> elementwise_ops_;
  typedef std::function<Costs(const OpInfo& op_feature)> CostImpl;
  std::map<string, CostImpl> device_cost_impl_;

 private:
  friend class OpLevelCostEstimatorTest;
};

}  // end namespace grappler
}  // end namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_OP_LEVEL_COST_ESTIMATOR_H_
