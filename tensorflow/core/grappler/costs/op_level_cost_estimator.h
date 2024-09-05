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

#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_context.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {
namespace grappler {

bool GetTensorShapeProtoFromTensorProto(const TensorProto& tensor_proto,
                                        TensorShapeProto* tensor_shape_proto);
std::vector<int64_t> MaybeGetMinimumShape(
    const TensorShapeProto& original_shape, int rank,
    bool* found_unknown_shapes);

// Node costs; an intermediate structure used within op level cost estimator.
struct NodeCosts {
  // If this FLAG is true, override calculated compute time with a minimum
  // value, instead of calculating it from num_compute_ops and compute ops/sec.
  // For example, PredictIdentity, PredictVariable, PredictMetadata set this
  // FLAG.
  bool minimum_cost_op = false;

  // Compute ops.
  int64_t num_compute_ops = 0;

  // Memory bytes accessed; note that these may be different to the size of
  // tensors.
  std::vector<int64_t> num_input_bytes_accessed;   // ordered by input tensors.
  std::vector<int64_t> num_output_bytes_accessed;  // ordered by output ports.
  int64_t internal_read_bytes = 0;
  int64_t internal_write_bytes = 0;

  // Convenience functions.
  int64_t num_total_input_bytes() const {
    return std::accumulate(num_input_bytes_accessed.begin(),
                           num_input_bytes_accessed.end(), 0LL);
  }
  int64_t num_total_read_bytes() const {
    return num_total_input_bytes() + internal_read_bytes;
  }
  int64_t num_total_output_bytes() const {
    return std::accumulate(num_output_bytes_accessed.begin(),
                           num_output_bytes_accessed.end(), 0LL);
  }
  int64_t num_total_write_bytes() const {
    return num_total_output_bytes() + internal_write_bytes;
  }
  int64_t num_bytes_accessed() const {
    return num_total_read_bytes() + num_total_write_bytes();
  }

  // Memory usage.
  int64_t max_memory = 0;
  int64_t persistent_memory = 0;
  int64_t temporary_memory = 0;

  // Stats.
  int64_t num_nodes = 1;
  int64_t num_nodes_with_unknown_shapes = 0;
  int64_t num_nodes_with_unknown_op_type = 0;
  int64_t num_nodes_with_pure_memory_op = 0;
  bool inaccurate = false;

  // TODO(dyoon): this is added for compatibility; some old code is hard to
  // migrate; hence, using these as a backup. Once we clean up, we'll delete
  // these fields. New code should not use these.
  bool has_costs = false;
  Costs costs;
};

class OpLevelCostEstimator {
 public:
  OpLevelCostEstimator();
  virtual ~OpLevelCostEstimator() {}

  virtual Costs PredictCosts(const OpContext& op_context) const;

  // Returns basic device performance info.
  virtual DeviceInfo GetDeviceInfo(const DeviceProperties& device) const;

 protected:
  // TODO(dyoon): Consider to remove PredictOpCountBasedCosts() with OpInfo.
  // Naive cost estimate based on the given operations count and total
  // input/output tensor sizes of the given op_info combined.
  Costs PredictOpCountBasedCost(double operations, const OpInfo& op_info) const;

  // Naive cost estimate based on the given operations count and the given total
  // io size in bytes. Sizes of op_info inputs and outputs are not taken into
  // consideration.
  Costs PredictOpCountBasedCost(double operations, double input_io_bytes,
                                double output_io_bytes,
                                const OpInfo& op_info) const;

  // Top-level method cost function (PredictCosts calls this method to get
  // NodeCosts, and then converts it to Costs). PredictNodeCosts() calls other
  // Predict methods depending on op types.
  absl::Status PredictNodeCosts(const OpContext& op_context,
                                NodeCosts* node_costs) const;

  // Predict cost of an op for which no accurate estimator is defined.
  absl::Status PredictCostOfAnUnknownOp(const OpContext& op_context,
                                        NodeCosts* node_costs) const;

  // This family of routines predicts the costs to
  // perform the specified TensorFlow Op on the
  // device represented by a subclass. The default
  // implementation just divides the operations to
  // perform the op (from the "Count" routines,
  // above) by the device peak operations per
  // second.
  // Implementation of costs other than
  // execution_time is optional, depending on the
  // device.
  absl::Status PredictNaryOp(const OpContext& op_context,
                             NodeCosts* node_costs) const;
  absl::Status PredictConv2D(const OpContext& op_context,
                             NodeCosts* node_costs) const;
  absl::Status PredictCwiseOp(const OpContext& op_context,
                              NodeCosts* node_costs) const;
  absl::Status PredictConv2DBackpropInput(const OpContext& op_context,
                                          NodeCosts* node_costs) const;
  absl::Status PredictConv2DBackpropFilter(const OpContext& op_context,
                                           NodeCosts* node_costs) const;
  absl::Status PredictFusedConv2DBiasActivation(const OpContext& op_context,
                                                NodeCosts* node_costs) const;
  absl::Status PredictMatMul(const OpContext& op_context,
                             NodeCosts* node_costs) const;
  absl::Status PredictSparseTensorDenseMatMul(const OpContext& op_context,
                                              NodeCosts* node_costs) const;
  absl::Status PredictNoOp(const OpContext& op_context,
                           NodeCosts* node_costs) const;
  absl::Status PredictIdentity(const OpContext& op_context,
                               NodeCosts* node_costs) const;
  absl::Status PredictVariable(const OpContext& op_context,
                               NodeCosts* node_costs) const;
  absl::Status PredictBatchMatMul(const OpContext& op_context,
                                  NodeCosts* node_costs) const;
  absl::Status PredictMetadata(const OpContext& op_context,
                               NodeCosts* node_costs) const;
  absl::Status PredictGatherOrSlice(const OpContext& op_context,
                                    NodeCosts* node_costs) const;
  absl::Status PredictScatter(const OpContext& op_context,
                              NodeCosts* node_costs) const;
  absl::Status PredictMaxPool(const OpContext& op_context,
                              NodeCosts* node_costs) const;
  absl::Status PredictMaxPoolGrad(const OpContext& op_context,
                                  NodeCosts* node_costs) const;
  absl::Status PredictAvgPool(const OpContext& op_context,
                              NodeCosts* node_costs) const;
  absl::Status PredictAvgPoolGrad(const OpContext& op_context,
                                  NodeCosts* node_costs) const;
  absl::Status PredictFusedBatchNorm(const OpContext& op_context,
                                     NodeCosts* node_costs) const;
  absl::Status PredictFusedBatchNormGrad(const OpContext& op_context,
                                         NodeCosts* node_costs) const;
  absl::Status PredictEinsum(const OpContext& op_context,
                             NodeCosts* node_costs) const;
  absl::Status PredictAssignVariableOps(const OpContext& op_context,
                                        NodeCosts* node_costs) const;
  absl::Status PredictPureMemoryOp(const OpContext& op_context,
                                   NodeCosts* node_costs) const;
  absl::Status PredictSoftmax(const OpContext& op_context,
                              NodeCosts* node_costs) const;
  absl::Status PredictResizeBilinear(const OpContext& op_context,
                                     NodeCosts* node_costs) const;
  absl::Status PredictCropAndResize(const OpContext& op_context,
                                    NodeCosts* node_costs) const;

  int64_t GetSoftmaxComputeOps(const OpContext& op_context) const;

  // Generic cost prediction method for fused operations.
  absl::Status PredictFusedOp(const OpContext& op_context,
                              const std::vector<OpContext>& fused_op_contexts,
                              NodeCosts* node_costs) const;

  // Utility function for safe division. Returns 0
  // if rhs is 0 or negative.
  static double SafeDiv(const double lhs, const double rhs) {
    if (rhs > 0) {
      return lhs / rhs;
    } else {
      return 0.0;
    }
  }

  // This family of routines counts the number of operations to perform the
  // specified TensorFlow Op.
  struct MatMulDimensions {
    int m;
    int n;
    int k;
  };
  struct BatchMatMulDimensions {
    std::vector<int> batch_dims;
    MatMulDimensions matmul_dims;
  };
  struct ConvolutionDimensions {
    int64_t batch;  // Batch size.
    int64_t ix;     // Input size x.
    int64_t iy;     // Input size y.
    int64_t iz;     // Input depth.
    int64_t kx;     // Kernel x.
    int64_t ky;     // Kernel y.
    int64_t kz;     // Kernel depth (in case of group convolution, this will be
                    // smaller than input depth).
    int64_t oz;     // Output depth.
    int64_t ox;     // Output size x.
    int64_t oy;     // Output size y.
    int64_t sx;     // Stride x.
    int64_t sy;     // Stride y.
    Padding padding;  // SAME or VALID.
  };
  static int64_t CountConv2DOperations(const OpInfo& op_info,
                                       bool* found_unknown_shapes);
  static int64_t CountConv2DOperations(const OpInfo& op_info,
                                       ConvolutionDimensions* conv_info,
                                       bool* found_unknown_shapes);
  static int64_t CountMatMulOperations(const OpInfo& op_info,
                                       bool* found_unknown_shapes);
  static int64_t CountMatMulOperations(const OpInfo& op_info,
                                       MatMulDimensions* mat_mul,
                                       bool* found_unknown_shapes);
  static int64_t CountMatMulOperations(const OpInfo& op_info, bool transpose_a,
                                       bool transpose_b,
                                       MatMulDimensions* mat_mul,
                                       bool* found_unknown_shapes);
  bool GenerateBatchMatmulContextFromEinsum(const OpContext& einsum_context,
                                            OpContext* batch_matmul_context,
                                            bool* found_unknown_shapes) const;
  static int64_t CountBatchMatMulOperations(const OpInfo& op_info,
                                            bool* found_unknown_shapes);
  static int64_t CountBatchMatMulOperations(
      const OpInfo& op_info, BatchMatMulDimensions* batch_mat_mul,
      bool* found_unknown_shapes);
  static int64_t CountConv2DBackpropInputOperations(
      const OpInfo& op_info, ConvolutionDimensions* returned_conv_dims,
      bool* found_unknown_shapes);
  static int64_t CountConv2DBackpropFilterOperations(
      const OpInfo& op_info, ConvolutionDimensions* returned_conv_dims,
      bool* found_unknown_shapes);

  // Calculate the element count of an input/output tensor.
  static int64_t CalculateTensorElementCount(
      const OpInfo::TensorProperties& tensor, bool* found_unknown_shapes);

  // Calculate the total size in bytes of an input/output tensor.
  static int64_t CalculateTensorSize(const OpInfo::TensorProperties& tensor,
                                     bool* found_unknown_shapes);

  // Calculate the element count of the largest
  // input of specified TensorFlow op.
  static int64_t CalculateLargestInputCount(const OpInfo& op_info,
                                            bool* found_unknown_shapes);

  // Calculate the total size in bytes of the all
  // the inputs of specified TensorFlow op.
  static int64_t CalculateInputSize(const OpInfo& op_info,
                                    bool* found_unknown_shapes);

  // Same, but a vector format: one for each input.
  static std::vector<int64_t> CalculateInputTensorSize(
      const OpInfo& op_info, bool* found_unknown_shapes);

  // Calculate the total size in bytes of the all
  // the outputs of specified TensorFlow op.
  static int64_t CalculateOutputSize(const OpInfo& op_info,
                                     bool* found_unknown_shapes);

  // Same, but a vector format: one for each output.
  static std::vector<int64_t> CalculateOutputTensorSize(
      const OpInfo& op_info, bool* found_unknown_shapes);

  // For convolution and its grad ops.
  static ConvolutionDimensions ConvolutionDimensionsFromInputs(
      const TensorShapeProto& original_image_shape,
      const TensorShapeProto& original_filter_shape, const OpInfo& op_info,
      bool* found_unknown_shapes);

  // For Pooling, FusedBatchNorm, and their grad ops.
  static absl::StatusOr<ConvolutionDimensions> OpDimensionsFromInputs(
      const TensorShapeProto& original_image_shape, const OpInfo& op_info,
      bool* found_unknown_shapes);

  // Helper to construct child operation contexts for the component operations
  // of fused ops.
  static OpContext FusedChildContext(
      const OpContext& parent, const string& op_name,
      const OpInfo::TensorProperties& output,
      const std::vector<OpInfo::TensorProperties>& inputs);

  // Helper to construct tensor shapes.
  static OpInfo::TensorProperties DescribeTensor(
      DataType type, const std::vector<int64_t>& dims);

  // Helper method for building common case NodeCosts.
  static absl::Status PredictDefaultNodeCosts(int64_t num_compute_ops,
                                              const OpContext& op_context,
                                              bool* found_unknown_shapes,
                                              NodeCosts* node_costs);

 protected:
  std::map<string, int> elementwise_ops_;
  typedef std::function<absl::Status(const OpContext& op_context, NodeCosts*)>
      CostImpl;
  std::map<string, CostImpl> device_cost_impl_;
  // If true, assume compute and memory overlap; hence, the op cost is max of
  // compute_time and memory_time, instead of sum of those two.
  bool compute_memory_overlap_;
  std::set<string> persistent_ops_;

 private:
  friend class OpLevelCostEstimatorTest;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_OP_LEVEL_COST_ESTIMATOR_H_
