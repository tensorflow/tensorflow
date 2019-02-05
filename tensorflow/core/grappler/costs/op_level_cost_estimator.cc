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

#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/utils.h"

namespace tensorflow {
namespace grappler {

constexpr int kOpsPerMac = 2;
constexpr char kConst[] = "Const";
constexpr char kGuaranteeConst[] = "GuaranteeConst";
constexpr char kConv2d[] = "Conv2D";
constexpr char kConv2dBackpropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv2dBackpropInput[] = "Conv2DBackpropInput";
constexpr char kFusedConv2dBiasActivation[] = "FusedConv2DBiasActivation";
constexpr char kDepthwiseConv2dNative[] = "DepthwiseConv2dNative";
constexpr char kDepthwiseConv2dNativeBackpropFilter[] =
    "DepthwiseConv2dNativeBackpropFilter";
constexpr char kDepthwiseConv2dNativeBackpropInput[] =
    "DepthwiseConv2dNativeBackpropInput";
constexpr char kMatMul[] = "MatMul";
constexpr char kSparseMatMul[] = "SparseMatMul";
constexpr char kPlaceholder[] = "Placeholder";
constexpr char kIdentity[] = "Identity";
constexpr char kIdentityN[] = "IdentityN";
constexpr char kRefIdentity[] = "RefIdentity";
constexpr char kNoOp[] = "NoOp";
constexpr char kReshape[] = "Reshape";
constexpr char kSqueeze[] = "Squeeze";
constexpr char kRecv[] = "_Recv";
constexpr char kSend[] = "_Send";
constexpr char kBatchMatMul[] = "BatchMatMul";
constexpr char kVariable[] = "Variable";
constexpr char kVariableV2[] = "VariableV2";
constexpr char kRank[] = "Rank";
constexpr char kShape[] = "Shape";
constexpr char kShapeN[] = "ShapeN";
constexpr char kSize[] = "Size";
constexpr char kStopGradient[] = "StopGradient";
constexpr char kPreventGradient[] = "PreventGradient";
constexpr char kGather[] = "Gather";
constexpr char kGatherV2[] = "GatherV2";
constexpr char kSlice[] = "Slice";
constexpr char kMaxPool[] = "MaxPool";
constexpr char kMaxPoolGrad[] = "MaxPoolGrad";
constexpr char kAvgPool[] = "AvgPool";
constexpr char kAvgPoolGrad[] = "AvgPoolGrad";
constexpr char kFusedBatchNorm[] = "FusedBatchNorm";
constexpr char kFusedBatchNormGrad[] = "FusedBatchNormGrad";
constexpr char kQuantizedMatMulV2[] = "QuantizedMatMulV2";

static const Costs::Duration kMinComputeTime(1);

namespace {

string GetDataFormat(const OpInfo& op_info) {
  string data_format = "NHWC";  // Default format.
  if (op_info.attr().find("data_format") != op_info.attr().end()) {
    data_format = op_info.attr().at("data_format").s();
  }
  return data_format;
}

string GetFilterFormat(const OpInfo& op_info) {
  string filter_format = "HWIO";  // Default format.
  if (op_info.attr().find("filter_format") != op_info.attr().end()) {
    filter_format = op_info.attr().at("filter_format").s();
  }
  return filter_format;
}

Padding GetPadding(const OpInfo& op_info) {
  if (op_info.attr().find("padding") != op_info.attr().end() &&
      op_info.attr().at("padding").s() == "VALID") {
    return Padding::VALID;
  }
  return Padding::SAME;  // Default padding.
}

bool IsTraining(const OpInfo& op_info) {
  if (op_info.attr().find("is_training") != op_info.attr().end() &&
      op_info.attr().at("is_training").b()) {
    return true;
  }
  return false;
}

// TODO(dyoon): support non-4D tensors in the c ost functions of convolution
// related ops (Conv, Pool, BatchNorm, and their backprops) and the related
// helper functions.
std::vector<int64> GetStrides(const OpInfo& op_info) {
  if (op_info.attr().find("strides") != op_info.attr().end()) {
    const auto strides = op_info.attr().at("strides").list().i();
    CHECK(strides.size() == 4)
        << "Attr strides is not a length-4 vector: " << op_info.DebugString();
    return {strides[0], strides[1], strides[2], strides[3]};
  }
  return {1, 1, 1, 1};
}

std::vector<int64> GetKernelSize(const OpInfo& op_info) {
  if (op_info.attr().find("ksize") != op_info.attr().end()) {
    const auto ksize = op_info.attr().at("ksize").list().i();
    CHECK(ksize.size() == 4)
        << "Attr ksize is not a length-4 vector: " << op_info.DebugString();
    return {ksize[0], ksize[1], ksize[2], ksize[3]};
  }
  // Note that FusedBatchNorm doesn't have ksize attr, but GetKernelSize returns
  // {1, 1, 1, 1} in that case.
  return {1, 1, 1, 1};
}

int64 GetOutputSize(const int64 input, const int64 filter, const int64 stride,
                    const Padding& padding) {
  // Logic for calculating output shape is from GetWindowedOutputSizeVerbose()
  // function in third_party/tensorflow/core/framework/common_shape_fns.cc.
  if (padding == Padding::VALID) {
    return (input - filter + stride) / stride;
  } else {  // SAME.
    return (input + stride - 1) / stride;
  }
}

// Return the output element count of a binary element-wise op considering
// broadcasting.
int64 CwiseOutputElementCount(const TensorShapeProto& input_shape_1,
                              const TensorShapeProto& input_shape_2) {
  bool found_unknown_shapes;
  int rank = std::max(1, input_shape_1.dim_size());
  TensorShapeProto output_shape =
      MaybeGetMinimumShape(input_shape_1, rank, &found_unknown_shapes);

  if (input_shape_1.dim_size() == input_shape_2.dim_size()) {
    auto shape_1 =
        MaybeGetMinimumShape(input_shape_1, rank, &found_unknown_shapes);
    auto shape_2 =
        MaybeGetMinimumShape(input_shape_2, rank, &found_unknown_shapes);
    if (shape_1.dim_size() == shape_2.dim_size()) {
      for (int i = 0; i < shape_1.dim_size(); i++) {
        output_shape.mutable_dim(i)->set_size(
            std::max(shape_1.dim(i).size(), shape_2.dim(i).size()));
      }
    }
  }

  int64 count = 1;
  for (int i = 0; i < output_shape.dim_size(); i++) {
    count *= output_shape.dim(i).size();
  }
  return count;
}

}  // namespace

// Return a minimum shape if the shape is unknown. If known, return the original
// shape.
TensorShapeProto MaybeGetMinimumShape(const TensorShapeProto& original_shape,
                                      int rank, bool* found_unknown_shapes) {
  auto shape = original_shape;
  bool is_scalar = !shape.unknown_rank() && shape.dim_size() == 0;

  if (shape.unknown_rank() || (!is_scalar && shape.dim_size() < rank)) {
    *found_unknown_shapes = true;
    VLOG(2) << "Use minimum shape because the rank is unknown.";
    // The size of each dimension is at least 1, if unknown.
    for (int i = shape.dim_size(); i < rank; i++) {
      shape.add_dim()->set_size(1);
    }
  } else if (is_scalar) {
    for (int i = 0; i < rank; i++) {
      shape.add_dim()->set_size(1);
    }
  } else if (shape.dim_size() > rank) {
    *found_unknown_shapes = true;
    shape.clear_dim();
    for (int i = 0; i < rank; i++) {
      shape.add_dim()->set_size(original_shape.dim(i).size());
    }
  } else {
    for (int i = 0; i < shape.dim_size(); i++) {
      if (shape.dim(i).size() < 0) {
        *found_unknown_shapes = true;
        VLOG(2) << "Use minimum dim size 1 because the shape is unknown.";
        // The size of each dimension is at least 1, if unknown.
        shape.mutable_dim(i)->set_size(1);
      }
    }
  }
  return shape;
}

OpLevelCostEstimator::OpLevelCostEstimator() {
  // Syntactic sugar to build and return a lambda that takes an OpInfo and
  // returns a cost.
  typedef Costs (OpLevelCostEstimator::*CostImpl)(const OpContext& op_context)
      const;
  auto wrap = [this](CostImpl impl) -> std::function<Costs(const OpContext&)> {
    return [this, impl](const OpContext& op_context) {
      return (this->*impl)(op_context);
    };
  };

  device_cost_impl_ = {
      {kConv2d, wrap(&OpLevelCostEstimator::PredictConv2D)},
      {kConv2dBackpropFilter,
       wrap(&OpLevelCostEstimator::PredictConv2DBackpropFilter)},
      {kConv2dBackpropInput,
       wrap(&OpLevelCostEstimator::PredictConv2DBackpropInput)},
      {kFusedConv2dBiasActivation,
       wrap(&OpLevelCostEstimator::PredictFusedConv2DBiasActivation)},
      // reuse Conv2D for DepthwiseConv2dNative because the caculation is the
      // same although the actual meaning of the parameters are different. See
      // comments in PredictConv2D and related functions
      {kDepthwiseConv2dNative, wrap(&OpLevelCostEstimator::PredictConv2D)},
      {kDepthwiseConv2dNativeBackpropFilter,
       wrap(&OpLevelCostEstimator::PredictConv2DBackpropFilter)},
      {kDepthwiseConv2dNativeBackpropInput,
       wrap(&OpLevelCostEstimator::PredictConv2DBackpropInput)},
      {kMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kSparseMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kBatchMatMul, wrap(&OpLevelCostEstimator::PredictBatchMatMul)},
      {kQuantizedMatMulV2, wrap(&OpLevelCostEstimator::PredictMatMul)},

      {kNoOp, wrap(&OpLevelCostEstimator::PredictNoOp)},
      {kGuaranteeConst, wrap(&OpLevelCostEstimator::PredictNoOp)},

      {kGather, wrap(&OpLevelCostEstimator::PredictGatherOrSlice)},
      {kGatherV2, wrap(&OpLevelCostEstimator::PredictGatherOrSlice)},
      {kSlice, wrap(&OpLevelCostEstimator::PredictGatherOrSlice)},

      {kPlaceholder, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kIdentity, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kIdentityN, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kRefIdentity, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kStopGradient, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kPreventGradient, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kReshape, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kSqueeze, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kRecv, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kSend, wrap(&OpLevelCostEstimator::PredictIdentity)},

      {kConst, wrap(&OpLevelCostEstimator::PredictVariable)},
      {kVariable, wrap(&OpLevelCostEstimator::PredictVariable)},
      {kVariableV2, wrap(&OpLevelCostEstimator::PredictVariable)},

      {kRank, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kShape, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kShapeN, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kSize, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kMaxPool, wrap(&OpLevelCostEstimator::PredictMaxPool)},
      {kMaxPoolGrad, wrap(&OpLevelCostEstimator::PredictMaxPoolGrad)},
      {kAvgPool, wrap(&OpLevelCostEstimator::PredictAvgPool)},
      {kAvgPoolGrad, wrap(&OpLevelCostEstimator::PredictAvgPoolGrad)},
      {kFusedBatchNorm, wrap(&OpLevelCostEstimator::PredictFusedBatchNorm)},
      {kFusedBatchNormGrad,
       wrap(&OpLevelCostEstimator::PredictFusedBatchNormGrad)},
  };

#define EIGEN_COST(X) Eigen::internal::functor_traits<Eigen::internal::X>::Cost

  // Quantize = apply min and max bounds, multiply by scale factor and round.
  const int quantize_v2_cost =
      EIGEN_COST(scalar_product_op<float>) + EIGEN_COST(scalar_max_op<float>) +
      EIGEN_COST(scalar_min_op<float>) + EIGEN_COST(scalar_round_op<float>);

  elementwise_ops_ = {
      // Unary ops alphabetically sorted
      {"Acos", EIGEN_COST(scalar_acos_op<float>)},
      {"Asin", EIGEN_COST(scalar_asin_op<float>)},
      {"Atan", EIGEN_COST(scalar_atan_op<float>)},
      {"Atan2", EIGEN_COST(scalar_quotient_op<float>) +
                    EIGEN_COST(scalar_atan_op<float>)},
      // For now, we use Eigen cost model for float to int16 cast as an example
      // case; Eigen cost model is zero when src and dst types are identical,
      // and it uses AddCost (1) when different. We may implement a separate
      // cost functions for cast ops, using the actual input and output types.
      {"Cast", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_cast_op<float, int16>>::Cost},
      {"Ceil", EIGEN_COST(scalar_ceil_op<float>)},
      {"Cos", EIGEN_COST(scalar_cos_op<float>)},
      {"Dequantize", EIGEN_COST(scalar_product_op<float>)},
      {"Erf", 1},
      {"Erfc", 1},
      {"Exp", EIGEN_COST(scalar_exp_op<float>)},
      {"Expm1", EIGEN_COST(scalar_expm1_op<float>)},
      {"Floor", EIGEN_COST(scalar_floor_op<float>)},
      {"Inv", EIGEN_COST(scalar_inverse_op<float>)},
      {"InvGrad", 1},
      {"Lgamma", 1},
      {"Log", EIGEN_COST(scalar_log_op<float>)},
      {"Log1p", EIGEN_COST(scalar_log1p_op<float>)},
      {"Neg", EIGEN_COST(scalar_opposite_op<float>)},
      {"QuantizeV2", quantize_v2_cost},
      {"Reciprocal", EIGEN_COST(scalar_inverse_op<float>)},
      {"Rint", 1},
      {"Round", EIGEN_COST(scalar_round_op<float>)},
      {"Rsqrt", EIGEN_COST(scalar_rsqrt_op<float>)},
      {"Sqrt", EIGEN_COST(scalar_sqrt_op<float>)},
      {"Square", EIGEN_COST(scalar_square_op<float>)},
      {"Tanh", EIGEN_COST(scalar_tanh_op<float>)},
      {"Relu", EIGEN_COST(scalar_max_op<float>)},
      {"Sigmoid", EIGEN_COST(scalar_logistic_op<float>)},
      {"QuantizedSigmoid", EIGEN_COST(scalar_logistic_op<float>)},
      {"Sign", EIGEN_COST(scalar_sign_op<float>)},
      {"Sin", EIGEN_COST(scalar_sin_op<float>)},
      {"Tan", EIGEN_COST(scalar_tan_op<float>)},
      // Binary ops alphabetically sorted
      {"Add", EIGEN_COST(scalar_sum_op<float>)},
      {"ApproximateEqual", 1},
      {"BiasAdd", EIGEN_COST(scalar_sum_op<float>)},
      {"QuantizedBiasAdd", EIGEN_COST(scalar_sum_op<float>)},
      {"Div", EIGEN_COST(scalar_quotient_op<float>)},
      {"Equal", 1},
      {"FloorDiv", EIGEN_COST(scalar_quotient_op<float>)},
      {"FloorMod", EIGEN_COST(scalar_mod_op<float>)},
      {"Greater", 1},
      {"GreaterEqual", 1},
      {"Less", 1},
      {"LessEqual", 1},
      {"LogicalAnd", EIGEN_COST(scalar_boolean_and_op)},
      {"LogicalNot", 1},
      {"LogicalOr", EIGEN_COST(scalar_boolean_or_op)},
      {"Maximum", EIGEN_COST(scalar_max_op<float>)},
      {"Minimum", EIGEN_COST(scalar_min_op<float>)},
      {"Mod", EIGEN_COST(scalar_mod_op<float>)},
      {"Mul", EIGEN_COST(scalar_product_op<float>)},
      {"NotEqual", 1},
      {"QuantizedAdd", EIGEN_COST(scalar_sum_op<float>)},
      {"QuantizedMul", EIGEN_COST(scalar_product_op<float>)},
      {"RealDiv", EIGEN_COST(scalar_quotient_op<float>)},
      {"ReluGrad", EIGEN_COST(scalar_max_op<float>)},
      {"SquareDifference", 1},
      {"Sub", EIGEN_COST(scalar_difference_op<float>)},
      {"TruncateDiv", EIGEN_COST(scalar_quotient_op<float>)},
      {"TruncateMod", EIGEN_COST(scalar_mod_op<float>)}};

#undef EIGEN_COST

  // By default, use sum of memory_time and compute_time for execution_time.
  compute_memory_overlap_ = false;
}

Costs OpLevelCostEstimator::PredictCosts(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  auto it = device_cost_impl_.find(op_info.op());
  if (it == device_cost_impl_.end()) {
    if (elementwise_ops_.find(op_info.op()) != elementwise_ops_.end()) {
      return PredictCwiseOp(op_context);
    }

    VLOG(1) << "Missing accurate estimator for op: " << op_info.op();

    return PredictCostOfAnUnknownOp(op_context);
  }

  std::function<Costs(const OpContext&)> estimator = it->second;
  Costs costs = estimator(op_context);
  VLOG(1) << "Operation " << op_info.op() << " takes "
          << costs.execution_time.count() << " ns.";
  return costs;
}

DeviceInfo OpLevelCostEstimator::GetDeviceInfo(
    const DeviceProperties& device) const {
  double gflops = -1;
  double gb_per_sec = -1;

  if (device.type() == "CPU") {
    // Check if vector instructions are available, and refine performance
    // prediction based on this.
    // Frequencies are stored in MHz in the DeviceProperties.
    gflops = device.num_cores() * device.frequency() * 1e-3;
    if (gb_per_sec < 0) {
      if (device.bandwidth() > 0) {
        gb_per_sec = device.bandwidth() / 1e6;
      } else {
        gb_per_sec = 32;
      }
    }
  } else if (device.type() == "GPU") {
    const string architecture = device.environment().at("architecture");
    int cores_per_multiprocessor;
    if (architecture < "3") {
      // Fermi
      cores_per_multiprocessor = 32;
    } else if (architecture < "4") {
      // Kepler
      cores_per_multiprocessor = 192;
    } else if (architecture < "6") {
      // Maxwell
      cores_per_multiprocessor = 128;
    } else {
      // Pascal (compute capability version 6) and Volta (compute capability
      // version 7)
      cores_per_multiprocessor = 64;
    }
    gflops = device.num_cores() * device.frequency() * 1e-3 *
             cores_per_multiprocessor * kOpsPerMac;
    if (device.bandwidth() > 0) {
      gb_per_sec = device.bandwidth() / 1e6;
    } else {
      gb_per_sec = 100;
    }
  }
  VLOG(1) << "Device: " << device.type() << " gflops: " << gflops
          << " gb_per_sec: " << gb_per_sec;

  DCHECK_LT(0, gflops) << device.DebugString();
  DCHECK_LT(0, gb_per_sec) << device.DebugString();

  return DeviceInfo(gflops, gb_per_sec);
}

Costs OpLevelCostEstimator::PredictCwiseOp(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  // For unary or binary element-wise operations, op count is the element count
  // of any input. We use the count for the largest input here to be more robust
  // in case that the shape is unknown or partially known for other input.
  int64 op_count = CalculateLargestInputCount(op_info, &found_unknown_shapes);
  // If output shape is available, try use the element count calcuated from
  // that.
  if (op_info.outputs_size() > 0) {
    op_count = std::max(
        op_count,
        CalculateTensorElementCount(op_info.outputs(0), &found_unknown_shapes));
  }
  // For binary ops, calculate the output shape possibly resulting from
  // broadcasting.
  if (op_info.inputs_size() >= 2) {
    op_count =
        std::max(op_count, CwiseOutputElementCount(op_info.inputs(0).shape(),
                                                   op_info.inputs(1).shape()));
  }

  int op_cost = 1;
  bool is_known_elementwise_op = false;
  auto it = elementwise_ops_.find(op_info.op());
  if (it != elementwise_ops_.end()) {
    op_cost = it->second;
    is_known_elementwise_op = true;
  } else {
    LOG(WARNING) << "Not a cwise op: " << op_info.op();
  }

  Costs costs = PredictOpCountBasedCost(op_count * op_cost, op_info);
  if (found_unknown_shapes || !is_known_elementwise_op) {
    costs.inaccurate = true;
  }
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictCostOfAnUnknownOp(
    const OpContext& op_context) const {
  // Don't assume the operation is cwise, return cost based on input/output size
  // and admit that it is inaccurate...
  auto costs = PredictOpCountBasedCost(0, op_context.op_info);
  costs.inaccurate = true;
  return costs;
}

Costs OpLevelCostEstimator::PredictOpCountBasedCost(
    double operations, const OpInfo& op_info) const {
  bool unknown_shapes = false;
  const double input_size = CalculateInputSize(op_info, &unknown_shapes);
  const double output_size = CalculateOutputSize(op_info, &unknown_shapes);
  Costs costs =
      PredictOpCountBasedCost(operations, input_size, output_size, op_info);
  costs.inaccurate = unknown_shapes;
  costs.num_ops_with_unknown_shapes = unknown_shapes;
  costs.max_memory = output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictOpCountBasedCost(
    double operations, double input_io_bytes, double output_io_bytes,
    const OpInfo& op_info) const {
  double total_io_bytes = input_io_bytes + output_io_bytes;
  const DeviceInfo device_info = GetDeviceInfo(op_info.device());
  if (device_info.gigaops <= 0 || device_info.gb_per_sec <= 0 ||
      device_info.intermediate_read_gb_per_sec <= 0 ||
      device_info.intermediate_write_gb_per_sec <= 0) {
    VLOG(1) << "BAD DEVICE. Op:" << op_info.op()
            << " device type:" << op_info.device().type()
            << " device model:" << op_info.device().model();
  }

  Costs::NanoSeconds compute_cost(std::ceil(operations / device_info.gigaops));
  VLOG(1) << "Op:" << op_info.op() << " GOps:" << operations / 1e9
          << " Compute Time (ns):" << compute_cost.count();

  Costs::NanoSeconds memory_cost(
      std::ceil(total_io_bytes / device_info.gb_per_sec));
  VLOG(1) << "Op:" << op_info.op() << " Size (KB):" << (total_io_bytes) / 1e3
          << " Memory Time (ns):" << memory_cost.count();

  // Check if bytes > 0.  If it's not and the bandwidth is set to infinity
  // then the result would be undefined.
  double intermediate_read_time =
      (input_io_bytes > 0)
          ? std::ceil(input_io_bytes / device_info.intermediate_read_gb_per_sec)
          : 0;

  double intermediate_write_time =
      (output_io_bytes > 0)
          ? std::ceil(output_io_bytes /
                      device_info.intermediate_write_gb_per_sec)
          : 0;

  Costs::NanoSeconds intermediate_memory_cost =
      compute_memory_overlap_
          ? std::max(intermediate_read_time, intermediate_write_time)
          : (intermediate_read_time + intermediate_write_time);
  VLOG(1) << "Op:" << op_info.op() << " Size (KB):" << (total_io_bytes) / 1e3
          << " Intermediate Memory Time (ns):"
          << intermediate_memory_cost.count();

  Costs costs;
  costs.compute_time = compute_cost;
  costs.memory_time = memory_cost;
  costs.intermediate_memory_time = intermediate_memory_cost;
  CombineCostsAndUpdateExecutionTime(&costs);
  return costs;
}

int64 OpLevelCostEstimator::CountConv2DOperations(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  return CountConv2DOperations(op_info, nullptr, found_unknown_shapes);
}

// Helper to translate the positional arguments into named fields.
OpLevelCostEstimator::ConvolutionDimensions
OpLevelCostEstimator::ConvolutionDimensionsFromInputs(
    const TensorShapeProto& original_image_shape,
    const TensorShapeProto& original_filter_shape, const OpInfo& op_info,
    bool* found_unknown_shapes) {
  VLOG(2) << "op features: " << op_info.DebugString();
  VLOG(2) << "Original image shape: " << original_image_shape.DebugString();
  VLOG(2) << "Original filter shape: " << original_filter_shape.DebugString();
  auto image_shape =
      MaybeGetMinimumShape(original_image_shape, 4, found_unknown_shapes);
  auto filter_shape =
      MaybeGetMinimumShape(original_filter_shape, 4, found_unknown_shapes);
  VLOG(2) << "Image shape: " << image_shape.DebugString();
  VLOG(2) << "Filter shape: " << filter_shape.DebugString();

  int x_index, y_index, channel_index;
  const string& data_format = GetDataFormat(op_info);
  if (data_format == "NCHW") {
    x_index = 2;
    y_index = 3;
    channel_index = 1;
  } else {
    // Use NHWC.
    x_index = 1;
    y_index = 2;
    channel_index = 3;
  }
  const string& filter_format = GetFilterFormat(op_info);
  int filter_x_index, filter_y_index, in_channel_index, out_channel_index;
  if (filter_format == "HWIO") {
    filter_x_index = 0;
    filter_y_index = 1;
    in_channel_index = 2;
    out_channel_index = 3;
  } else {
    // Use OIHW
    filter_x_index = 2;
    filter_y_index = 3;
    in_channel_index = 1;
    out_channel_index = 0;
  }
  int64 batch = image_shape.dim(0).size();
  int64 ix = image_shape.dim(x_index).size();
  int64 iy = image_shape.dim(y_index).size();
  int64 iz = image_shape.dim(channel_index).size();
  int64 kx = filter_shape.dim(filter_x_index).size();
  int64 ky = filter_shape.dim(filter_y_index).size();
  std::vector<int64> strides = GetStrides(op_info);
  const auto padding = GetPadding(op_info);
  int64 sx = strides[x_index];
  int64 sy = strides[y_index];
  int64 ox = GetOutputSize(ix, kx, sx, padding);
  int64 oy = GetOutputSize(iy, ky, sy, padding);
  int64 oz = filter_shape.dim(out_channel_index).size();
  // Only check equality when both sizes are known (in other words, when
  // neither is set to a minimum dimension size of 1).
  if (iz != 1 && filter_shape.dim(in_channel_index).size() != 1) {
    CHECK_EQ(iz, filter_shape.dim(in_channel_index).size());
  } else {
    iz = std::max<int64>(iz, filter_shape.dim(in_channel_index).size());
  }
  OpLevelCostEstimator::ConvolutionDimensions conv_dims = {
      batch, ix, iy, iz, kx, ky, oz, ox, oy, sx, sy, padding};

  VLOG(1) << "Batch Size:" << batch;
  VLOG(1) << "Image Dims:" << ix << "," << iy;
  VLOG(1) << "Input Features:" << iz;
  VLOG(1) << "Kernel Dims:" << kx << "," << ky;
  VLOG(1) << "Output Features:" << oz;
  VLOG(1) << "Output Dims:" << ox << "," << oy;
  VLOG(1) << "Strides:" << sx << "," << sy;
  VLOG(1) << "Padding:" << (padding == Padding::VALID ? "VALID" : "SAME");
  return conv_dims;
}

int64 OpLevelCostEstimator::CountConv2DOperations(
    const OpInfo& op_info, ConvolutionDimensions* conv_info,
    bool* found_unknown_shapes) const {
  DCHECK(op_info.op() == kConv2d || op_info.op() == kDepthwiseConv2dNative)
      << "Invalid Operation: not Conv2D nor DepthwiseConv2dNative";

  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      op_info.inputs(0).shape(), op_info.inputs(1).shape(), op_info,
      found_unknown_shapes);

  //  in DepthwiseConv2dNative conv_dims.oz is actually the channel depth
  //  multiplier; The effective output channel depth oz_effective is
  //  conv_dims.iz * conv_dims.oz. thus # ops = N x H x W x oz_effective x 2RS.
  //  Compare to Conv2D where # ops =  N x H x W x iz x oz x 2RS,
  //  oz = oz_effective,  then Conv2D_ops / Depthwise_conv2d_native_ops = iz.
  int64 ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  if (op_info.op() == kConv2d) {
    ops *= conv_dims.iz * conv_dims.oz;
  } else {
    // To ensure output tensor dims to be correct for DepthwiseConv2DNative,
    // although ops are the same as Conv2D.
    conv_dims.oz *= conv_dims.iz;
    ops *= conv_dims.oz;
  }
  ops *= kOpsPerMac;

  if (conv_info != nullptr) {
    *conv_info = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  return CountMatMulOperations(op_info, nullptr, found_unknown_shapes);
}

// TODO(nishantpatil): Create separate estimator for Sparse Matmul
int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_info, MatMulDimensions* mat_mul,
    bool* found_unknown_shapes) const {
  double ops = 0;

  if (op_info.inputs_size() < 2) {
    LOG(ERROR) << "Need 2 inputs but got " << op_info.inputs_size();
    // TODO(pcma): Try to separate invalid inputs from unknown shapes
    *found_unknown_shapes = true;
    return 0;
  }

  auto& a_matrix = op_info.inputs(0);
  auto& b_matrix = op_info.inputs(1);

  bool transpose_a = false;
  bool transpose_b = false;

  double m_dim, n_dim, k_dim, k_dim_b = 0;

  for (const auto& item : op_info.attr()) {
    VLOG(1) << "Key:" << item.first
            << " Value:" << SummarizeAttrValue(item.second);
    if (item.first == "transpose_a" && item.second.b() == true)
      transpose_a = true;
    if (item.first == "transpose_b" && item.second.b() == true)
      transpose_b = true;
  }
  VLOG(1) << "transpose_a:" << transpose_a;
  VLOG(1) << "transpose_b:" << transpose_b;
  auto a_matrix_shape =
      MaybeGetMinimumShape(a_matrix.shape(), 2, found_unknown_shapes);
  auto b_matrix_shape =
      MaybeGetMinimumShape(b_matrix.shape(), 2, found_unknown_shapes);
  if (transpose_a) {
    m_dim = a_matrix_shape.dim(1).size();
    k_dim = a_matrix_shape.dim(0).size();
  } else {
    m_dim = a_matrix_shape.dim(0).size();
    k_dim = a_matrix_shape.dim(1).size();
  }
  if (transpose_b) {
    k_dim_b = b_matrix_shape.dim(1).size();
    n_dim = b_matrix_shape.dim(0).size();
  } else {
    k_dim_b = b_matrix_shape.dim(0).size();
    n_dim = b_matrix_shape.dim(1).size();
  }

  VLOG(1) << "M, N, K: " << m_dim << "," << n_dim << "," << k_dim;
  // Only check equality when both sizes are known (in other words, when
  // neither is set to a minimum dimension size of 1).
  if (k_dim_b != 1 && k_dim != 1 && k_dim_b != k_dim) {
    LOG(ERROR) << "Incompatible Matrix dimensions";
    return ops;
  } else {
    // One of k_dim and k_dim_b might be 1 (mininum dimension size).
    k_dim = std::max(k_dim, k_dim_b);
  }

  ops = m_dim * n_dim * k_dim * 2;
  VLOG(1) << "Operations for Matmul: " << ops;

  if (mat_mul != nullptr) {
    mat_mul->m = m_dim;
    mat_mul->n = n_dim;
    mat_mul->k = k_dim;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountBatchMatMulOperations(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  if (op_info.op() != kBatchMatMul) {
    LOG(ERROR) << "Invalid Operation: " << op_info.op();
    // TODO(pcma): Try to separate invalid inputs from unknown shapes
    *found_unknown_shapes = true;
    return 0;
  }
  if (op_info.inputs_size() != 2) {
    LOG(ERROR) << "Expected 2 inputs but got " << op_info.inputs_size();
    // TODO(pcma): Try to separate invalid inputs from unknown shapes
    *found_unknown_shapes = true;
    return 0;
  }

  double ops = 0;
  const auto& a_input = op_info.inputs(0);
  const auto& b_input = op_info.inputs(1);

  // BatchMatMul requires inputs of at least matrix shape (rank 2).
  // The two most minor dimensions of each input are matrices that
  // need to be multiplied together. The other dimensions determine
  // the number of such MatMuls.  For example, if the BatchMatMul has
  // inputs of shape:
  //   a_input_shape = [2, 3, 4, 5]
  //   b_input_shape = [2, 3, 5, 6]
  // then there are 2*3 = 6 MatMuls of dimensions m = 4, k = 5, n = 6
  // in this BatchMatMul.
  const int matrix_rank = 2;

  bool a_input_shape_unknown = false;
  bool b_input_shape_unknown = false;

  TensorShapeProto a_input_shape = MaybeGetMinimumShape(
      a_input.shape(), std::max(matrix_rank, a_input.shape().dim_size()),
      &a_input_shape_unknown);
  TensorShapeProto b_input_shape = MaybeGetMinimumShape(
      b_input.shape(), std::max(matrix_rank, b_input.shape().dim_size()),
      &b_input_shape_unknown);

  *found_unknown_shapes = a_input_shape_unknown || b_input_shape_unknown ||
                          (a_input.shape().dim_size() < matrix_rank) ||
                          (b_input.shape().dim_size() < matrix_rank);

  // Compute the number of matmuls as the max indicated at each dimension
  // by either input. Note that the shapes do not have to have
  // the same rank due to incompleteness.
  TensorShapeProto* bigger_rank_shape = &a_input_shape;
  TensorShapeProto* smaller_rank_shape = &b_input_shape;
  if (b_input_shape.dim_size() > a_input_shape.dim_size()) {
    bigger_rank_shape = &b_input_shape;
    smaller_rank_shape = &a_input_shape;
  }
  int num_matmuls = 1;
  for (int b_i = 0,
           s_i = smaller_rank_shape->dim_size() - bigger_rank_shape->dim_size();
       b_i < bigger_rank_shape->dim_size() - matrix_rank; ++b_i, ++s_i) {
    int b_dim = bigger_rank_shape->dim(b_i).size();
    int s_dim = 1;
    if (s_i >= 0) {
      s_dim = smaller_rank_shape->dim(s_i).size();
    }
    num_matmuls *= std::max(b_dim, s_dim);
  }

  // Build the MatMul. Note that values are ignored here since we are just
  // counting ops (e.g. only shapes matter).
  OpInfo matmul_op_info;
  matmul_op_info.set_op("MatMul");

  AttrValue transpose_a;
  transpose_a.set_b(false);
  if (op_info.attr().find("adj_x") != op_info.attr().end()) {
    transpose_a.set_b(op_info.attr().at("adj_x").b());
  }
  (*matmul_op_info.mutable_attr())["transpose_a"] = transpose_a;

  AttrValue transpose_b;
  transpose_b.set_b(false);
  if (op_info.attr().find("adj_y") != op_info.attr().end()) {
    transpose_b.set_b(op_info.attr().at("adj_y").b());
  }
  (*matmul_op_info.mutable_attr())["transpose_b"] = transpose_b;

  OpInfo::TensorProperties* a_matrix = matmul_op_info.add_inputs();
  a_matrix->set_dtype(a_input.dtype());
  TensorShapeProto* a_matrix_shape = a_matrix->mutable_shape();
  for (int i = std::max(0, a_input_shape.dim_size() - matrix_rank);
       i < a_input_shape.dim_size(); ++i) {
    *(a_matrix_shape->add_dim()) = a_input_shape.dim(i);
  }

  OpInfo::TensorProperties* b_matrix = matmul_op_info.add_inputs();
  b_matrix->set_dtype(b_input.dtype());
  TensorShapeProto* b_matrix_shape = b_matrix->mutable_shape();
  for (int i = std::max(0, b_input_shape.dim_size() - matrix_rank);
       i < b_input_shape.dim_size(); ++i) {
    *(b_matrix_shape->add_dim()) = b_input_shape.dim(i);
  }

  for (int i = 0; i < num_matmuls; ++i) {
    bool matmul_unknown_shapes = false;
    ops += CountMatMulOperations(matmul_op_info, &matmul_unknown_shapes);
    *found_unknown_shapes |= matmul_unknown_shapes;
  }
  return ops;
}

bool GetTensorShapeProtoFromTensorProto(const TensorProto& tensor_proto,
                                        TensorShapeProto* tensor_shape_proto) {
  tensor_shape_proto->Clear();
  // First convert TensorProto into Tensor class so that it correctly parses
  // data values within TensorProto (whether it's in int_val, int64_val,
  // tensor_content, or anything.
  Tensor tensor(tensor_proto.dtype());
  if (!tensor.FromProto(tensor_proto)) {
    LOG(WARNING) << "GetTensorShapeProtoFromTensorProto() -- "
                 << "failed to parse TensorProto: "
                 << tensor_proto.DebugString();
    return false;
  }
  if (tensor.dims() != 1) {
    LOG(WARNING) << "GetTensorShapeProtoFromTensorProto() -- "
                 << "tensor is not 1D: " << tensor.dims();
    return false;
  }
  // Then, convert it back to TensorProto using AsProtoField, which makes sure
  // the data is in int_val, int64_val, or such repeated data fields, not in
  // tensor_content.
  TensorProto temp_tensor;
  tensor.AsProtoField(&temp_tensor);

#define TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO(type)        \
  do {                                                   \
    for (const auto& value : temp_tensor.type##_val()) { \
      tensor_shape_proto->add_dim()->set_size(value);    \
    }                                                    \
  } while (0)

  if (tensor.dtype() == DT_INT32 || tensor.dtype() == DT_INT16 ||
      tensor.dtype() == DT_INT8 || tensor.dtype() == DT_UINT8) {
    TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO(int);
  } else if (tensor.dtype() == DT_INT64) {
    TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO(int64);
  } else if (tensor.dtype() == DT_UINT32) {
    TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO(uint32);
  } else if (tensor.dtype() == DT_UINT64) {
    TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO(uint64);
  } else {
    LOG(WARNING) << "GetTensorShapeProtoFromTensorProto() -- "
                 << "Unsupported dtype: " << tensor.dtype();
    return false;
  }
#undef TENSOR_VALUES_TO_TENSOR_SHAPE_PROTO

  return true;
}

// TODO(cliffy): Dedup this method and CountConv2DBackpropFilterOperations.
int64 OpLevelCostEstimator::CountConv2DBackpropInputOperations(
    const OpInfo& op_info, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;

  DCHECK(op_info.op() == kConv2dBackpropInput ||
         op_info.op() == kDepthwiseConv2dNativeBackpropInput)
      << "Invalid Operation: not kConv2dBackpropInput nor"
         "kDepthwiseConv2dNativeBackpropInput";

  if (op_info.inputs_size() < 2) {
    // TODO(pcma): Try to separate invalid inputs from unknown shapes
    *found_unknown_shapes = true;
    return ops;
  }

  TensorShapeProto input_shape;
  bool shape_found = false;
  if (op_info.inputs(0).has_value()) {
    const TensorProto& value = op_info.inputs(0).value();
    shape_found = GetTensorShapeProtoFromTensorProto(value, &input_shape);
  }
  if (!shape_found && op_info.outputs_size() == 1) {
    input_shape = op_info.outputs(0).shape();
    shape_found = true;
  }
  if (!shape_found) {
    // Set the minimum filter size that's feasible.
    input_shape.Clear();
    for (int i = 0; i < 4; ++i) {
      input_shape.add_dim()->set_size(1);
    }
    *found_unknown_shapes = true;
  }

  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      input_shape, op_info.inputs(1).shape(), op_info, found_unknown_shapes);

  ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  if (op_info.op() == kConv2dBackpropInput) {
    ops *= conv_dims.iz * conv_dims.oz;
  } else {
    // conv_dims always use forward path definition regardless
    conv_dims.oz *= conv_dims.iz;
    ops *= conv_dims.oz;
  }
  ops *= kOpsPerMac;

  VLOG(1) << "Operations for" << op_info.op() << "  " << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountConv2DBackpropFilterOperations(
    const OpInfo& op_info, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;

  DCHECK(op_info.op() == kConv2dBackpropFilter ||
         op_info.op() == kDepthwiseConv2dNativeBackpropFilter)
      << "Invalid Operation: not kConv2dBackpropFilter nor"
         "kDepthwiseConv2dNativeBackpropFilter";

  TensorShapeProto filter_shape;
  bool shape_found = false;
  if (op_info.inputs_size() >= 2 && op_info.inputs(1).has_value()) {
    const TensorProto& value = op_info.inputs(1).value();
    shape_found = GetTensorShapeProtoFromTensorProto(value, &filter_shape);
  }
  if (!shape_found && op_info.outputs_size() == 1) {
    filter_shape = op_info.outputs(0).shape();
    shape_found = true;
  }
  if (!shape_found) {
    // Set the minimum filter size that's feasible.
    filter_shape.Clear();
    for (int i = 0; i < 4; ++i) {
      filter_shape.add_dim()->set_size(1);
    }
    *found_unknown_shapes = true;
  }

  if (op_info.inputs_size() < 1) {
    // TODO(pcma): Try to separate invalid inputs from unknown shapes
    *found_unknown_shapes = true;
    return ops;
  }
  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      op_info.inputs(0).shape(), filter_shape, op_info, found_unknown_shapes);

  ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  if (op_info.op() == kConv2dBackpropFilter) {
    ops *= conv_dims.iz * conv_dims.oz;
  } else {
    // conv_dims always use forward path definition regardless
    conv_dims.oz *= conv_dims.iz;
    ops *= conv_dims.oz;
  }
  ops *= kOpsPerMac;
  VLOG(1) << "Operations for" << op_info.op() << "  " << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CalculateTensorElementCount(
    const OpInfo::TensorProperties& tensor, bool* found_unknown_shapes) const {
  VLOG(2) << "   with " << DataTypeString(tensor.dtype()) << " tensor of shape "
          << tensor.shape().DebugString();
  int64 tensor_size = 1;
  int num_dims = std::max(1, tensor.shape().dim_size());
  auto tensor_shape =
      MaybeGetMinimumShape(tensor.shape(), num_dims, found_unknown_shapes);
  for (const auto& dim : tensor_shape.dim()) {
    tensor_size *= dim.size();
  }
  return tensor_size;
}

int64 OpLevelCostEstimator::CalculateTensorSize(
    const OpInfo::TensorProperties& tensor, bool* found_unknown_shapes) const {
  int64 count = CalculateTensorElementCount(tensor, found_unknown_shapes);
  int size = DataTypeSize(BaseType(tensor.dtype()));
  VLOG(2) << "Count: " << count << " DataTypeSize: " << size;
  return count * size;
}

int64 OpLevelCostEstimator::CalculateInputSize(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  int64 total_input_size = 0;
  for (auto& input : op_info.inputs()) {
    int64 input_size = CalculateTensorSize(input, found_unknown_shapes);
    total_input_size += input_size;
    VLOG(1) << "Input Size: " << input_size
            << " Total Input Size:" << total_input_size;
  }
  return total_input_size;
}

int64 OpLevelCostEstimator::CalculateLargestInputCount(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  int64 largest_input_count = 0;
  for (auto& input : op_info.inputs()) {
    int64 input_count =
        CalculateTensorElementCount(input, found_unknown_shapes);
    if (input_count > largest_input_count) {
      largest_input_count = input_count;
    }
    VLOG(1) << "Input Count: " << input_count
            << " Largest Input Count:" << largest_input_count;
  }
  return largest_input_count;
}

int64 OpLevelCostEstimator::CalculateOutputSize(
    const OpInfo& op_info, bool* found_unknown_shapes) const {
  int64 total_output_size = 0;
  // use float as default for calculations
  for (const auto& output : op_info.outputs()) {
    DataType dt = output.dtype();
    const auto& original_output_shape = output.shape();
    int64 output_size = DataTypeSize(BaseType(dt));
    int num_dims = std::max(1, original_output_shape.dim_size());
    auto output_shape = MaybeGetMinimumShape(original_output_shape, num_dims,
                                             found_unknown_shapes);
    for (const auto& dim : output_shape.dim()) {
      output_size *= dim.size();
    }
    total_output_size += output_size;
    VLOG(1) << "Output Size: " << output_size
            << " Total Output Size:" << total_output_size;
  }
  return total_output_size;
}

Costs OpLevelCostEstimator::PredictConv2D(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountConv2DOperations(op_info, &found_unknown_shapes), op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackpropInput(
    const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackpropInputOperations(
                                  op_info, nullptr, &found_unknown_shapes),
                              op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackpropFilter(
    const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackpropFilterOperations(
                                  op_info, nullptr, &found_unknown_shapes),
                              op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictFusedConv2DBiasActivation(
    const OpContext& op_context) const {
  // FusedConv2DBiasActivation computes a fused kernel which implements:
  // 2D convolution, adds side input with separate scaling on convolution and
  // side inputs, then adds bias, and finally applies the ReLU activation
  // function to the result:
  //
  // Input -> Conv2D  ->  Add  -> BiasAdd  -> ReLU
  //            ^          ^         ^
  //          Filter   Side Input   Bias
  //
  // Note that when adding the side input, the operation multiplies the output
  // of Conv2D by conv_input_scale, confusingly, and the side_input by
  // side_input_scale.
  //
  // Note that in the special case that side_input_scale is 0, which we infer
  // from side_input having dimensions [], we skip that addition operation.
  //
  // For more information, see
  // contrib/fused_conv/kernels/fused_conv2d_bias_activation_op.cc

  // TODO(yaozhang): Support other data formats (NCHW_VECT_C, NHWC_VECT_W) and
  // filter formats (OIHW_VECT_I).
  string data_format = GetDataFormat(op_context.op_info);
  if (data_format != "NCHW" && data_format != "NHWC") {
    LOG(WARNING) << "unsupported data format: " << data_format;
    Costs cost = Costs::ZeroCosts();
    cost.inaccurate = true;
    return cost;
  }
  string filter_format = GetFilterFormat(op_context.op_info);
  if (filter_format != "HWIO" && filter_format != "OIHW") {
    LOG(WARNING) << "unsupported filter format: " << filter_format;
    Costs cost = Costs::ZeroCosts();
    cost.inaccurate = true;
    return cost;
  }

  auto& conv_input = op_context.op_info.inputs(0);
  auto& filter = op_context.op_info.inputs(1);
  auto& bias = op_context.op_info.inputs(2);
  auto& side_input = op_context.op_info.inputs(3);
  auto& conv_input_scale = op_context.op_info.inputs(4);
  auto& side_input_scale = op_context.op_info.inputs(5);

  // Manually compute our convolution dimensions.
  bool found_unknown_shapes = false;
  auto dims = ConvolutionDimensionsFromInputs(
      conv_input.shape(), filter.shape(), op_context.op_info,
      &found_unknown_shapes);

  // Construct the shape of our output tensor from our convolution dimensions
  // and format, as it may not be available yet.
  // TODO(varomodt): should we centralize the Conv2D input/output shapes?
  OpInfo::TensorProperties output;
  if (data_format == "NCHW") {
    output = DescribeTensor(DT_FLOAT, {dims.batch, dims.oz, dims.ox, dims.oy});
  } else if (data_format == "NHWC") {
    output = DescribeTensor(DT_FLOAT, {dims.batch, dims.ox, dims.oy, dims.oz});
  }

  // Add the operations the fused op always computes.
  std::vector<OpContext> component_ops = {
      FusedChildContext(op_context, "Conv2D", output, {conv_input, filter}),
      FusedChildContext(op_context, "Mul", output, {output, conv_input_scale}),
      FusedChildContext(op_context, "BiasAdd", output, {output, bias}),
      FusedChildContext(op_context, "Relu", output, {output})};

  // Add our side_input iff it's non-empty.
  if (side_input.shape().dim_size() > 0) {
    component_ops.push_back(FusedChildContext(op_context, "Mul", side_input,
                                              {side_input, side_input_scale}));
    component_ops.push_back(
        FusedChildContext(op_context, "Add", output, {side_input, output}));
  }

  // Construct an op_context which definitely has our output shape.
  auto op_context_with_output = op_context;
  op_context_with_output.op_info.mutable_outputs()->Clear();
  *op_context_with_output.op_info.mutable_outputs()->Add() = output;

  // Construct component operations and run the cost computation.
  auto costs = PredictFusedOp(op_context_with_output, component_ops);
  costs.inaccurate |= found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = costs.inaccurate;
  return costs;
}

Costs OpLevelCostEstimator::PredictMatMul(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountMatMulOperations(op_info, &found_unknown_shapes), op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictNoOp(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  VLOG(1) << "Op:" << op_info.op() << " Execution Time 0 (ns)";
  return Costs::ZeroCosts();
}

Costs OpLevelCostEstimator::PredictIdentity(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  VLOG(1) << "Op:" << op_info.op() << " Execution Time 0 (ns)";
  Costs result = Costs::ZeroCosts();
  result.max_memory = CalculateOutputSize(op_info, &result.inaccurate);
  result.num_ops_with_unknown_shapes = result.inaccurate;
  // Assign the minimum amount of time we can represent to the identity op since
  // it tends to be really cheap.
  result.compute_time = kMinComputeTime;
  result.execution_time = result.compute_time;
  return result;
}

Costs OpLevelCostEstimator::PredictVariable(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  VLOG(1) << "Op:" << op_info.op() << " Execution Time 0 (ns)";
  Costs result = Costs::ZeroCosts();
  result.persistent_memory = CalculateOutputSize(op_info, &result.inaccurate);
  result.num_ops_with_unknown_shapes = result.inaccurate;

  result.compute_time = kMinComputeTime;
  result.execution_time = result.execution_time;
  return result;
}

Costs OpLevelCostEstimator::PredictBatchMatMul(
    const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  bool found_unknown_shapes = false;
  Costs costs = PredictOpCountBasedCost(
      CountBatchMatMulOperations(op_info, &found_unknown_shapes), op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictMetadata(const OpContext& op_context) const {
  const auto& op_info = op_context.op_info;
  Costs costs = Costs::ZeroCosts();
  costs.max_memory = CalculateOutputSize(op_info, &costs.inaccurate);
  costs.num_ops_with_unknown_shapes = costs.inaccurate;
  // Metadata operations are so cheap we assume they take the minimum amount of
  // time we can represent (1 ns).
  costs.compute_time = kMinComputeTime;
  costs.execution_time = costs.compute_time;

  return costs;
}

Costs OpLevelCostEstimator::PredictGatherOrSlice(
    const OpContext& op_context) const {
  // Gather & Slice ops can have a very large input, but only access a small
  // part of it. For these op the size of the output determines the memory cost.
  const auto& op_info = op_context.op_info;

  const int inputs_needed = op_info.op() == "Slice" ? 3 : 2;
  if (op_info.outputs_size() == 0 || op_info.inputs_size() < inputs_needed) {
    Costs costs = Costs::ZeroCosts();
    costs.inaccurate = true;
    return costs;
  }

  bool unknown_shapes = false;

  // Each output element is a copy of some element from input.
  // For roofline estimate we assume each copy has a unit cost.
  const int64 op_count =
      CalculateTensorElementCount(op_info.outputs(0), &unknown_shapes);

  const double output_size = CalculateOutputSize(op_info, &unknown_shapes);
  double input_size = output_size;
  if (op_info.op() == "Slice") {
    // Add 'begin' & 'size' tensors sizes.
    input_size +=
        CalculateTensorElementCount(op_info.inputs(1), &unknown_shapes) +
        CalculateTensorElementCount(op_info.inputs(2), &unknown_shapes);
  } else {
    // Assuming this is "Gather" or "GatherV2" op, add 'indices' size.
    input_size +=
        CalculateTensorElementCount(op_info.inputs(1), &unknown_shapes);
  }

  Costs costs =
      PredictOpCountBasedCost(op_count, input_size, output_size, op_info);
  costs.inaccurate = unknown_shapes;
  costs.num_ops_with_unknown_shapes = unknown_shapes;
  costs.max_memory = output_size;

  return costs;
}

Costs OpLevelCostEstimator::PredictFusedOp(
    const OpContext& op_context,
    const std::vector<OpContext>& fused_op_contexts) const {
  // Note that PredictOpCountBasedCost will get the correct memory_time from
  // the node's inputs and outputs; but we don't want to have to re-implement
  // the logic for computing the operation count of each of our component
  // operations here; so we simply add the compute times of each component
  // operation, then update the execution time.
  Costs fused_cost = PredictOpCountBasedCost(0, op_context.op_info);

  fused_cost.compute_time = 0;
  fused_cost.inaccurate = false;
  for (auto& fused_op : fused_op_contexts) {
    auto op_cost = PredictCosts(fused_op);

    fused_cost.compute_time += op_cost.compute_time;
    fused_cost.inaccurate |= op_cost.inaccurate;
    fused_cost.intermediate_memory_time += op_cost.intermediate_memory_time;
  }

  CombineCostsAndUpdateExecutionTime(&fused_cost);
  return fused_cost;
}

/* static */
OpContext OpLevelCostEstimator::FusedChildContext(
    const OpContext& parent, const string& op_name,
    const OpInfo::TensorProperties& output,
    const std::vector<OpInfo::TensorProperties>& inputs) {
  // Setup the base parameters of our new context.
  OpContext new_context;
  new_context.name = op_name;
  new_context.device_name = parent.device_name;
  new_context.op_info = parent.op_info;
  new_context.op_info.set_op(op_name);

  // Setup the inputs of our new context.
  new_context.op_info.mutable_inputs()->Clear();
  for (const auto& input : inputs) {
    *new_context.op_info.mutable_inputs()->Add() = input;
  }

  // Setup the output of our new context.
  new_context.op_info.mutable_outputs()->Clear();
  *new_context.op_info.mutable_outputs()->Add() = output;

  return new_context;
}

/* static */
OpInfo::TensorProperties OpLevelCostEstimator::DescribeTensor(
    DataType type, const std::vector<int64>& dims) {
  OpInfo::TensorProperties ret;
  ret.set_dtype(type);

  auto shape = ret.mutable_shape();
  for (const int dim : dims) {
    shape->add_dim()->set_size(dim);
  }

  return ret;
}

/* static */
OpLevelCostEstimator::ConvolutionDimensions
OpLevelCostEstimator::OpDimensionsFromInputs(
    const TensorShapeProto& original_image_shape, const OpInfo& op_info,
    bool* found_unknown_shapes) {
  VLOG(2) << "op features: " << op_info.DebugString();
  VLOG(2) << "Original image shape: " << original_image_shape.DebugString();
  auto image_shape =
      MaybeGetMinimumShape(original_image_shape, 4, found_unknown_shapes);
  VLOG(2) << "Image shape: " << image_shape.DebugString();

  int x_index, y_index, channel_index;
  const string& data_format = GetDataFormat(op_info);
  if (data_format == "NCHW") {
    x_index = 2;
    y_index = 3;
    channel_index = 1;
  } else {
    x_index = 1;
    y_index = 2;
    channel_index = 3;
  }
  int64 batch = image_shape.dim(0).size();
  int64 ix = image_shape.dim(x_index).size();
  int64 iy = image_shape.dim(y_index).size();
  int64 iz = image_shape.dim(channel_index).size();

  // Note that FusedBatchNorm doesn't have ksize attr, but GetKernelSize returns
  // {1, 1, 1, 1} in that case.
  std::vector<int64> ksize = GetKernelSize(op_info);
  int64 kx = ksize[x_index];
  int64 ky = ksize[y_index];

  std::vector<int64> strides = GetStrides(op_info);
  int64 sx = strides[x_index];
  int64 sy = strides[y_index];
  const auto padding = GetPadding(op_info);

  int64 ox = GetOutputSize(ix, kx, sx, padding);
  int64 oy = GetOutputSize(iy, ky, sy, padding);
  int64 oz = iz;

  OpLevelCostEstimator::ConvolutionDimensions conv_dims = {
      batch, ix, iy, iz, kx, ky, oz, ox, oy, sx, sy, padding};
  return conv_dims;
}

Costs OpLevelCostEstimator::PredictMaxPool(const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // x: op_info.inputs(0)
  ConvolutionDimensions dims = OpDimensionsFromInputs(
      op_info.inputs(0).shape(), op_info, &found_unknown_shapes);
  // kx * ky - 1 comparisons per output (kx * xy > 1)
  // or 1 copy per output (kx * k1 = 1).
  int per_output_ops = dims.kx * dims.ky == 1 ? 1 : dims.kx * dims.ky - 1;
  int64 ops = dims.batch * dims.ox * dims.oy * dims.oz * per_output_ops;

  double total_input_size = 0;
  if (dims.ky >= dims.sy) {
    total_input_size =
        CalculateTensorSize(op_info.inputs(0), &found_unknown_shapes);
  } else {  // dims.ky < dims.sy
    // Vertical stride is larger than vertical kernel; assuming row-major
    // format, skip unnecessary rows (or read every kx rows per sy rows, as the
    // others are not used for output).
    const auto data_size = DataTypeSize(BaseType(op_info.inputs(0).dtype()));
    total_input_size =
        data_size * dims.batch * dims.ix * dims.ky * dims.oy * dims.iz;
  }
  const double total_output_size =
      CalculateOutputSize(op_info, &found_unknown_shapes);

  Costs costs = PredictOpCountBasedCost(ops, total_input_size,
                                        total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictMaxPoolGrad(
    const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // x: op_info.inputs(0)
  // y: op_info.inputs(1)
  // y_grad: op_info.inputs(2)
  ConvolutionDimensions dims = OpDimensionsFromInputs(
      op_info.inputs(0).shape(), op_info, &found_unknown_shapes);

  int64 ops = 0;
  if (dims.kx == 1 && dims.ky == 1) {
    // 1x1 window. No need to know which input was max.
    ops = dims.batch * dims.ix * dims.iy * dims.iz;
  } else if (dims.kx <= dims.sx && dims.ky <= dims.sy) {
    // Non-overlapping window: re-run maxpool, then assign zero or y_grad.
    ops = dims.batch * dims.iz *
          (dims.ox * dims.oy * (dims.kx * dims.ky - 1) + dims.ix * dims.iy);
  } else {
    // Overlapping window: initialize with zeros, re-run maxpool, then
    // accumulate y_gad to proper x_grad locations.
    ops = dims.batch * dims.iz *
          (dims.ox * dims.oy * (dims.kx * dims.ky - 1) + dims.ix * dims.iy * 2);
  }

  // Just read x and y_grad; no need to read y as we assume MaxPoolGrad re-run
  // MaxPool internally.
  double total_input_size =
      CalculateTensorSize(op_info.inputs(0), &found_unknown_shapes);
  total_input_size +=
      CalculateTensorSize(op_info.inputs(2), &found_unknown_shapes);
  // Write x_grad; size equal to x.
  const double total_output_size =
      CalculateTensorSize(op_info.inputs(0), &found_unknown_shapes);

  Costs costs = PredictOpCountBasedCost(ops, total_input_size,
                                        total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictAvgPool(const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // x: op_info.inputs(0)
  ConvolutionDimensions dims = OpDimensionsFromInputs(
      op_info.inputs(0).shape(), op_info, &found_unknown_shapes);

  // kx * ky - 1 additions and 1 multiplication per output.
  int64 ops = dims.batch * dims.ox * dims.oy * dims.oz * dims.kx * dims.ky;

  double total_input_size = 0;
  if (dims.ky >= dims.sy) {
    total_input_size =
        CalculateTensorSize(op_info.inputs(0), &found_unknown_shapes);
  } else {  // dims.ky < dims.sy
    // vertical stride is larger than vertical kernel; assuming row-major
    // format, skip unnecessary rows (or read every kx rows per sy rows, as the
    // others are not used for output).
    const auto data_size = DataTypeSize(BaseType(op_info.inputs(0).dtype()));
    total_input_size =
        data_size * dims.batch * dims.ix * dims.ky * dims.oy * dims.iz;
  }
  const double total_output_size =
      CalculateOutputSize(op_info, &found_unknown_shapes);

  Costs costs = PredictOpCountBasedCost(ops, total_input_size,
                                        total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictAvgPoolGrad(
    const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // x's shape: op_info.inputs(0)
  // y_grad: op_info.inputs(1)

  // Extract x_shape from op_info.inputs(0).value() or op_info.outputs(0).
  bool shape_found = false;
  TensorShapeProto x_shape;
  if (op_info.inputs_size() >= 1 && op_info.inputs(0).has_value()) {
    const TensorProto& value = op_info.inputs(0).value();
    shape_found = GetTensorShapeProtoFromTensorProto(value, &x_shape);
  }
  if (!shape_found && op_info.outputs_size() > 0) {
    x_shape = op_info.outputs(0).shape();
    shape_found = true;
  }
  if (!shape_found) {
    // Set the minimum shape that's feasible.
    x_shape.Clear();
    for (int i = 0; i < 4; ++i) {
      x_shape.add_dim()->set_size(1);
    }
    found_unknown_shapes = true;
  }

  ConvolutionDimensions dims =
      OpDimensionsFromInputs(x_shape, op_info, &found_unknown_shapes);

  int64 ops = 0;
  if (dims.kx <= dims.sx && dims.ky <= dims.sy) {
    // Non-overlapping window.
    ops = dims.batch * dims.iz * (dims.ix * dims.iy + dims.ox * dims.oy);
  } else {
    // Overlapping window.
    ops = dims.batch * dims.iz *
          (dims.ix * dims.iy + dims.ox * dims.oy * (dims.kx * dims.ky + 1));
  }

  const double total_input_size =
      CalculateInputSize(op_info, &found_unknown_shapes);
  const double total_output_size =
      CalculateOutputSize(op_info, &found_unknown_shapes);

  Costs costs = PredictOpCountBasedCost(ops, total_input_size,
                                        total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictFusedBatchNorm(
    const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // x: op_info.inputs(0)
  // scale: op_info.inputs(1)
  // offset: op_info.inputs(2)
  // mean: op_info.inputs(3)  --> only for inference
  // variance: op_info.inputs(4) --> only for inference
  ConvolutionDimensions dims = OpDimensionsFromInputs(
      op_info.inputs(0).shape(), op_info, &found_unknown_shapes);
  const bool is_training = IsTraining(op_info);

  int64 ops = 0;
  const auto rsqrt_cost = Eigen::internal::functor_traits<
      Eigen::internal::scalar_rsqrt_op<float>>::Cost;
  if (is_training) {
    ops = dims.iz * (dims.batch * dims.ix * dims.iy * 4 + 6 + rsqrt_cost);
  } else {
    ops = dims.batch * dims.ix * dims.iy * dims.iz * 2;
  }

  const double size_nhwc =
      CalculateTensorSize(op_info.inputs(0), &found_unknown_shapes);
  const double size_c =
      CalculateTensorSize(op_info.inputs(1), &found_unknown_shapes);
  double total_input_size = 0.0;
  double total_internal_read_size = 0.0;
  double total_output_size = 0.0;
  if (is_training) {
    total_input_size = size_nhwc + size_c * 2;
    total_output_size = size_nhwc + size_c * 4;
    total_internal_read_size = size_nhwc;
  } else {
    total_input_size = size_nhwc + size_c * 4;
    total_output_size = size_nhwc;
  }

  Costs costs =
      PredictOpCountBasedCost(ops, total_input_size + total_internal_read_size,
                              total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

Costs OpLevelCostEstimator::PredictFusedBatchNormGrad(
    const OpContext& op_context) const {
  bool found_unknown_shapes = false;
  const auto& op_info = op_context.op_info;
  // y_backprop: op_info.inputs(0)
  // x: op_info.inputs(1)
  // scale: op_info.inputs(2)
  // mean: op_info.inputs(3)
  // variance or inverse of variance: op_info.inputs(4)
  ConvolutionDimensions dims = OpDimensionsFromInputs(
      op_info.inputs(1).shape(), op_info, &found_unknown_shapes);

  int64 ops = 0;
  const auto rsqrt_cost = Eigen::internal::functor_traits<
      Eigen::internal::scalar_rsqrt_op<float>>::Cost;
  ops = dims.iz * (dims.batch * dims.ix * dims.iy * 11 + 5 + rsqrt_cost);

  const double size_nhwc =
      CalculateTensorSize(op_info.inputs(1), &found_unknown_shapes);
  const double size_c =
      CalculateTensorSize(op_info.inputs(2), &found_unknown_shapes);
  double total_input_size = size_nhwc * 2 + size_c * 2;
  double total_internal_read_size = size_nhwc;
  double total_output_size = size_nhwc * 1 + size_c * 2;

  Costs costs =
      PredictOpCountBasedCost(ops, total_input_size + total_internal_read_size,
                              total_output_size, op_info);
  costs.inaccurate = found_unknown_shapes;
  costs.num_ops_with_unknown_shapes = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

/* static */
void OpLevelCostEstimator::CombineCostsAndUpdateExecutionTime(
    Costs* costs) const {
  if (compute_memory_overlap_) {
    costs->execution_time =
        std::max(costs->intermediate_memory_time,
                 std::max(costs->compute_time, costs->memory_time));
  } else {
    costs->execution_time = costs->compute_time + costs->memory_time +
                            costs->intermediate_memory_time;
  }
}
}  // end namespace grappler
}  // end namespace tensorflow
