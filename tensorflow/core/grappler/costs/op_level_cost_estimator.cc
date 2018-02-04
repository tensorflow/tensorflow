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
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/utils.h"

namespace tensorflow {
namespace grappler {

constexpr int kOpsPerMac = 2;
constexpr char kConst[] = "Const";
constexpr char kConv2d[] = "Conv2D";
constexpr char kConv2dBackpropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv2dBackpropInput[] = "Conv2DBackpropInput";
constexpr char kMatMul[] = "MatMul";
constexpr char kSparseMatMul[] = "SparseMatMul";
constexpr char kPlaceholder[] = "Placeholder";
constexpr char kIdentity[] = "Identity";
constexpr char kRefIdentity[] = "RefIdentity";
constexpr char kNoOp[] = "NoOp";
constexpr char kReshape[] = "Reshape";
constexpr char kRecv[] = "_Recv";
constexpr char kSend[] = "_Send";
constexpr char kBatchMatMul[] = "BatchMatMul";
constexpr char kVariable[] = "Variable";
constexpr char kVariableV2[] = "VariableV2";
constexpr char kRank[] = "Rank";
constexpr char kShape[] = "Shape";
constexpr char kSize[] = "Size";
constexpr char kStopGradient[] = "StopGradient";
constexpr char kPreventGradient[] = "PreventGradient";

static const Costs::Duration kMinComputeTime(1);

namespace {

string GetDataFormat(const OpInfo& op_features) {
  string data_format = "NHWC";  // Default format.
  if (op_features.attr().find("data_format") != op_features.attr().end()) {
    data_format = op_features.attr().at("data_format").s();
  }
  return data_format;
}

Padding GetPadding(const OpInfo& op_features) {
  if (op_features.attr().find("padding") != op_features.attr().end() &&
      op_features.attr().at("padding").s() == "VALID") {
    return Padding::VALID;
  }
  return Padding::SAME;  // Default padding.
}

std::vector<int64> GetStrides(const OpInfo& op_features) {
  if (op_features.attr().find("strides") != op_features.attr().end()) {
    const auto strides = op_features.attr().at("strides").list().i();
    return {strides[0], strides[1], strides[2], strides[3]};
  }
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

// Return a minimum shape if the shape is unknown. If known, return the original
// shape.
TensorShapeProto MaybeGetMinimumShape(const TensorShapeProto& original_shape,
                                      int rank, bool* found_unknown_shapes) {
  auto shape = original_shape;
  if (shape.unknown_rank() || shape.dim_size() < rank) {
    *found_unknown_shapes = true;
    TensorShapeProto::Dim dim;
    VLOG(2) << "Use minimum shape because the rank is unknown.";
    // The size of each dimension is at least 1, if unknown.
    dim.set_size(1);
    for (int i = 0; i < rank; i++) {
      *shape.add_dim() = dim;
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
      {kMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kSparseMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kBatchMatMul, wrap(&OpLevelCostEstimator::PredictBatchMatMul)},

      {kNoOp, wrap(&OpLevelCostEstimator::PredictNoOp)},

      {kPlaceholder, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kIdentity, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kRefIdentity, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kStopGradient, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kPreventGradient, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kReshape, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kRecv, wrap(&OpLevelCostEstimator::PredictIdentity)},
      {kSend, wrap(&OpLevelCostEstimator::PredictIdentity)},

      {kConst, wrap(&OpLevelCostEstimator::PredictVariable)},
      {kVariable, wrap(&OpLevelCostEstimator::PredictVariable)},
      {kVariableV2, wrap(&OpLevelCostEstimator::PredictVariable)},

      {kRank, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kShape, wrap(&OpLevelCostEstimator::PredictMetadata)},
      {kSize, wrap(&OpLevelCostEstimator::PredictMetadata)}};

  elementwise_ops_ = {
      // Unary ops alphabetically sorted
      {"Acos", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_acos_op<float>>::Cost},
      {"Asin", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_asin_op<float>>::Cost},
      {"Atan", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_atan_op<float>>::Cost},
      {"Atan2", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_quotient_op<float>>::Cost +
                    Eigen::internal::functor_traits<
                        Eigen::internal::scalar_atan_op<float>>::Cost},
      {"Ceil", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_ceil_op<float>>::Cost},
      {"Cos", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_cos_op<float>>::Cost},
      {"Erf", 1},
      {"Erfc", 1},
      {"Exp", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_exp_op<float>>::Cost},
      {"Expm1", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_expm1_op<float>>::Cost},
      {"Floor", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_floor_op<float>>::Cost},
      {"Inv", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_inverse_op<float>>::Cost},
      {"InvGrad", 1},
      {"Lgamma", 1},
      {"Log", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_log_op<float>>::Cost},
      {"Log1p", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_log1p_op<float>>::Cost},
      {"Neg", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_opposite_op<float>>::Cost},
      {"Reciprocal", Eigen::internal::functor_traits<
                         Eigen::internal::scalar_inverse_op<float>>::Cost},
      {"Rint", 1},
      {"Round", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_round_op<float>>::Cost},
      {"Rsqrt", Eigen::internal::functor_traits<
                    Eigen::internal::scalar_rsqrt_op<float>>::Cost},
      {"Sqrt", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_sqrt_op<float>>::Cost},
      {"Square", Eigen::internal::functor_traits<
                     Eigen::internal::scalar_square_op<float>>::Cost},
      {"Tanh", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_tanh_op<float>>::Cost},
      {"Relu", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_max_op<float>>::Cost},
      {"Sigmoid", Eigen::internal::functor_traits<
                      Eigen::internal::scalar_sigmoid_op<float>>::Cost},
      {"Sign", Eigen::internal::functor_traits<
                   Eigen::internal::scalar_sign_op<float>>::Cost},
      {"Sin", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_sin_op<float>>::Cost},
      {"Tan", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_tan_op<float>>::Cost},
      // Binary ops alphabetically sorted
      {"Add", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_sum_op<float>>::Cost},
      {"ApproximateEqual", 1},
      {"Div", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_quotient_op<float>>::Cost},
      {"Equal", 1},
      {"FloorDiv", Eigen::internal::functor_traits<
                       Eigen::internal::scalar_quotient_op<float>>::Cost},
      {"FloorMod", Eigen::internal::functor_traits<
                       Eigen::internal::scalar_mod_op<float>>::Cost},
      {"Greater", 1},
      {"GreaterEqual", 1},
      {"Less", 1},
      {"LessEqual", 1},
      {"LogicalAnd", Eigen::internal::functor_traits<
                         Eigen::internal::scalar_boolean_and_op>::Cost},
      {"LogicalNot", 1},
      {"LogicalOr", Eigen::internal::functor_traits<
                        Eigen::internal::scalar_boolean_or_op>::Cost},
      {"Maximum", Eigen::internal::functor_traits<
                      Eigen::internal::scalar_max_op<float>>::Cost},
      {"Minimum", Eigen::internal::functor_traits<
                      Eigen::internal::scalar_min_op<float>>::Cost},
      {"Mod", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_mod_op<float>>::Cost},
      {"Mul", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_product_op<float>>::Cost},
      {"NotEqual", 1},
      {"QuantizedAdd", Eigen::internal::functor_traits<
                           Eigen::internal::scalar_sum_op<float>>::Cost},
      {"QuantizedMul", Eigen::internal::functor_traits<
                           Eigen::internal::scalar_product_op<float>>::Cost},
      {"RealDiv", Eigen::internal::functor_traits<
                      Eigen::internal::scalar_quotient_op<float>>::Cost},
      {"SquareDifference", 1},
      {"Sub", Eigen::internal::functor_traits<
                  Eigen::internal::scalar_difference_op<float>>::Cost},
      {"TruncateDiv", Eigen::internal::functor_traits<
                          Eigen::internal::scalar_quotient_op<float>>::Cost},
      {"TruncateMod", Eigen::internal::functor_traits<
                          Eigen::internal::scalar_mod_op<float>>::Cost}};

  // By default, use sum of memory_time and compute_time for execution_time.
  compute_memory_overlap_ = false;
}

Costs OpLevelCostEstimator::PredictCosts(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  auto it = device_cost_impl_.find(op_features.op());
  if (it == device_cost_impl_.end()) {
    if (elementwise_ops_.find(op_features.op()) != elementwise_ops_.end()) {
      return PredictCwiseOp(op_context);
    }

    VLOG(1) << "Missing accurate estimator for op: " << op_features.op();

    return PredictCostOfAnUnknownOp(op_context);
  }

  std::function<Costs(const OpContext&)> estimator = it->second;
  Costs costs = estimator(op_context);
  VLOG(1) << "Operation " << op_features.op() << " takes "
          << costs.execution_time.count() << " ns.";
  return costs;
}

OpLevelCostEstimator::DeviceInfo OpLevelCostEstimator::GetDeviceInfo(
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

  return {gflops, gb_per_sec};
}

Costs OpLevelCostEstimator::PredictCwiseOp(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  // For unary or binary element-wise operations, op count is the element count
  // of any input. We use the count for the largest input here to be more robust
  // in case that the shape is unknown or partially known for other input.
  int64 op_count =
      CalculateLargestInputCount(op_features, &found_unknown_shapes);
  // If output shape is available, try use the element count calcuated from
  // that.
  if (op_features.outputs_size() > 0) {
    op_count =
        std::max(op_count, CalculateTensorElementCount(op_features.outputs(0),
                                                       &found_unknown_shapes));
  }
  // For binary ops, calculate the output shape possibly resulting from
  // broadcasting.
  if (op_features.inputs_size() >= 2) {
    op_count = std::max(op_count,
                        CwiseOutputElementCount(op_features.inputs(0).shape(),
                                                op_features.inputs(1).shape()));
  }

  int op_cost = 1;
  bool is_known_elementwise_op = false;
  auto it = elementwise_ops_.find(op_features.op());
  if (it != elementwise_ops_.end()) {
    op_cost = it->second;
    is_known_elementwise_op = true;
  } else {
    LOG(WARNING) << "Not a cwise op: " << op_features.op();
  }

  Costs costs = PredictOpCountBasedCost(op_count * op_cost, op_features);
  if (found_unknown_shapes || !is_known_elementwise_op) {
    costs.inaccurate = true;
  }
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
    double operations, const OpInfo& op_features) const {
  DeviceInfo device_perf = GetDeviceInfo(op_features.device());

  Costs::NanoSeconds compute_cost(std::ceil(operations / device_perf.gigaops));
  VLOG(1) << "Op:" << op_features.op() << " GOps:" << operations / 1e9
          << " Execution Time (ns):" << compute_cost.count();

  bool found_unknown_shapes = false;
  const double total_input_size =
      CalculateInputSize(op_features, &found_unknown_shapes);
  const double total_output_size =
      CalculateOutputSize(op_features, &found_unknown_shapes);
  const double total_io_size = total_input_size + total_output_size;

  Costs::NanoSeconds memory_cost(
      std::ceil(total_io_size / device_perf.gb_per_sec));
  VLOG(1) << "Op:" << op_features.op() << " Size (KB):" << (total_io_size) / 1e3
          << " Memory Time (ns):" << memory_cost.count();

  Costs costs;
  costs.compute_time = compute_cost;
  costs.memory_time = memory_cost;
  if (compute_memory_overlap_) {
    costs.execution_time = std::max(compute_cost, memory_cost);
  } else {
    costs.execution_time = compute_cost + memory_cost;
  }
  costs.inaccurate = found_unknown_shapes;
  costs.max_memory = total_output_size;
  return costs;
}

int64 OpLevelCostEstimator::CountConv2DOperations(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  return CountConv2DOperations(op_features, nullptr, found_unknown_shapes);
}

// Helper to translate the positional arguments into named fields.
OpLevelCostEstimator::ConvolutionDimensions
OpLevelCostEstimator::ConvolutionDimensionsFromInputs(
    const TensorShapeProto& original_image_shape,
    const TensorShapeProto& original_filter_shape, const OpInfo& op_features,
    bool* found_unknown_shapes) {
  auto image_shape =
      MaybeGetMinimumShape(original_image_shape, 4, found_unknown_shapes);
  auto filter_shape =
      MaybeGetMinimumShape(original_filter_shape, 4, found_unknown_shapes);

  int x_index, y_index, channel_index;
  const string& data_format = GetDataFormat(op_features);
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
  int64 kx = filter_shape.dim(0).size();
  int64 ky = filter_shape.dim(1).size();
  std::vector<int64> strides = GetStrides(op_features);
  const auto padding = GetPadding(op_features);
  int64 sx = strides[x_index];
  int64 sy = strides[y_index];
  int64 ox = GetOutputSize(ix, kx, sx, padding);
  int64 oy = GetOutputSize(iy, ky, sy, padding);
  int64 oz = filter_shape.dim(3).size();
  // Only check equality when both sizes are known (in other words, when
  // neither is set to a minimum dimension size of 1).
  if (iz != 1 && filter_shape.dim(2).size() != 1) {
    CHECK_EQ(iz, filter_shape.dim(2).size());
  } else {
    iz = std::max<int64>(iz, filter_shape.dim(2).size());
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
    const OpInfo& op_features, ConvolutionDimensions* conv_info,
    bool* found_unknown_shapes) const {
  if (op_features.op() != kConv2d) {
    LOG(ERROR) << "Invalid Operation";
    return 0;
  }
  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      op_features.inputs(0).shape(), op_features.inputs(1).shape(), op_features,
      found_unknown_shapes);

  int64 ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  ops *= conv_dims.iz * conv_dims.oz;
  ops *= kOpsPerMac;
  VLOG(1) << "Operations for Conv2D " << ops;

  if (conv_info != nullptr) {
    *conv_info = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  return CountMatMulOperations(op_features, nullptr, found_unknown_shapes);
}

// TODO(nishantpatil): Create separate estimator for Sparse Matmul
int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_features, MatMulDimensions* mat_mul,
    bool* found_unknown_shapes) const {
  double ops = 0;

  if (op_features.inputs_size() < 2) {
    LOG(ERROR) << "Need 2 inputs but got " << op_features.inputs_size();
    *found_unknown_shapes = true;
    return 0;
  }

  auto& a_matrix = op_features.inputs(0);
  auto& b_matrix = op_features.inputs(1);

  bool transpose_a = false;
  bool transpose_b = false;

  double m_dim, n_dim, k_dim, k_dim_b = 0;

  for (const auto& item : op_features.attr()) {
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
  VLOG(1) << "Operations for Matmul" << ops;

  if (mat_mul != nullptr) {
    mat_mul->m = m_dim;
    mat_mul->n = n_dim;
    mat_mul->k = k_dim;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountBatchMatMulOperations(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  if (op_features.op() != kBatchMatMul) {
    LOG(ERROR) << "Invalid Operation: " << op_features.op();
    *found_unknown_shapes = true;
    return 0;
  }
  if (op_features.inputs_size() != 2) {
    LOG(ERROR) << "Expected 2 inputs but got " << op_features.inputs_size();
    *found_unknown_shapes = true;
    return 0;
  }

  double ops = 0;
  const auto& a_input = op_features.inputs(0);
  const auto& b_input = op_features.inputs(1);

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
  OpInfo matmul_op_features;
  matmul_op_features.set_op("MatMul");

  AttrValue transpose_a;
  transpose_a.set_b(false);
  if (op_features.attr().find("adj_x") != op_features.attr().end()) {
    transpose_a.set_b(op_features.attr().at("adj_x").b());
  }
  (*matmul_op_features.mutable_attr())["transpose_a"] = transpose_a;

  AttrValue transpose_b;
  transpose_b.set_b(false);
  if (op_features.attr().find("adj_y") != op_features.attr().end()) {
    transpose_b.set_b(op_features.attr().at("adj_y").b());
  }
  (*matmul_op_features.mutable_attr())["transpose_b"] = transpose_b;

  OpInfo::TensorProperties* a_matrix = matmul_op_features.add_inputs();
  a_matrix->set_dtype(a_input.dtype());
  TensorShapeProto* a_matrix_shape = a_matrix->mutable_shape();
  for (int i = std::max(0, a_input_shape.dim_size() - matrix_rank);
       i < a_input_shape.dim_size(); ++i) {
    *(a_matrix_shape->add_dim()) = a_input_shape.dim(i);
  }

  OpInfo::TensorProperties* b_matrix = matmul_op_features.add_inputs();
  b_matrix->set_dtype(b_input.dtype());
  TensorShapeProto* b_matrix_shape = b_matrix->mutable_shape();
  for (int i = std::max(0, b_input_shape.dim_size() - matrix_rank);
       i < b_input_shape.dim_size(); ++i) {
    *(b_matrix_shape->add_dim()) = b_input_shape.dim(i);
  }

  for (int i = 0; i < num_matmuls; ++i) {
    bool matmul_unknown_shapes = false;
    ops += CountMatMulOperations(matmul_op_features, &matmul_unknown_shapes);
    *found_unknown_shapes |= matmul_unknown_shapes;
  }
  return ops;
}

// TODO(cliffy): Dedup this method and CountConv2DBackpropFilterOperations.
int64 OpLevelCostEstimator::CountConv2DBackpropInputOperations(
    const OpInfo& op_features, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;

  if (op_features.op() != kConv2dBackpropInput) {
    LOG(ERROR) << "Invalid Operation";
    return ops;
  }

  if (op_features.outputs_size() != 1) {
    // Need _output_shapes for input shape.
    LOG(ERROR) << "No output shape in Conv2DBackpropInput op.";
    return ops;
  }

  const auto& input_shape = op_features.outputs(0).shape();
  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      input_shape, op_features.inputs(1).shape(), op_features,
      found_unknown_shapes);

  ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  ops *= conv_dims.iz * conv_dims.oz;
  ops *= kOpsPerMac;

  VLOG(1) << "Operations for Conv2DBackpropInput " << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountConv2DBackpropFilterOperations(
    const OpInfo& op_features, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;
  if (op_features.op() != kConv2dBackpropFilter) {
    LOG(ERROR) << "Invalid Operation";
    return ops;
  }

  if (op_features.outputs_size() != 1) {
    // Need _output_shapes for input shape.
    LOG(ERROR) << "No output shape in Conv2DBackpropFilter op.";
    return ops;
  }

  const auto& filter_shape = op_features.outputs(0).shape();
  ConvolutionDimensions conv_dims = ConvolutionDimensionsFromInputs(
      op_features.inputs(0).shape(), filter_shape, op_features,
      found_unknown_shapes);

  ops = conv_dims.batch;
  ops *= conv_dims.ox * conv_dims.oy;
  ops *= conv_dims.kx * conv_dims.ky;
  ops *= conv_dims.iz * conv_dims.oz;
  ops *= kOpsPerMac;

  VLOG(1) << "Operations for Conv2DBackpropFilter" << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CalculateTensorElementCount(
    const OpInfo::TensorProperties& tensor, bool* found_unknown_shapes) const {
  VLOG(2) << "   with " << tensor.dtype() << " tensor of shape "
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
  return CalculateTensorElementCount(tensor, found_unknown_shapes) *
         DataTypeSize(BaseType(tensor.dtype()));
}

int64 OpLevelCostEstimator::CalculateInputSize(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  int64 total_input_size = 0;
  for (auto& input : op_features.inputs()) {
    int64 input_size = CalculateTensorSize(input, found_unknown_shapes);
    total_input_size += input_size;
    VLOG(1) << "Input Size: " << input_size
            << " Total Input Size:" << total_input_size;
  }
  return total_input_size;
}

int64 OpLevelCostEstimator::CalculateLargestInputCount(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  int64 largest_input_count = 0;
  for (auto& input : op_features.inputs()) {
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
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  int64 total_output_size = 0;
  // use float as default for calculations
  for (const auto& output : op_features.outputs()) {
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
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountConv2DOperations(op_features, &found_unknown_shapes), op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackpropInput(
    const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackpropInputOperations(
                                  op_features, nullptr, &found_unknown_shapes),
                              op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackpropFilter(
    const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackpropFilterOperations(
                                  op_features, nullptr, &found_unknown_shapes),
                              op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictMatMul(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountMatMulOperations(op_features, &found_unknown_shapes), op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictNoOp(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  VLOG(1) << "Op:" << op_features.op() << " Execution Time 0 (ns)";
  return Costs::ZeroCosts();
}

Costs OpLevelCostEstimator::PredictIdentity(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  VLOG(1) << "Op:" << op_features.op() << " Execution Time 0 (ns)";
  Costs result = Costs::ZeroCosts();
  result.max_memory = CalculateOutputSize(op_features, &result.inaccurate);
  // Assign the minimum amount of time we can represent to the identity op since
  // it tends to be really cheap.
  result.compute_time = kMinComputeTime;
  result.execution_time = result.compute_time;
  return result;
}

Costs OpLevelCostEstimator::PredictVariable(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  VLOG(1) << "Op:" << op_features.op() << " Execution Time 0 (ns)";
  Costs result = Costs::ZeroCosts();
  result.persistent_memory =
      CalculateOutputSize(op_features, &result.inaccurate);

  result.compute_time = kMinComputeTime;
  result.execution_time = result.execution_time;
  return result;
}

Costs OpLevelCostEstimator::PredictBatchMatMul(
    const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  bool found_unknown_shapes = false;
  Costs costs = PredictOpCountBasedCost(
      CountBatchMatMulOperations(op_features, &found_unknown_shapes),
      op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictMetadata(const OpContext& op_context) const {
  const auto& op_features = op_context.op_info;
  Costs costs = Costs::ZeroCosts();
  costs.max_memory = CalculateOutputSize(op_features, &costs.inaccurate);
  // Metadata operations are so cheap we assume they take the minimum amount of
  // time we can represent (1 ns).
  costs.compute_time = kMinComputeTime;
  costs.execution_time = costs.compute_time;

  return costs;
}

}  // end namespace grappler
}  // end namespace tensorflow
