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
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/clusters/utils.h"

namespace tensorflow {
namespace grappler {

constexpr int kOpsPerMac = 2;
constexpr char kConv2d[] = "Conv2D";
constexpr char kConv2dBackPropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv2dBackPropInput[] = "Conv2DBackpropInput";
constexpr char kMatMul[] = "MatMul";
constexpr char kSparseMatMul[] = "SparseMatMul";
constexpr char kIdentity[] = "Identity";
constexpr char kNoOp[] = "NoOp";
constexpr char kReshape[] = "Reshape";
constexpr char kRecv[] = "_Recv";
constexpr char kBatchMatMul[] = "BatchMatMul";

OpLevelCostEstimator::OpLevelCostEstimator() {
  // Syntactic sugar to build and return a lambda that takes an OpInfo and
  // returns a cost.
  typedef Costs (OpLevelCostEstimator::*CostImpl)(const OpInfo& op_feature)
      const;
  auto wrap = [this](CostImpl impl) -> std::function<Costs(const OpInfo&)> {
    return [this, impl](const OpInfo& op) { return (this->*impl)(op); };
  };

  device_cost_impl_ = {
      {kConv2d, wrap(&OpLevelCostEstimator::PredictConv2D)},
      {kConv2dBackPropFilter,
       wrap(&OpLevelCostEstimator::PredictConv2DBackPropFilter)},
      {kConv2dBackPropInput,
       wrap(&OpLevelCostEstimator::PredictConv2DBackPropInput)},
      {kMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kSparseMatMul, wrap(&OpLevelCostEstimator::PredictMatMul)},
      {kIdentity, wrap(&OpLevelCostEstimator::PredictNoOp)},
      {kNoOp, wrap(&OpLevelCostEstimator::PredictNoOp)},
      {kReshape, wrap(&OpLevelCostEstimator::PredictNoOp)},
      {kRecv, wrap(&OpLevelCostEstimator::PredictNoOp)},
      {kBatchMatMul, wrap(&OpLevelCostEstimator::PredictBatchMatMul)}};
}

Costs OpLevelCostEstimator::PredictCosts(const OpInfo& op_features) const {
  auto it = device_cost_impl_.find(op_features.op());
  if (it == device_cost_impl_.end()) {
    VLOG(1) << "Missing implementation for op: " << op_features.op();
    Costs costs;
    costs = DummyExecutionTime(op_features);
    return costs;
  }

  std::function<Costs(const OpInfo&)> estimator = it->second;
  Costs costs = estimator(op_features);
  VLOG(1) << "Operation " << op_features.op() << " takes "
          << costs.execution_time.count() << " ns.";
  return costs;
}

std::pair<double, double> OpLevelCostEstimator::GetDeviceInfo(
    const DeviceProperties& device) const {
  double gflops = -1;
  double bandwidth = -1;

  if (device.type() == "CPU") {
    // Check if vector instructions are available, and refine performance
    // prediction based on this.
    // Frequencies are stored in MHz in the DeviceProperties.
    gflops = device.num_cores() * device.frequency() * 1e-3;
    if (bandwidth < 0) {
      if (device.bandwidth() > 0) {
        bandwidth = device.bandwidth() / 1e6;
      } else {
        bandwidth = 32;
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
      // Pascal
      cores_per_multiprocessor = 64;
    }
    gflops = device.num_cores() * device.frequency() * 1e-3 *
             cores_per_multiprocessor * kOpsPerMac;
    if (device.bandwidth() > 0) {
      bandwidth = device.bandwidth() / 1e6;
    } else {
      bandwidth = 100;
    }
  }

  return std::make_pair(gflops, bandwidth);
}

Costs OpLevelCostEstimator::DummyExecutionTime(
    const OpInfo& op_features) const {
  Costs costs = PredictOpCountBasedCost(0, op_features);
  costs.inaccurate = true;
  return costs;
}

Costs OpLevelCostEstimator::PredictOpCountBasedCost(
    double operations, const OpInfo& op_features) const {
  std::pair<double, double> device_perf = GetDeviceInfo(op_features.device());
  Costs::NanoSeconds compute_cost(operations / device_perf.first);
  VLOG(1) << "Op:" << op_features.op() << " GOps:" << operations / 1e9
          << " Execution Time (ns):" << compute_cost.count();

  bool found_unknown_shapes = false;
  double total_input_size =
      CalculateInputSize(op_features, &found_unknown_shapes);
  double total_output_size =
      CalculateOutputSize(op_features, &found_unknown_shapes);
  double total_io_size = total_input_size + total_output_size;

  Costs::NanoSeconds memory_cost(total_io_size / device_perf.second);
  VLOG(1) << "Op:" << op_features.op() << " Size (KB):" << (total_io_size) / 1e3
          << " Memory Time (ns):" << memory_cost.count();

  Costs costs;
  costs.compute_time = compute_cost;
  costs.memory_time = memory_cost;
  costs.execution_time = compute_cost + memory_cost;
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

int64 OpLevelCostEstimator::CountConv2DOperations(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  return CountConv2DOperations(op_features, nullptr, found_unknown_shapes);
}

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
  if (shape.unknown_rank()) {
    *found_unknown_shapes = true;
  }
  if (shape.unknown_rank() || shape.dim_size() == 0) {
    TensorShapeProto::Dim dim;
    VLOG(1) << "WARNING: Use minimum shape because the shape is unknown.";
    // The size of each dimension is at least 1, if unknown.
    dim.set_size(1);
    for (int i = 0; i < rank; i++) {
      *shape.add_dim() = dim;
    }
  } else {
    CHECK_EQ(shape.dim_size(), rank);
    for (int i = 0; i < rank; i++) {
      if (shape.dim(i).size() == -1) {
        *found_unknown_shapes = true;
        VLOG(1)
            << "WARNING: Use minimum dim size 1 because the shape is unknown.";
        // The size of each dimension is at least 1, if unknown.
        shape.mutable_dim(i)->set_size(1);
      }
    }
  }
  return shape;
}
}  // namespace

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
  VLOG(1) << "Operations for Conv2D" << ops;

  if (conv_info != nullptr) {
    *conv_info = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  return CountMatMulOperations(op_features, nullptr, found_unknown_shapes);
}

int64 OpLevelCostEstimator::CountMatMulOperations(
    const OpInfo& op_features, MatMulDimensions* mat_mul,
    bool* found_unknown_shapes) const {
  double ops = 0;

  // TODO(nishantpatil): Create separate estimator for Sparse Matmul
  if ((op_features.op() != kMatMul) && (op_features.op() != kSparseMatMul)) {
    LOG(ERROR) << "Invalid Operation";
    return ops;
  }

  // first matrix
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
  auto& a_input = op_features.inputs(0);
  auto& b_input = op_features.inputs(1);

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
  for (int i = a_input_shape.dim_size() - matrix_rank;
       i < a_input_shape.dim_size(); ++i) {
    *(a_matrix_shape->add_dim()) = a_input_shape.dim(i);
  }

  OpInfo::TensorProperties* b_matrix = matmul_op_features.add_inputs();
  b_matrix->set_dtype(b_input.dtype());
  TensorShapeProto* b_matrix_shape = b_matrix->mutable_shape();
  for (int i = b_input_shape.dim_size() - matrix_rank;
       i < b_input_shape.dim_size(); ++i) {
    *(b_matrix_shape->add_dim()) = b_input_shape.dim(i);
  }

  for (int i = 0; i < num_matmuls; ++i) {
    bool matmul_unknown_shapes = false;
    ops += CountMatMulOperations(matmul_op_features, &matmul_unknown_shapes);
    CHECK_EQ(false, matmul_unknown_shapes);
  }
  return ops;
}

// TODO(cliffy): Dedup this method and CountConv2DBackPropFilterOperations.
int64 OpLevelCostEstimator::CountConv2DBackPropInputOperations(
    const OpInfo& op_features, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;

  if (op_features.op() != kConv2dBackPropInput) {
    LOG(ERROR) << "Invalid Operation";
    return ops;
  }

  if (op_features.outputs_size() != 1) {
    // Need _output_shapes for input shape.
    LOG(ERROR) << "No output shape in Conv2DBackPropInput op.";
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

  VLOG(1) << "Operations for Conv2DBackPropInput" << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CountConv2DBackPropFilterOperations(
    const OpInfo& op_features, ConvolutionDimensions* returned_conv_dims,
    bool* found_unknown_shapes) const {
  int64 ops = 0;
  if (op_features.op() != kConv2dBackPropFilter) {
    LOG(ERROR) << "Invalid Operation";
    return ops;
  }

  if (op_features.outputs_size() != 1) {
    // Need _output_shapes for input shape.
    LOG(ERROR) << "No output shape in Conv2DBackPropFilter op.";
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

  VLOG(1) << "Operations for Conv2DBackPropFilter" << ops;

  if (returned_conv_dims != nullptr) {
    *returned_conv_dims = conv_dims;
  }
  return ops;
}

int64 OpLevelCostEstimator::CalculateSingleInputSize(
    const OpInfo::TensorProperties& input, bool* found_unknown_shapes) const {
  VLOG(1) << "   with " << input.dtype() << " input of shape "
          << input.shape().DebugString();
  int64 input_size = 1;
  int num_dims = std::max(1, input.shape().dim_size());
  auto input_shape =
      MaybeGetMinimumShape(input.shape(), num_dims, found_unknown_shapes);
  for (const auto& dim : input_shape.dim()) {
    input_size *= dim.size();
  }
  return input_size * DataTypeSize(input.dtype());
}

int64 OpLevelCostEstimator::CalculateInputSize(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  int64 total_input_size = 0;
  for (auto& input : op_features.inputs()) {
    int64 input_size = CalculateSingleInputSize(input, found_unknown_shapes);
    total_input_size += input_size;
    VLOG(1) << "Input Size: " << input_size
            << " Total Input Size:" << total_input_size;
  }
  return total_input_size;
}

int64 OpLevelCostEstimator::CalculateOutputSize(
    const OpInfo& op_features, bool* found_unknown_shapes) const {
  int64 total_output_size = 0;
  // use float as default for calculations
  for (const auto& output : op_features.outputs()) {
    DataType dt = output.dtype();
    const auto& original_output_shape = output.shape();
    int64 output_size = DataTypeSize(dt);
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

Costs OpLevelCostEstimator::PredictConv2D(const OpInfo& op_features) const {
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountConv2DOperations(op_features, &found_unknown_shapes), op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackPropInput(
    const OpInfo& op_features) const {
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackPropInputOperations(
                                  op_features, nullptr, &found_unknown_shapes),
                              op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictConv2DBackPropFilter(
    const OpInfo& op_features) const {
  bool found_unknown_shapes = false;
  auto costs =
      PredictOpCountBasedCost(CountConv2DBackPropFilterOperations(
                                  op_features, nullptr, &found_unknown_shapes),
                              op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictMatMul(const OpInfo& op_features) const {
  bool found_unknown_shapes = false;
  auto costs = PredictOpCountBasedCost(
      CountMatMulOperations(op_features, &found_unknown_shapes), op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

Costs OpLevelCostEstimator::PredictNoOp(const OpInfo& op_features) const {
  VLOG(1) << "Op:" << op_features.op() << " Execution Time 0 (ns)";
  return Costs::ZeroCosts();
}

Costs OpLevelCostEstimator::PredictBatchMatMul(
    const OpInfo& op_features) const {
  bool found_unknown_shapes = false;
  Costs costs = PredictOpCountBasedCost(
      CountBatchMatMulOperations(op_features, &found_unknown_shapes),
      op_features);
  costs.inaccurate = found_unknown_shapes;
  return costs;
}

}  // end namespace grappler
}  // end namespace tensorflow
