/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/dnn.h"

#include <Eigen/Core>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/stream_executor/data_type.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "tsl/platform/ml_dtypes.h"

namespace stream_executor {
namespace dnn {

namespace {

bool ProtoMapIsSubset(const google::protobuf::Map<int64_t, int64_t>& x,
                      const google::protobuf::Map<int64_t, int64_t>& y) {
  for (const auto& ypair : y) {
    const auto it = x.find(ypair.first);
    if (it == x.end() || it->second != ypair.second) return false;
  }
  return true;
}

bool ProtoMapsEqual(const google::protobuf::Map<int64_t, int64_t>& x,
                    const google::protobuf::Map<int64_t, int64_t>& y) {
  return ProtoMapIsSubset(x, y) && ProtoMapIsSubset(y, x);
}

}  // namespace

constexpr DataType ToDataType<tsl::float4_e2m1fn>::value;
constexpr DataType ToDataType<tsl::float8_e3m4>::value;
constexpr DataType ToDataType<tsl::float8_e4m3>::value;
constexpr DataType ToDataType<tsl::float8_e4m3fn>::value;
constexpr DataType ToDataType<tsl::float8_e4m3fnuz>::value;
constexpr DataType ToDataType<tsl::float8_e5m2>::value;
constexpr DataType ToDataType<tsl::float8_e5m2fnuz>::value;
constexpr DataType ToDataType<tsl::float8_e8m0fnu>::value;
constexpr DataType ToDataType<float>::value;
constexpr DataType ToDataType<double>::value;
constexpr DataType ToDataType<Eigen::half>::value;
constexpr DataType ToDataType<Eigen::bfloat16>::value;
constexpr DataType ToDataType<int8_t>::value;
constexpr DataType ToDataType<int32_t>::value;
constexpr DataType ToDataType<int64_t>::value;
constexpr DataType ToDataType<std::complex<float>>::value;
constexpr DataType ToDataType<std::complex<double>>::value;

AlgorithmDesc::AlgorithmDesc(
    int64_t engine_id,
    const std::vector<std::pair<int64_t, int64_t>>& tuning_knobs,
    std::optional<uint64_t> workspace_size) {
  proto_.set_is_cudnn_frontend(true);
  proto_.set_algo_id(engine_id);
  if (workspace_size) {
    proto_.mutable_workspace_size()->set_value(*workspace_size);
  }
  for (const auto& pair : tuning_knobs) {
    (*proto_.mutable_tuning_knobs())[pair.first] = pair.second;
  }
}

uint64_t AlgorithmDesc::hash() const {
  return tsl::DeterministicProtoHash64(proto_);
}

bool AlgorithmDesc::operator==(const AlgorithmDesc& other) const {
  if (is_cudnn_frontend()) {
    return other.is_cudnn_frontend() && algo_id() == other.algo_id() &&
           ProtoMapsEqual(proto_.tuning_knobs(), other.proto_.tuning_knobs());
  }
  return !other.is_cudnn_frontend() && algo_id() == other.algo_id() &&
         tensor_ops_enabled() == other.tensor_ops_enabled();
}

std::string AlgorithmDesc::ToString() const {
  if (is_cudnn_frontend()) {
    // Format similarly to cudnn_frontend::ExecutionPlan::getTag(), e.g.
    // "eng2{k1=2,k3=4}".
    absl::btree_map<int64_t, int64_t> tuning_knobs_sorted;
    absl::c_copy(proto_.tuning_knobs(),
                 std::inserter(tuning_knobs_sorted, tuning_knobs_sorted.end()));
    return absl::StrFormat(
        "eng%d{%s}", proto_.algo_id(),
        absl::StrJoin(
            tuning_knobs_sorted, ",",
            [](std::string* out, const std::pair<int64_t, int64_t>& pair) {
              absl::StrAppendFormat(out, "k%d=%d", pair.first, pair.second);
            }));
  }
  if (tensor_ops_enabled()) {
    return absl::StrCat(algo_id(), "#TC");
  } else {
    return absl::StrCat(algo_id());
  }
}

std::vector<std::pair<int64_t, int64_t>> AlgorithmDesc::TuningKnobs() const {
  std::vector<std::pair<int64_t, int64_t>> result;
  result.reserve(proto_.tuning_knobs().size());
  for (const auto& pair : proto_.tuning_knobs()) {
    result.emplace_back(pair.first, pair.second);
  }
  return result;
}

absl::Status DnnSupport::GetConvolveRunners(
    bool /* use_cudnn_frontend */, dnn::ConvolutionKind /*kind*/,
    dnn::DataType /*input_type*/, dnn::DataType /*output_type*/,
    Stream* /*stream*/, const dnn::BatchDescriptor& /*input_descriptor*/,
    DeviceMemoryBase /*input_data*/,
    const dnn::FilterDescriptor& /*filter_descriptor*/,
    DeviceMemoryBase /*filter_data*/,
    const dnn::BatchDescriptor& /*output_descriptor*/,
    DeviceMemoryBase /*output_data*/,
    const dnn::ConvolutionDescriptor& /*convolution_descriptor*/,
    bool /*use_fallback*/, ScratchAllocator* /*scratch_allocator*/,
    const NumericOptions& /*numeric_options*/,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* /*exec_plans*/) {
  return absl::UnimplementedError("GetConvolveRunners not implemented.");
}

absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
DnnSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  return absl::UnimplementedError("ConvolveRunnerFromDesc not implemented.");
}

absl::Status DnnSupport::GetGraphConvolveRunners(
    dnn::ConvolutionKind /*kind*/, dnn::DataType /*input_type*/,
    dnn::DataType /*output_type*/, Stream* /*stream*/,
    const dnn::BatchDescriptor& /*input_descriptor*/,
    const dnn::FilterDescriptor& /*filter_descriptor*/,
    const dnn::BatchDescriptor& /*output_descriptor*/,
    const dnn::ConvolutionDescriptor& /*convolution_descriptor*/,
    bool /*use_fallback*/, const NumericOptions& /*numeric_options*/,
    std::vector<std::unique_ptr<const dnn::GraphConvRunner>>* /*exec_plans*/,
    std::string /*serialized_graph*/) {
  return absl::UnimplementedError("GetGraphConvolveRunners not implemented.");
}

absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>>
DnnSupport::GraphConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    std::string serialized_graph) {
  return absl::UnimplementedError(
      "GraphConvolveRunnerFromDesc not implemented.");
}

absl::Status DnnSupport::GetFusedConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType element_type, dnn::DataType bias_type,
    dnn::DataType output_type, double conv_input_scale, double side_input_scale,
    double leakyrelu_alpha, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    dnn::ActivationMode activation_mode, const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans) {
  return absl::UnimplementedError("GetFusedConvolveRunners not implemented.");
}

absl::Status DnnSupport::GetFusedMatmulRunners(
    bool use_cudnn_frontend, dnn::DataType element_type,
    dnn::DataType bias_type, dnn::DataType output_type, Stream* stream,
    bool trans_a, bool trans_b, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
    int64_t ldb, int64_t ldc, dnn::ActivationMode activation_mode,
    bool use_fallback, const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
        out_exec_plans) {
  return absl::UnimplementedError("GetFusedMatmulRunners not implemented.");
}

absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
DnnSupport::FusedConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::ActivationMode activation_mode) {
  return absl::UnimplementedError(
      "FusedConvolveRunnerFromDesc not implemented.");
}

absl::StatusOr<std::unique_ptr<const dnn::NormRunner>>
DnnSupport::NormRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::NormKind kind, double epsilon,
    const dnn::TensorDescriptor& x_descriptor,
    const dnn::TensorDescriptor& scale_descriptor,
    const dnn::TensorDescriptor& y_or_dx_descriptor,
    std::optional<dnn::TensorDescriptor> bias_descriptor,
    std::optional<dnn::TensorDescriptor> dy_descriptor,
    std::optional<dnn::TensorDescriptor> expectation_descriptor,
    std::optional<dnn::TensorDescriptor> norm_factor_descriptor,
    std::optional<dnn::TensorDescriptor> dscale_descriptor,
    std::optional<dnn::TensorDescriptor> dbias_descriptor) {
  return absl::UnimplementedError("NormRunnerFromDesc not implemented.");
}

bool DnnSupport::GetMIOpenConvolveAlgorithms(
    dnn::ConvolutionKind /*kind*/, dnn::DataType /*element_type*/,
    dnn::DataType /*output_type*/, Stream* /*stream*/,
    const dnn::BatchDescriptor& /*input_descriptor*/,
    DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& /*filter_descriptor*/,
    DeviceMemoryBase filter_data,
    const dnn::BatchDescriptor& /*output_descriptor*/,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& /*convolution_descriptor*/,
    ScratchAllocator* scratch_allocator,
    std::vector<ProfileResult>* /*out_algorithms*/) {
  return false;
}

bool DnnSupport::GetRnnAlgorithms(std::vector<AlgorithmDesc>* out_algorithms) {
  return false;
}

absl::Status DnnSupport::DoPoolForward(
    DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const NumericOptions& numeric_options,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    ScratchAllocator* workspace_allocator) {
  // Ignore numeric options. Subclasses can override this method to use it.
  return DoPoolForward(element_type, stream, pooling_dimensions,
                       input_dimensions, input_data, output_dimensions,
                       output_data, workspace_allocator);
}

absl::Status DnnSupport::DoPoolBackward(
    DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const NumericOptions& numeric_options,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    DeviceMemoryBase input_diff_data, DeviceMemoryBase output_diff_data,
    ScratchAllocator* workspace_allocator) {
  // Ignore numeric options. Subclasses can override this method to use it.
  return DoPoolBackward(element_type, stream, pooling_dimensions,
                        input_dimensions, input_data, output_dimensions,
                        output_data, input_diff_data, output_diff_data,
                        workspace_allocator);
}

std::string ActivationModeString(ActivationMode mode) {
  switch (mode) {
    case ActivationMode::kNone:
      return "none";
    case ActivationMode::kSigmoid:
      return "sigmoid";
    case ActivationMode::kRelu:
      return "relu";
    case ActivationMode::kRelu6:
      return "relu6";
    case ActivationMode::kReluX:
      return "reluX";
    case ActivationMode::kTanh:
      return "tanh";
    case ActivationMode::kBandPass:
      return "bandpass";
    case ActivationMode::kElu:
      return "elu";
    case ActivationMode::kLeakyRelu:
      return "leakyrelu";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(mode));
  }
}

std::string DataLayoutString(DataLayout layout) {
  switch (layout) {
    case DataLayout::kYXDepthBatch:
      return "YXDepthBatch";
    case DataLayout::kYXBatchDepth:
      return "YXBatchDepth";
    case DataLayout::kBatchYXDepth:
      return "BatchYXDepth";
    case DataLayout::kBatchDepthYX:
      return "BatchDepthYX";
    case DataLayout::kBatchDepthYX4:
      return "BatchDepthYX4";
    case DataLayout::kBatchDepthYX32:
      return "BatchDepthYX32";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(layout));
  }
}

std::string FilterLayoutString(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX:
      return "OutputInputYX";
    case FilterLayout::kOutputYXInput:
      return "OutputYXInput";
    case FilterLayout::kOutputInputYX4:
      return "OutputInputYX4";
    case FilterLayout::kOutputInputYX32:
      return "OutputInputYX32";
    case FilterLayout::kOutputInputYX32_CudnnReordered:
      return "OutputInputYX32_CudnnReordered";
    case FilterLayout::kInputYXOutput:
      return "InputYXOutput";
    case FilterLayout::kYXInputOutput:
      return "YXInputOutput";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(layout));
  }
}

std::string PadAlignmentString(PadAlignment alignment) {
  switch (alignment) {
    case PadAlignment::kDefault:
      return "default";
    case PadAlignment::kCudnnPadding:
      return "cuDNN padding";
    case PadAlignment::kTensorFlowPadding:
      return "TensorFlow padding";
    default:
      return absl::StrCat("unknown: ", static_cast<int32_t>(alignment));
  }
}

std::ostream& operator<<(std::ostream& str, dnn::PadAlignment alignment) {
  return str << PadAlignmentString(alignment);
}

struct ConvDimIndices {
  union {
    struct {
      int depth_idx;
      int batch_idx;
      int spatial_idx;
    } data;
    struct {
      int output_idx;
      int input_idx;
      int spatial_idx;
    } filter;
  };
};

ConvDimIndices GetDimIndices(const DataLayout& layout, const int data_dims) {
  ConvDimIndices dim_indices;
  switch (layout) {
    case DataLayout::kYXBatchDepth:
      dim_indices.data.depth_idx = data_dims - 1;
      dim_indices.data.batch_idx = data_dims - 2;
      dim_indices.data.spatial_idx = 0;
      break;

    case DataLayout::kYXDepthBatch:
      dim_indices.data.depth_idx = data_dims - 2;
      dim_indices.data.batch_idx = data_dims - 1;
      dim_indices.data.spatial_idx = 0;
      break;

    case DataLayout::kBatchYXDepth:
      dim_indices.data.depth_idx = data_dims - 1;
      dim_indices.data.batch_idx = 0;
      dim_indices.data.spatial_idx = 1;
      break;

    case DataLayout::kBatchDepthYX:
    case DataLayout::kBatchDepthYX4:
    case DataLayout::kBatchDepthYX32:
      dim_indices.data.depth_idx = 1;
      dim_indices.data.batch_idx = 0;
      dim_indices.data.spatial_idx = 2;
      break;

    default:
      LOG(FATAL) << "Unknown layout " << layout;
  }

  return dim_indices;
}

ConvDimIndices GetDimIndices(const FilterLayout& layout, const int data_dims) {
  ConvDimIndices dim_indices;
  switch (layout) {
    case FilterLayout::kOutputInputYX:
    case FilterLayout::kOutputInputYX4:
    case FilterLayout::kOutputInputYX32:
    case FilterLayout::kOutputInputYX32_CudnnReordered:
      dim_indices.filter.input_idx = 1;
      dim_indices.filter.output_idx = 0;
      dim_indices.filter.spatial_idx = 2;
      break;

    case FilterLayout::kOutputYXInput:
      dim_indices.filter.input_idx = data_dims - 1;
      dim_indices.filter.output_idx = 0;
      dim_indices.filter.spatial_idx = 1;
      break;

    case FilterLayout::kInputYXOutput:
      dim_indices.filter.input_idx = 0;
      dim_indices.filter.output_idx = data_dims - 1;
      dim_indices.filter.spatial_idx = 1;
      break;

    case FilterLayout::kYXInputOutput:
      dim_indices.filter.input_idx = data_dims - 2;
      dim_indices.filter.output_idx = data_dims - 1;
      dim_indices.filter.spatial_idx = 0;
      break;

    default:
      LOG(FATAL) << "Unknown layout " << layout;
  }

  return dim_indices;
}

std::vector<int64_t> ReorderDims(const std::vector<int64_t>& input,
                                 const DataLayout& from, const DataLayout& to) {
  if (from == to) return input;

  ConvDimIndices from_indices = GetDimIndices(from, input.size());
  ConvDimIndices to_indices = GetDimIndices(to, input.size());

  std::vector<int64_t> reordered(input.size());
  reordered[to_indices.data.batch_idx] = input[from_indices.data.batch_idx];
  reordered[to_indices.data.depth_idx] = input[from_indices.data.depth_idx];

  int spatial_idx_from = from_indices.data.spatial_idx;
  int spatial_idx_to = to_indices.data.spatial_idx;
  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

std::vector<int64_t> ReorderDims(const std::vector<int64_t>& input,
                                 const FilterLayout& from,
                                 const FilterLayout& to) {
  if (from == to) return input;

  ConvDimIndices from_indices = GetDimIndices(from, input.size());
  ConvDimIndices to_indices = GetDimIndices(to, input.size());

  std::vector<int64_t> reordered(input.size());
  reordered[to_indices.filter.output_idx] =
      input[from_indices.filter.output_idx];
  reordered[to_indices.filter.input_idx] = input[from_indices.filter.input_idx];

  int spatial_idx_from = from_indices.filter.spatial_idx;
  int spatial_idx_to = to_indices.filter.spatial_idx;
  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

// -- AlgorithmConfig

std::string AlgorithmConfig::ToString() const {
  std::string algo = "none";
  if (algorithm().has_value()) {
    algo = algorithm()->ToString();
  }
  std::string algo_no_scratch = "none";
  if (algorithm_no_scratch().has_value()) {
    algo_no_scratch = algorithm_no_scratch()->ToString();
  }
  return absl::StrCat(algo, ", ", algo_no_scratch);
}

// -- TensorDescriptor

int TensorDescriptor::ndims() const {
  CHECK_EQ(dimensions_.size(), minor_to_major_.size());
  return dimensions_.size();
}

absl::StatusOr<std::vector<int64_t>>
TensorDescriptor::GetPhysicalDimensionsMajorToMinor() const {
  std::vector<int64_t> logical_to_physical(minor_to_major_.size());
  for (int64_t physical = 0; physical < logical_to_physical.size();
       ++physical) {
    int64_t logical = minor_to_major_.at(minor_to_major_.size() - 1 - physical);
    logical_to_physical[logical] = physical;
  }
  if (dimensions_.size() != minor_to_major_.size())
    return absl::InternalError("Dimensions size should match the layout size.");

  std::vector<int64_t> physical_dims(dimensions_.size());
  for (int64_t i = 0; i < physical_dims.size(); ++i) {
    physical_dims[logical_to_physical[i]] = dimensions_[i];
  }
  return physical_dims;
}

std::vector<int64_t> TensorDescriptor::GetPhysicalStridesMajorToMinor() const {
  std::vector<int64_t> phys_dims = GetPhysicalDimensionsMajorToMinor().value();
  std::vector<int64_t> phys_strides(ndims());
  phys_strides[ndims() - 1] = 1;
  for (int i = ndims() - 2; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return phys_strides;
}

std::vector<int64_t> TensorDescriptor::GetLogicalStrides() const {
  std::vector<int64_t> physical_strides = GetPhysicalStridesMajorToMinor();
  std::reverse(physical_strides.begin(), physical_strides.end());
  std::vector<int64_t> logical_strides(physical_strides.size());
  for (int i = 0; i < ndims(); i++) {
    logical_strides[minor_to_major_[i]] = physical_strides[i];
  }
  return logical_strides;
}

/*static*/ TensorDescriptor TensorDescriptor::For(
    DataType type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> minor_to_major) {
  std::vector<int64_t> dims(dimensions.size());
  std::vector<int64_t> minor_to_major_vec(minor_to_major.size());
  CHECK_EQ(dimensions.size(), minor_to_major.size());
  for (int i = 0; i < dimensions.size(); i++) {
    dims[i] = dimensions[i];
    minor_to_major_vec[i] = minor_to_major[i];
  }
  return TensorDescriptor(type, dims, minor_to_major_vec);
}

std::string TensorDescriptor::ToString() const {
  return absl::StrFormat("{dimensions: %s minor_to_major: %s}",
                         absl::StrJoin(dimensions(), ","),
                         absl::StrJoin(minor_to_major(), ","));
}

// -- MatmulTensorDescriptor

absl::StatusOr<std::vector<int64_t>>
MatmulTensorDescriptor::GetNonContractingDims() const {
  auto nc = xla::GetNonContractingDims(
      tensor_.dimensions().size(), contracting_dim_, batch_dimension_numbers_);
  std::vector<int64_t> non_contracting_dims(nc.begin(), nc.end());

  if (batch_dimension_numbers_.size() + contracting_dim_.size() +
          non_contracting_dims.size() !=
      tensor_.dimensions().size())
    return absl::InternalError(
        "Batch_dimension_numbers, contracting_dim and non_contracting_dims "
        "should sum up to the total number of dimensions.");
  return non_contracting_dims;
}

absl::StatusOr<std::vector<int64_t>>
MatmulTensorDescriptor::MakeCudnnCompatible(const std::vector<int64_t>& vec,
                                            bool is_lhs) const {
  std::vector<int64_t> cudnn_compatible(vec.size());
  int batch_dim_size = batch_dimension_numbers_.size();
  CHECK_LT(batch_dim_size, vec.size());
  for (int i = 0; i < batch_dim_size; i++) {
    cudnn_compatible[i] = vec.at(batch_dimension_numbers_.at(i));
  }
  std::vector<int64_t> non_contracting_dims = GetNonContractingDims().value();
  if (batch_dimension_numbers_.size() + contracting_dim_.size() +
          non_contracting_dims.size() !=
      vec.size())
    return absl::InternalError(
        "Batch_dimension_numbers, contracting_dim and non_contracting_dims "
        "should sum up to the total number of dimensions.");
  if (is_lhs) /* lhs -> {b0, b1,....bk, m, k} */ {
    for (int i = 0; i < non_contracting_dims.size(); i++) {
      cudnn_compatible[batch_dim_size + i] = vec.at(non_contracting_dims.at(i));
    }
    for (int i = 0; i < contracting_dim_.size(); i++) {
      cudnn_compatible[batch_dim_size + non_contracting_dims.size() + i] =
          vec.at(contracting_dim_.at(i));
    }
  } else /* rhs -> {b0, b1, ... bk, k, n} */ {
    for (int i = 0; i < contracting_dim_.size(); i++) {
      cudnn_compatible[batch_dim_size + i] = vec.at(contracting_dim_.at(i));
    }
    for (int i = 0; i < non_contracting_dims.size(); i++) {
      cudnn_compatible[batch_dim_size + contracting_dim_.size() + i] =
          vec.at(non_contracting_dims.at(i));
    }
  }
  return cudnn_compatible;
}

std::vector<int64_t> MatmulTensorDescriptor::GetCudnnCompatibleDimensions(
    bool is_lhs) const {
  std::vector<int64_t> cudnn_compatible_dims =
      MakeCudnnCompatible(tensor_.dimensions(), is_lhs).value();
  return cudnn_compatible_dims;
}

std::vector<int64_t> MatmulTensorDescriptor::GetCudnnCompatibleStrides(
    bool is_lhs) const {
  std::vector<int64_t> cudnn_compatible_strides =
      MakeCudnnCompatible(tensor_.GetLogicalStrides(), is_lhs).value();
  return cudnn_compatible_strides;
}

/*static*/ MatmulTensorDescriptor MatmulTensorDescriptor::For(
    DataType type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> minor_to_major,
    absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims) {
  std::vector<int64_t> batch_dims_vec(batch_dims.size());
  std::vector<int64_t> contracting_dims_vec(contracting_dims.size());
  for (int i = 0; i < batch_dims.size(); i++) {
    batch_dims_vec[i] = batch_dims[i];
  }
  for (int i = 0; i < contracting_dims.size(); i++) {
    contracting_dims_vec[i] = contracting_dims[i];
  }
  return MatmulTensorDescriptor(
      TensorDescriptor::For(type, dimensions, minor_to_major), batch_dims_vec,
      contracting_dims_vec);
}

std::string MatmulTensorDescriptor::ToString() const {
  return absl::StrFormat(
      "{%s, batch_dimension_numbers: %s contracting_dim: %s}",
      tensor_.ToString(), absl::StrJoin(batch_dimension_numbers_, ","),
      absl::StrJoin(contracting_dim_, ","));
}

// -- BatchDescriptor

BatchDescriptor::BatchDescriptor(int ndims)
    : value_max_(0.0),
      value_min_(0.0),
      quantized_activation_mode_(QuantizedActivationMode::k8Bit) {
  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(DataLayout::kYXDepthBatch);
}

BatchDescriptor::BatchDescriptor() : BatchDescriptor(/*ndims=*/2) {}

std::vector<int64_t> BatchDescriptor::full_dims(
    const DataLayout& layout) const {
  std::vector<int64_t> bdyx_dims(ndims() + 2);
  bdyx_dims[0] = count();
  bdyx_dims[1] = feature_map_count();
  std::copy(spatial_size().begin(), spatial_size().end(),
            bdyx_dims.begin() + 2);
  return ReorderDims(bdyx_dims, DataLayout::kBatchDepthYX, layout);
}

std::vector<int64_t> BatchDescriptor::full_strides(
    const DataLayout& layout) const {
  std::vector<int64_t> phys_dims = full_dims(this->layout());
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[ndims() + 1] = 1;
  for (int i = ndims(); i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

std::vector<int64_t> BatchDescriptor::vectorized_dims(const DataLayout& layout,
                                                      int vector_size,
                                                      int vector_dim) const {
  std::vector<int64_t> bdyx_dims = full_dims(dnn::DataLayout::kBatchDepthYX);
  if (vector_dim != -1) {
    bdyx_dims[vector_dim] /= vector_size;
  }
  return dnn::ReorderDims(bdyx_dims, dnn::DataLayout::kBatchDepthYX, layout);
}

std::vector<int64_t> BatchDescriptor::vectorized_strides(
    const DataLayout& layout, int vector_size, int vector_dim) const {
  std::vector<int64_t> phys_dims =
      vectorized_dims(this->layout(), vector_size, vector_dim);
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[phys_dims.size() - 1] = 1;
  for (int i = phys_dims.size() - 2; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

std::string BatchDescriptor::ToString() const {
  std::string spatial;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }
  return absl::StrFormat(
      "{count: %d feature_map_count: %d spatial: %s "
      "value_min: %f value_max: %f layout: %s}",
      count(), feature_map_count(), spatial, value_min_, value_max_,
      DataLayoutString(layout()));
}

std::string BatchDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  std::string depth = absl::StrCat("d", feature_map_count());
  std::string batch = absl::StrCat("b", count());

  std::string spatial = "s";
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }

  std::string suffix;
  if (value_min() != value_max()) {
    absl::StrAppend(&suffix, "[", value_min(), ";", value_max(), "]");
  }
  if (quantized_activation_mode() == QuantizedActivationMode::k16Bit) {
    suffix += "_16bit";
  }

  switch (layout()) {
    case DataLayout::kYXDepthBatch:
      return absl::StrCat(spatial, depth, batch, suffix);
    case DataLayout::kYXBatchDepth:
      return absl::StrCat(spatial, batch, depth, suffix);
    case DataLayout::kBatchYXDepth:
      return absl::StrCat(batch, spatial, depth, suffix);
    case DataLayout::kBatchDepthYX:
      return absl::StrCat(batch, depth, spatial, suffix);
    case DataLayout::kBatchDepthYX4:
    case DataLayout::kBatchDepthYX32:
      return absl::StrCat(batch, depth, spatial, suffix, "(VECT_C)");
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32_t>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64_t BatchDescriptor::NodesPerFeatureMap() const {
  int64_t ret = 1;
  for (int i = 0; i < ndims(); i++) {
    ret *= spatial_size()[i];
  }
  return ret;
}

int64_t BatchDescriptor::NodesAcrossFeatureMaps() const {
  return NodesPerFeatureMap() * feature_map_count();
}

TensorDescriptorProto BatchDescriptor::ToProto(DataType data_type) const {
  CHECK_EQ(0.0, value_max_);
  CHECK_EQ(0.0, value_min_);
  CHECK(quantized_activation_mode_ == QuantizedActivationMode::k8Bit);

  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- FilterDescriptor

FilterDescriptor::FilterDescriptor(int ndims) {
  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(FilterLayout::kOutputInputYX);
}

FilterDescriptor::FilterDescriptor() : FilterDescriptor(/*ndims=*/2) {}

FilterDescriptor::~FilterDescriptor() {}

std::string FilterDescriptor::ToString() const {
  std::string desc = absl::StrFormat(
      "{output_feature_map_count: %d input_feature_map_count: %d "
      "layout: %s shape: ",
      output_feature_map_count(), input_feature_map_count(),
      FilterLayoutString(layout()));
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "%d ", input_filter_dims()[i]);
  }
  absl::StrAppend(&desc, "}");

  return desc;
}

std::vector<int64_t> FilterDescriptor::full_dims(
    const FilterLayout& layout) const {
  std::vector<int64_t> oiyx_dims(ndims() + 2);
  oiyx_dims[0] = output_feature_map_count();
  oiyx_dims[1] = input_feature_map_count();
  std::copy(input_filter_dims().begin(), input_filter_dims().end(),
            oiyx_dims.begin() + 2);
  return ReorderDims(oiyx_dims, FilterLayout::kOutputInputYX, layout);
}

std::vector<int64_t> FilterDescriptor::full_strides(
    const FilterLayout& layout) const {
  std::vector<int64_t> phys_dims = full_dims(this->layout());
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[ndims() + 1] = 1;
  for (int i = ndims(); i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

std::vector<int64_t> FilterDescriptor::vectorized_dims(
    const FilterLayout& layout, int vector_size, int vector_dim) const {
  std::vector<int64_t> oiyx_dims = full_dims(dnn::FilterLayout::kOutputInputYX);
  if (vector_dim != -1) {
    oiyx_dims[vector_dim] /= vector_size;
  }
  return ReorderDims(oiyx_dims, FilterLayout::kOutputInputYX, layout);
}

std::vector<int64_t> FilterDescriptor::vectorized_strides(
    const FilterLayout& layout, int vector_size, int vector_dim) const {
  std::vector<int64_t> phys_dims =
      vectorized_dims(this->layout(), vector_size, vector_dim);
  std::vector<int64_t> phys_strides(phys_dims.size());
  phys_strides[phys_dims.size() - 1] = 1;
  for (int i = phys_dims.size() - 2; i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

TensorDescriptorProto FilterDescriptor::ToProto(DataType data_type) const {
  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- ConvolutionDescriptor

ConvolutionDescriptor::ConvolutionDescriptor(int ndims) {
  proto_.mutable_paddings()->Resize(ndims, 0);
  proto_.mutable_strides()->Resize(ndims, 1);
  proto_.mutable_dilations()->Resize(ndims, 1);
  proto_.set_group_count(1);
  proto_.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
}

ConvolutionDescriptor::ConvolutionDescriptor()
    : ConvolutionDescriptor(/*ndims=*/2) {}

ConvolutionDescriptor::~ConvolutionDescriptor() {}

std::string ConvolutionDescriptor::ToString() const {
  std::string padding;
  std::string strides;
  std::string dilations;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&padding, "%d ", this->padding()[i]);
    absl::StrAppendFormat(&strides, "%d ", this->strides()[i]);
    absl::StrAppendFormat(&dilations, "%d ", this->dilations()[i]);
  }

  return absl::StrFormat(
      "{zero_padding: %s pad_alignment: %s filter_strides: %s dilation_rates: "
      "%s}",
      padding, PadAlignmentString(pad_alignment()), strides, dilations);
}

// -- PoolingDescriptor

PoolingDescriptor::PoolingDescriptor(int ndims)
    : mode_(dnn::PoolingMode::kMaximum),
      ndims_(ndims),
      propagate_nans_(false),
      window_(ndims, 0),
      padding_(ndims, 0),
      strides_(ndims, 1) {}

PoolingDescriptor::PoolingDescriptor() : PoolingDescriptor(/*ndims=*/2) {}

// -- NormalizeDescriptor

NormalizeDescriptor::NormalizeDescriptor()
    : bias_(0.0),
      range_(0),
      alpha_(0.0),
      beta_(0.0),
      wrap_around_(false),
      segment_size_(0) {}

bool DnnSupport::IsStatusOk(const absl::Status& status, bool report_error) {
  if (status.ok()) {
    return true;
  }
  if (report_error) {
    LOG(ERROR) << status.message();
  }
  return false;
}

absl::Status DnnSupport::DoCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const RnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const RnnStateTensorDescriptor& grads_desc, DeviceMemoryBase grads_data,
    DeviceMemory<uint8_t> scratch_memory, int ctc_loss_algo_id) {
  return absl::UnimplementedError("CtcLoss not implemented");
}

}  // namespace dnn
}  // namespace stream_executor
