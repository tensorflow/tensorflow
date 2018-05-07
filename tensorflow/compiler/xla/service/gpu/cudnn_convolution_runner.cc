/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::Stream;
using se::dnn::AlgorithmConfig;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::DimIndex;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;
using se::dnn::ProfileResult;

// A StreamExecutor ScratchAllocator that wraps a single XLA allocation,
// returning it (in its entirety) the first time Allocate() is called.
class ScratchBufAllocator : public se::ScratchAllocator {
 public:
  explicit ScratchBufAllocator(se::DeviceMemoryBase scratch)
      : scratch_(scratch) {}

  ~ScratchBufAllocator() override = default;

  int64 GetMemoryLimitInBytes(se::Stream* /*stream*/) override {
    return scratch_.size();
  }

  se::port::StatusOr<DeviceMemory<uint8>> AllocateBytes(
      se::Stream* stream, int64 byte_size) override {
    if (allocated_) {
      return se::port::InternalError(
          "Can't allocate twice from a ScratchBufAllocator.");
    }
    if (byte_size > scratch_.size()) {
      return se::port::InternalError(tensorflow::strings::StrCat(
          "Can't allocate ", byte_size,
          " bytes from a ScratchBufAllocator of size ", scratch_.size()));
    }

    allocated_ = true;
    return se::DeviceMemory<uint8>(scratch_);
  }

 private:
  se::DeviceMemoryBase scratch_;
  bool allocated_ = false;
};

template <typename T>
Status RunCudnnConvolution(
    CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
    const Shape& output_shape, DeviceMemory<T> input_buf,
    DeviceMemory<T> filter_buf, DeviceMemory<T> output_buf,
    se::ScratchAllocator* scratch_allocator, const Window& window,
    const ConvolutionDimensionNumbers& dnums, AlgorithmConfig algorithm,
    Stream* stream, ProfileResult* profile_result /*= nullptr*/) {
  VLOG(3) << "Convolution Algorithm: " << algorithm.algorithm().algo_id();
  VLOG(3) << "tensor_ops_enabled: "
          << algorithm.algorithm().tensor_ops_enabled();
  VLOG(3) << "Convolution kind: " << CudnnConvKindToString(kind);
  VLOG(3) << "input shape: { " << ShapeUtil::HumanString(input_shape) << " }";
  VLOG(3) << "filter shape: { " << ShapeUtil::HumanString(filter_shape) << " }";
  VLOG(3) << "Output shape: { " << ShapeUtil::HumanString(output_shape) << " }";
  VLOG(3) << "Window: { " << window.ShortDebugString() << " }";
  VLOG(3) << "Dim nums: { " << dnums.ShortDebugString() << " }";

  const int num_dimensions = window.dimensions_size();
  CHECK_LE(num_dimensions, 3);
  // cuDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  if (std::is_same<T, float>::value) {
    CHECK_EQ(F32, output_shape.element_type())
        << ShapeUtil::HumanString(output_shape);
  } else if (std::is_same<T, Eigen::half>::value) {
    CHECK_EQ(F16, output_shape.element_type())
        << ShapeUtil::HumanString(output_shape);
  } else {
    LOG(FATAL) << ShapeUtil::HumanString(output_shape);
  }

  CHECK_EQ(num_dimensions, dnums.input_spatial_dimensions_size());
  CHECK_EQ(num_dimensions, dnums.kernel_spatial_dimensions_size());
  CHECK_EQ(num_dimensions, dnums.output_spatial_dimensions_size());
  for (const WindowDimension& dim : window.dimensions()) {
    CHECK_EQ(dim.padding_low(), dim.padding_high());
  }

  // cuDNN's convolution APIs support the BDYX layout for activations/output and
  // the OIYX layout for weights.
  BatchDescriptor input_descriptor(effective_num_dimensions);
  input_descriptor.set_layout(DataLayout::kBatchDepthYX)
      .set_feature_map_count(
          input_shape.dimensions(dnums.input_feature_dimension()))
      .set_count(input_shape.dimensions(dnums.input_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    // Note that the dimensions are reversed. The same holds below.
    input_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        input_shape.dimensions(dnums.input_spatial_dimensions(dim)));
  }

  FilterDescriptor filter_descriptor(effective_num_dimensions);
  filter_descriptor.set_layout(FilterLayout::kOutputInputYX)
      .set_input_feature_map_count(
          filter_shape.dimensions(dnums.kernel_input_feature_dimension()))
      .set_output_feature_map_count(
          filter_shape.dimensions(dnums.kernel_output_feature_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    filter_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        filter_shape.dimensions(dnums.kernel_spatial_dimensions(dim)));
  }

  ConvolutionDescriptor convolution_descriptor(effective_num_dimensions);
  for (int dim = 0; dim < num_dimensions; ++dim) {
    convolution_descriptor
        .set_zero_padding(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).padding_low())
        .set_filter_stride(
            static_cast<DimIndex>(effective_num_dimensions - dim - 1),
            window.dimensions(dim).stride());
  }

  BatchDescriptor output_descriptor(effective_num_dimensions);
  output_descriptor.set_layout(DataLayout::kBatchDepthYX)
      .set_feature_map_count(
          output_shape.dimensions(dnums.output_feature_dimension()))
      .set_count(output_shape.dimensions(dnums.output_batch_dimension()));
  for (int dim = 0; dim < num_dimensions; ++dim) {
    output_descriptor.set_spatial_dim(
        static_cast<DimIndex>(effective_num_dimensions - dim - 1),
        output_shape.dimensions(dnums.output_spatial_dimensions(dim)));
  }

  // Add a singleton dimension in the 1D convolution case.
  if (num_dimensions == 1) {
    input_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    output_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    filter_descriptor.set_spatial_dim(static_cast<DimIndex>(0), 1);
    convolution_descriptor.set_zero_padding(static_cast<DimIndex>(0), 0)
        .set_filter_stride(static_cast<DimIndex>(0), 1);
  }

  switch (kind) {
    case CudnnConvKind::kForward:
      stream->ThenConvolveWithAlgorithm(
          input_descriptor, input_buf, filter_descriptor, filter_buf,
          convolution_descriptor, output_descriptor, &output_buf,
          scratch_allocator, algorithm, profile_result);
      break;
    case CudnnConvKind::kBackwardInput:
      stream->ThenConvolveBackwardDataWithAlgorithm(
          filter_descriptor, filter_buf, output_descriptor, output_buf,
          convolution_descriptor, input_descriptor, &input_buf,
          scratch_allocator, algorithm, profile_result);
      break;
    case CudnnConvKind::kBackwardFilter:
      stream->ThenConvolveBackwardFilterWithAlgorithm(
          input_descriptor, input_buf, output_descriptor, output_buf,
          convolution_descriptor, filter_descriptor, &filter_buf,
          scratch_allocator, algorithm, profile_result);
      break;
  }

  if (!stream->ok()) {
    return InternalError(
        "Unable to launch convolution with type %s and algorithm (%lld, %lld)",
        CudnnConvKindToString(kind).c_str(), algorithm.algorithm().algo_id(),
        algorithm.algorithm_no_scratch().algo_id());
  }
  return Status::OK();
}

}  // anonymous namespace

string CudnnConvKindToString(CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kForward:
      return "forward";
    case CudnnConvKind::kBackwardFilter:
      return "backward_filter";
    case CudnnConvKind::kBackwardInput:
      return "backward_input";
  }
}

Status RunCudnnConvolution(
    CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
    const Shape& output_shape, se::DeviceMemoryBase input_buf,
    se::DeviceMemoryBase filter_buf, se::DeviceMemoryBase output_buf,
    se::DeviceMemoryBase scratch_buf, const Window& window,
    const ConvolutionDimensionNumbers& dnums,
    se::dnn::AlgorithmConfig algorithm, se::Stream* stream,
    se::dnn::ProfileResult* profile_result) {
  ScratchBufAllocator scratch_allocator(scratch_buf);
  return RunCudnnConvolution(kind, input_shape, filter_shape, output_shape,
                             input_buf, filter_buf, output_buf,
                             &scratch_allocator, window, dnums, algorithm,
                             stream, profile_result);
}

Status RunCudnnConvolution(
    CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
    const Shape& output_shape, se::DeviceMemoryBase input_buf,
    se::DeviceMemoryBase filter_buf, se::DeviceMemoryBase output_buf,
    se::ScratchAllocator* scratch_allocator, const Window& window,
    const ConvolutionDimensionNumbers& dnums,
    se::dnn::AlgorithmConfig algorithm, se::Stream* stream,
    se::dnn::ProfileResult* profile_result) {
  PrimitiveType output_primitive_type = output_shape.element_type();
  CHECK(output_primitive_type == F32 || output_primitive_type == F16)
      << ShapeUtil::HumanString(output_shape);
  if (output_primitive_type == F32) {
    return RunCudnnConvolution(
        kind, input_shape, filter_shape, output_shape,
        se::DeviceMemory<float>(input_buf), se::DeviceMemory<float>(filter_buf),
        se::DeviceMemory<float>(output_buf), scratch_allocator, window, dnums,
        algorithm, stream, profile_result);
  }
  return RunCudnnConvolution(kind, input_shape, filter_shape, output_shape,
                             se::DeviceMemory<Eigen::half>(input_buf),
                             se::DeviceMemory<Eigen::half>(filter_buf),
                             se::DeviceMemory<Eigen::half>(output_buf),
                             scratch_allocator, window, dnums, algorithm,
                             stream, profile_result);
}

}  // namespace gpu
}  // namespace xla
