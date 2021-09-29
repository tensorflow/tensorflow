/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <limits>
#ifdef GOOGLE_CUDA&& GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

namespace {

// The parameters for convolution when implementing resize using
// the LHS-dilated convolution strategy, which is equivelent to the sequence
// (TransposedConv, Slice) in TRT.
template <size_t NumSpatialDims>
struct ResizeConvolutionParameters {
  // Spatial dimension for the resize filter.
  std::array<int64_t, NumSpatialDims> filter_size;

  // This corresponds to "stride" parameter in TRT's IDeconvolutionLayer. The
  // XLA term is LHS dilation.
  std::array<int64_t, NumSpatialDims> lhs_dilation;

  // Padding parameters
  std::array<int64_t, NumSpatialDims> upper_padding;
  std::array<int64_t, NumSpatialDims> lower_padding;

  // The amount to extend-pad the input on the lower side.
  std::array<int64_t, NumSpatialDims> extension_size;

  // Parameters for the final slice operation.
  std::array<int64_t, NumSpatialDims> slice_stride;
  std::array<int64_t, NumSpatialDims> slice_start;
  std::array<int64_t, NumSpatialDims> slice_size;
};

// Returns an initialized struct.
template <size_t NumSpatialDims = 2>
StatusOr<ResizeConvolutionParameters<NumSpatialDims>>
CreateResizeConvParameters(absl::Span<const int64_t> in_size,
                           absl::Span<const int64_t> out_size,
                           const int64_t num_input_channels,
                           bool align_corners) {
  TRT_ENSURE(in_size.size() == out_size.size());
  TRT_ENSURE(in_size.size() == NumSpatialDims);

  // This upper padding formula comes from TF2XLA, we use it as a blackbox here.
  auto calculate_upper_padding = [](int64_t in_size, int64_t out_size,
                                    int64_t kernel_size, int64_t stride) {
    int64_t padding = (2 * kernel_size - 1) + (out_size - 1) * stride -
                      (kernel_size - 1) - 1 - (kernel_size * (in_size - 1));
    return padding;
  };

  ResizeConvolutionParameters<NumSpatialDims> dims;
  for (int i = 0; i < NumSpatialDims; ++i) {
    if (in_size[i] == 1 || out_size[i] == 1) {
      dims.slice_stride[i] = 1;
      dims.filter_size[i] = 1;
    } else {
      // The scaling factor changes depending on the alignment of corners.
      const int64_t in_size_factor =
          align_corners ? in_size[i] - 1 : in_size[i];
      const int64_t out_size_factor =
          align_corners ? out_size[i] - 1 : out_size[i];
      // The efficiency of the Conv-Stride workaround depends on the following
      // GCD being high. A low GCD implies a high stride. Since TRT does not
      // enable non-unit stride on the output of an input-dilated conv
      // (transposed conv), we implement the stide after the conv finishes.
      // Thus, the stide parameter here is directly proportional to the memory
      // required to hold the output of the result of the convolution.
      int64_t gcd = MathUtil::GCD(static_cast<uint64_t>(in_size_factor),
                                  static_cast<uint64_t>(out_size_factor));
      dims.slice_stride[i] = in_size_factor / gcd;
      dims.filter_size[i] = out_size_factor / gcd;
    }

    dims.upper_padding[i] = dims.filter_size[i] - 1;
    dims.lower_padding[i] = dims.filter_size[i] - 1;
    dims.lhs_dilation[i] = dims.filter_size[i];
    dims.extension_size[i] = 0;

    if (!align_corners) {
      dims.upper_padding[i] = calculate_upper_padding(
          in_size[i], out_size[i], dims.filter_size[i], dims.slice_stride[i]);
      dims.extension_size[i] = dims.upper_padding[i] / dims.filter_size[i];
      dims.upper_padding[i] = calculate_upper_padding(
          in_size[i] + dims.extension_size[i], out_size[i], dims.filter_size[i],
          dims.slice_stride[i]);
    }

    dims.slice_start[i] = (!align_corners && dims.extension_size[i] > 0)
                              ? dims.slice_stride[i]
                              : 0;
    dims.slice_size[i] = out_size[i];
  }
  return dims;
}

// Returns the approximate cost of a resize operation in one dimension. See the
// explanation in CreateResizeConvParameters for the rationale behind using
// stride as cost.
StatusOr<int64_t> GetResizeCost(int a, int b, bool align_corners) {
  StatusOr<ResizeConvolutionParameters<2>> params =
      CreateResizeConvParameters<2>({a, 1}, {b, 1}, 1, align_corners);
  TRT_ENSURE_OK(params);
  return params->slice_stride[0];
}

// Create a 1D triangular filter. The data vector is resized to 2*n-1.
Status MakeTriFilter1D(const int64_t n, std::vector<float>* data) {
  TRT_ENSURE(n > 0);
  TRT_ENSURE(data != nullptr);
  data->resize((2 * n) - 1);
  for (int i = 0; i < n; i++) {
    (*data)[i] = static_cast<float>(i + 1) / static_cast<float>(n);
    const auto i2 = data->size() - (i + 1);
    (*data)[i2] = (*data)[i];
  }
  return Status::OK();
}

// Creates a bilinear resize kernel for the TransposedConv-Stride workaround
// strategy. We do this by taking outer product of two 1D filters.
// Note that the current compilation of the workaround always splits the kernel,
// so typically we will not use 2D filters.
Status MakeGeneralResizeKernel(const std::array<int64_t, 2>& kernel_size,
                               int64_t channels, std::vector<float>* data) {
  TRT_ENSURE(data != nullptr);
  TRT_ENSURE(kernel_size[0] > 0 && kernel_size[1] > 0 && channels > 0);
  std::vector<float> a;
  TF_RETURN_IF_ERROR(MakeTriFilter1D(kernel_size[0], &a));
  std::vector<float> b;
  TF_RETURN_IF_ERROR(MakeTriFilter1D(kernel_size[1], &b));
  data->resize(a.size() * b.size() * channels);
  // Compute outer product (a * b^T) and broadcast over all channels of c.
  for (int row = 0; row < a.size(); ++row) {
    for (int col = 0; col < b.size(); ++col) {
      const float val = a[row] * b[col];
      for (int channel = 0; channel < channels; ++channel) {
        (*data)[channel * a.size() * b.size() + row * b.size() + col] = val;
      }
    }
  }
  return Status::OK();
}

// Implements bilinear resize using "LHS-dilated convolution" workaround. This
// function implements a single resize command by composing 1D convolutions and
// slicing the result at the end. A single TF bilinear resize operation may be
// broken up into multiple calls to this function to minimize maximum overall
// memory cost of this function's kernels.
template <size_t NumSpatialDims>
StatusOr<nvinfer1::ITensor*> ResizeUsingDilationAndConvolution(
    TRTNetworkBuilder* builder, nvinfer1::ITensor* input,
    const std::array<int64_t, NumSpatialDims>& out_size,
    const bool align_corners, bool implicit_batch_mode) {
  TRT_ENSURE(builder != nullptr);
  const nvinfer1::Dims& input_dims = input->getDimensions();
  TRT_ENSURE(input_dims.nbDims ==
             NumSpatialDims + (implicit_batch_mode ? 1 : 2));

  int64_t num_input_channels =
      input->getDimensions().d[implicit_batch_mode ? 0 : 1];
  TRT_ENSURE(num_input_channels > 0);

  std::array<int64_t, NumSpatialDims> in_size;
  for (int i = 0; i < NumSpatialDims; i++) {
    in_size[i] = input_dims.d[input_dims.nbDims - NumSpatialDims + i];
  }

  StatusOr<ResizeConvolutionParameters<2>> params =
      CreateResizeConvParameters<2>(
          absl::Span<const int64_t>(in_size.data(), in_size.size()),
          absl::Span<const int64_t>(out_size.data(), out_size.size()),
          num_input_channels, align_corners);

  nvinfer1::ITensor* input_data = input;
  if (!align_corners) {
    // Extend-pad the input.
    StatusOr<nvinfer1::ITensor*> result =
        builder->MaybeConstantExtendSides2D(input_data, params->extension_size);
    TRT_ENSURE_OK(result);
    input_data = *result;
  }

  for (int i = 0; i < in_size.size(); i++) {
    if (in_size[i] != out_size[i]) {
      // Do the convolution
      auto set_param = [i](auto* dst, const auto& src, int64_t default_val) {
        *dst = src;
        (*dst)[(i + 1) % NumSpatialDims] = default_val;
      };

      TRTNetworkBuilder::TransposedConvolutionSpec<NumSpatialDims> spec;
      set_param(&spec.kernel_size, params->filter_size, 1);
      set_param(&spec.lower_padding, params->lower_padding, 0);
      set_param(&spec.upper_padding, params->upper_padding, 0);
      set_param(&spec.stride, params->lhs_dilation, 1);

      // Create the filter.
      std::vector<float> weights;
      TF_RETURN_IF_ERROR(MakeGeneralResizeKernel(spec.kernel_size,
                                                 num_input_channels, &weights));
      // Set the kernel size to the actual size rather than the parameter used
      // for triangle filter.
      spec.kernel_size[i] = params->filter_size[i] * 2 - 1;
      spec.kernel_size[(i + 1) % NumSpatialDims] = 1;
      StatusOr<nvinfer1::IDeconvolutionLayer*> conv =
          builder->TransposedConvolution<2>(input_data, num_input_channels,
                                            spec, weights, num_input_channels);
      TRT_ENSURE_PTR_OK(conv);
      input_data = (*conv)->getOutput(0);
    }
  }

  // Do the slice.
  const nvinfer1::Dims& pre_slice_dims = input_data->getDimensions();
  const int64_t spatial_dims_offset = pre_slice_dims.nbDims - NumSpatialDims;

  nvinfer1::Dims slice_start{pre_slice_dims.nbDims, {}};
  std::fill_n(slice_start.d, slice_start.nbDims, 0);
  absl::c_copy(params->slice_start, slice_start.d + spatial_dims_offset);

  nvinfer1::Dims slice_size = pre_slice_dims;
  absl::c_copy(params->slice_size, slice_size.d + spatial_dims_offset);

  nvinfer1::Dims slice_stride{pre_slice_dims.nbDims, {}};
  std::fill_n(slice_stride.d, slice_stride.nbDims, 1);
  absl::c_copy(params->slice_stride, slice_stride.d + spatial_dims_offset);

  StatusOr<nvinfer1::ISliceLayer*> slice =
      builder->Slice(input_data, slice_start, slice_size, slice_stride);
  TRT_ENSURE_OK(slice);
  return (*slice)->getOutput(0);
}

// Given (N)CHW dimensions for the input tensor, returns new dimensions
// identical to input tensor dims, but with spatial dimensions replaced with
// those given.
StatusOr<nvinfer1::Dims> GetStaticOutputDims(
    const nvinfer1::Dims& input_tensor_dims,
    const std::array<int, 2>& output_spatial_dims) {
  nvinfer1::Dims output_shape_dims;
  output_shape_dims.nbDims = input_tensor_dims.nbDims;
  for (int i = 0; i < output_shape_dims.nbDims; ++i) {
    output_shape_dims.d[i] = input_tensor_dims.d[i];
  }
  output_shape_dims.d[output_shape_dims.nbDims - 2] = output_spatial_dims[0];
  output_shape_dims.d[output_shape_dims.nbDims - 1] = output_spatial_dims[1];
  return output_shape_dims;
}

// Checks whether the dimensions of given (N)HWC dims are statically specified
// (not -1) and optionally returns the static shape.
StatusOr<bool> HWCDimsStaticSpatialShape(
    const nvinfer1::Dims& nhwc_dims,
    std::array<int64_t, 2>* static_spatial_shape) {
  TRT_ENSURE(nhwc_dims.nbDims == 4 || nhwc_dims.nbDims == 3);
  const int first_dim = nhwc_dims.nbDims == 4 ? 1 : 0;
  if (nhwc_dims.d[first_dim] != -1 && nhwc_dims.d[first_dim + 1] != -1) {
    if (static_spatial_shape != nullptr) {
      (*static_spatial_shape)[0] = nhwc_dims.d[first_dim];
      (*static_spatial_shape)[1] = nhwc_dims.d[first_dim + 1];
    }
    return true;
  }
  return false;
}

// This function either computes the output size value or constructs the
// computation graph for the dynamic output size value, then sets the correct
// parameters for the given IResizeLayer. The output is either a set of
// numbers computable at build time, or it is a dynamic value which is
// represented by a shape tensor.
StatusOr<std::pair<nvinfer1::ITensor*, nvinfer1::Dims>> ComputeOutputShape(
    ITensorProxyPtr input_tensor, TRT_TensorOrWeights out_size,
    TRTNetworkBuilder* builder, bool has_static_input_shape,
    bool has_static_output_shape) {
  TRT_ENSURE(builder != nullptr)
  const nvinfer1::Dims& input_dims = input_tensor->getDimensions();
  bool dynamic_input_shape = !HasStaticShape(input_dims);

  // For both static shapes, we can make a quick shortcut.
  if (!dynamic_input_shape && has_static_output_shape) {
    const int* weights_ptr = out_size.weights().GetPointer<int>();
    StatusOr<nvinfer1::Dims> output_shape_dims = GetStaticOutputDims(
        input_tensor->getDimensions(), {weights_ptr[0], weights_ptr[1]});
    TRT_ENSURE_OK(output_shape_dims);
    return std::pair<nvinfer1::ITensor*, nvinfer1::Dims>(nullptr,
                                                         *output_shape_dims);
  }

  // For dynamic shape, build the output shape as a shape tensor that will be
  // computed at run time. The batch size and num of channels will be copied
  // from the input shape, which may be either static or dynamic.
  nvinfer1::ITensor* batch_and_chan_dim{nullptr};
  if (dynamic_input_shape) {
    StatusOr<nvinfer1::IGatherLayer*> batch_and_chan_gather =
        builder->GatherDims(input_tensor->trt_tensor(),
                            /*indices=*/{0, 1});
    TRT_ENSURE_PTR_OK(batch_and_chan_gather);
    batch_and_chan_dim = (*batch_and_chan_gather)->getOutput(0);
  } else {
    const nvinfer1::Dims& input_dims = input_tensor->getDimensions();
    StatusOr<nvinfer1::IConstantLayer*> shape_const = builder->ConstantShape(
        nvinfer1::Dims{2, {input_dims.d[0], input_dims.d[1]}});
    TRT_ENSURE_PTR_OK(shape_const);
    batch_and_chan_dim = (*shape_const)->getOutput(0);
  }

  // The height and width will be obtained from the requested output size.
  nvinfer1::ITensor* spatial_dims{nullptr};
  if (has_static_output_shape) {
    const int* weights_ptr = out_size.weights().GetPointer<int>();
    StatusOr<nvinfer1::IConstantLayer*> output_shape_spatial =
        builder->Constant(std::vector<int>{weights_ptr[0], weights_ptr[1]});
    TRT_ENSURE_PTR_OK(output_shape_spatial);
    spatial_dims = (*output_shape_spatial)->getOutput(0);
  } else {
    spatial_dims = out_size.tensor()->trt_tensor();
  }
  StatusOr<nvinfer1::IConcatenationLayer*> result =
      builder->Concat({batch_and_chan_dim, spatial_dims}, 0);
  TRT_ENSURE_PTR_OK(result);

  nvinfer1::ITensor* output_shape_tensor{nullptr};
  output_shape_tensor = (*result)->getOutput(0);
  TRT_ENSURE(output_shape_tensor != nullptr);
  return std::pair<nvinfer1::ITensor*, nvinfer1::Dims>(output_shape_tensor,
                                                       nvinfer1::Dims{});
}

}  // namespace

class ConvertResize : public OpConverterBase<ConvertResize> {
 public:
  explicit ConvertResize(OpConverterParams* params)
      : OpConverterBase<ConvertResize>(params) {}

  struct ResizeAttributes {
    nvinfer1::ResizeMode mode{nvinfer1::ResizeMode::kNEAREST};
    bool align_corners{false};
    bool half_pixel_centers{false};

    bool has_static_output_shape{false};
    bool has_static_input_shape{false};

    // Spatial input/output dimensions in the static case.
    std::array<int64_t, 2> static_input_size;
    std::array<int64_t, 2> static_output_size;

    uint64_t num_channels;
  };

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return {InputArgSpec::Create("input", TrtInputArg::kTensor),
            InputArgSpec::Create("size", TrtInputArg::kBoth)};
  }

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  Status Validate() {
    const auto& inputs = params_->inputs;
    const auto& node_def = params_->node_def;

    // Get input tensor.
    input_tensor_ = inputs.at(0).tensor();
    TRT_ENSURE(input_tensor_ != nullptr);

    // Check whether the spatial dimensions are static.
    const nvinfer1::Dims& input_dims = input_tensor_->getDimensions();
    StatusOr<bool> has_static_spatial_input_shape =
        HWCDimsStaticSpatialShape(input_dims, &attrs_.static_input_size);
    TRT_ENSURE_OK(has_static_spatial_input_shape);
    attrs_.has_static_input_shape = *has_static_spatial_input_shape;

    attrs_.has_static_output_shape = inputs.at(1).is_weights();
    if (attrs_.has_static_output_shape) {
      const auto* size_ptr = inputs.at(1).weights().GetPointer<int>();
      attrs_.static_output_size = {size_ptr[0], size_ptr[1]};
    }

    // Check output size. It must constain two values i.e. [H_out, W_out]
    if (inputs.at(1).is_weights()) {
      // Output size is given as a constant.
      if (inputs.at(1).weights().count() != 2) {
        return errors::Unimplemented(
            "Resize requires a 2D value for the size, at ", node_def.name());
      }
    } else {
      // Output size is given as a tensor, possibly as the result of shape
      // calculation ops in the graph.
      if (params_->use_implicit_batch) {
        return errors::Unimplemented(
            "Resize requires constant output size in implicit batch mode, at ",
            node_def.name());
      }

      // Check that this is a shape tensor and it has two values.
      ITensorProxyPtr size_param = inputs.at(1).tensor();
      TRT_ENSURE(size_param->getType() == nvinfer1::DataType::kINT32);
      TRT_ENSURE(size_param->getDimensions().nbDims == 1);
      if (size_param->getDimensions().d[0] != 2) {
        return errors::Unimplemented(
            "Resize requires a 2D value for the size, at ", node_def.name());
      }
    }

    // Verify and consume node attributes.
    StatusOr<bool> align_corners = GetAttrValue<bool>("align_corners");
    TRT_ENSURE_OK(align_corners);
    attrs_.align_corners = *align_corners;
    StatusOr<bool> half_pixel_centers =
        GetAttrValue<bool>("half_pixel_centers");
    TRT_ENSURE_OK(half_pixel_centers);
    attrs_.half_pixel_centers = *half_pixel_centers;

    if (node_def.op() == "ResizeBilinear") {
      attrs_.mode = nvinfer1::ResizeMode::kLINEAR;
    } else if (node_def.op() == "ResizeNearestNeighbor") {
      attrs_.mode = nvinfer1::ResizeMode::kNEAREST;
    } else {
      return errors::Unimplemented(node_def.op(), " is not yet implemented at ",
                                   node_def.name());
    }

    // If we require workaround and have dynamic spatial shape, this is not
    // supported.
    if (UseConvResizeWorkaround(attrs_)) {
      if (!attrs_.align_corners && attrs_.half_pixel_centers) {
        return errors::Unimplemented(
            "Resize parameter combination of (align_corners=False, "
            "half_pixel_centers=true) is unsupported.");
      }
      if (!attrs_.align_corners &&
          !(attrs_.has_static_input_shape && attrs_.has_static_output_shape) &&
          attrs_.mode == nvinfer1::ResizeMode::kLINEAR) {
        return errors::Unimplemented(
            "Resize parameter combination of (mode=bilinear, align "
            "corners=False, dynamic spatial shape) is unsupported.");
      }
    }
    return Status::OK();
  }

  static bool UseConvResizeWorkaround(const ResizeAttributes& attrs) {
    if (!attrs.align_corners && attrs.mode == nvinfer1::ResizeMode::kLINEAR) {
      if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
        return true;
      }

      int64_t force_bilinear_resize_war = 0;
      ReadInt64FromEnvVar("TF_TRT_FORCE_BILINEAR_RESIZE_WAR", 0,
                          &force_bilinear_resize_war);
      if (force_bilinear_resize_war > 0) {
        LOG(WARNING) << "Forcing the use of LHS-dilated convolution WAR for "
                        "ResizeBilinear when aligned_corners=false";
        return true;
      }
    }
    return false;
  }

  Status Convert() {
    // Transpose tensor from NHWC to NCHW format. The TF BilinearResize
    // operation only supports the NHWC format.
    TF_RETURN_IF_ERROR(params_->converter->TransposeTensor(
        input_tensor_, {0, 3, 1, 2}, &input_tensor_, params_->node_def,
        "to_NCHW"));

    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params_->converter->network(), params_->weight_store);
    TRT_ENSURE_OK(builder);

    // Calculate the output shape as static dimensions or a shape tensor:
    // Given input shape [N, C, H, W] and output size [H_out, W_out],
    // output shape equals [N, C, H_out, W_out].
    TRT_TensorOrWeights size_input = params_->inputs.at(1);

    ITensorProxyPtr output;
    // For TensorRT < 8, we require a workaround for (bilinear,
    // align_corners=false). Otherwise, create a single TRT IResizeLayer.
    if (UseConvResizeWorkaround(attrs_)) {
      StatusOr<nvinfer1::ITensor*> out =
          ResizeUsingConvAndDilation(input_tensor_->trt_tensor());
      TRT_ENSURE_PTR_OK(out);
      output = *out;
    } else {
      // Compute the output shape.
      StatusOr<std::pair<nvinfer1::ITensor*, nvinfer1::Dims>> output_shape =
          ComputeOutputShape(input_tensor_, params_->inputs.at(1), &*builder,
                             attrs_.has_static_input_shape,
                             attrs_.has_static_output_shape);
      TRT_ENSURE_OK(output_shape);

      // Add resize layer.
      nvinfer1::IResizeLayer* layer = params_->converter->network()->addResize(
          *input_tensor_->trt_tensor());
      TRT_ENSURE(layer);
      params_->converter->SetLayerName(layer, params_->node_def);

      // Set layer parameters.
      layer->setResizeMode(attrs_.mode);
      layer->setAlignCorners(attrs_.align_corners);
      TF_RETURN_IF_ERROR(SetCoordinateTransform(layer));

      if (output_shape->first == nullptr) {
        // Use the static output dimensions.
        layer->setOutputDimensions(output_shape->second);
      } else {
        // Use the dynamic output dimensions.
        layer->setInput(1, *(output_shape->first));
      }

      output = layer->getOutput(0);
    }

    // Get output tensor. Transpose it from NCHW to NHWC.
    TF_RETURN_IF_ERROR(params_->converter->TransposeTensor(
        output, {0, 2, 3, 1}, &output, params_->node_def, "to_NHWC"));
    this->AddOutput(output);
    return Status::OK();
  }

  // For TensorRT >= 8, all possible attribute combinations are directly
  // supported.
  Status SetCoordinateTransform(nvinfer1::IResizeLayer* layer) const {
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    nvinfer1::ResizeCoordinateTransformation transform;
    if (!attrs_.half_pixel_centers && !attrs_.align_corners) {
      transform = nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC;
    } else if (attrs_.half_pixel_centers && !attrs_.align_corners) {
      transform = nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL;
    } else if (attrs_.align_corners && !attrs_.half_pixel_centers) {
      transform = nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS;
    } else {
      return errors::InvalidArgument(
          "invalid combination of half_pixel_centers and align_corners");
    }
    layer->setCoordinateTransformation(transform);
#endif
    return Status::OK();
  }

  // Workaround for align_corners=false and half_pixel_centers=false. Returns
  // output of final convolution layer. This should only be used for TRT <= 8.0.
  StatusOr<nvinfer1::ITensor*> ResizeUsingConvAndDilation(
      nvinfer1::ITensor* input) {
    // Currently we limit this to bilinear resize and parameters not covered by
    // TRT directly.
    TRT_ENSURE(!attrs_.align_corners && !attrs_.half_pixel_centers);
    TRT_ENSURE(attrs_.mode == nvinfer1::ResizeMode::kLINEAR);

    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params_->converter->network(), params_->weight_store);
    TRT_ENSURE_OK(builder);

    std::array<int64_t, 2> in_size = attrs_.static_input_size;
    std::array<int64_t, 2>& out_size = attrs_.static_output_size;
    for (int dim = 0; dim < attrs_.static_input_size.size(); ++dim) {
      if (in_size[dim] != out_size[dim]) {
        StatusOr<nvinfer1::ITensor*> result =
            this->ResizeUsingConvAndDilationOneDim(&*builder, input, dim,
                                                   &in_size, out_size);
        TRT_ENSURE_PTR_OK(result);
        input = *result;
      }
    }
    return input;
  }

  // Executes a sequence of one or more 1D resize operations to expand the given
  // input tensor from in_size[dim] to out_size[dim].
  StatusOr<nvinfer1::ITensor*> ResizeUsingConvAndDilationOneDim(
      TRTNetworkBuilder* builder, nvinfer1::ITensor* input_tensor,
      const int64_t dim, std::array<int64_t, 2>* in_size,
      std::array<int64_t, 2> out_size) {
    TRT_ENSURE(in_size != nullptr);
    TRT_ENSURE(builder != nullptr);
    TRT_ENSURE(input_tensor != nullptr);
    std::vector<int64_t> distances(out_size[dim] + 1, 0);
    std::vector<int64_t> steps(out_size[dim] + 1, 0);

    // A simple dynamic programming strategy builds the optimal sequence given
    // the cost model implemented in GetResizeCost.
    for (int i = static_cast<int>(distances.size()) - 2; i >= in_size->at(dim);
         --i) {
      distances[i] = std::numeric_limits<int64_t>::max();
      for (int j = i + 1; j < distances.size(); ++j) {
        StatusOr<int64_t> cost = GetResizeCost(i, j, attrs_.align_corners);
        TRT_ENSURE_OK(cost);
        // TF2XLA computes cost additively, but instead we say the cost of a
        // resize path (i->j->k) is the maximum cost of the individual costs of
        // resize kernels i->j and j->k. The reason is that the memory usage of
        // these kernels can be high. Going directly from i->k in a single
        // kernel may have lower overall cost but exceed GPU memory budget.
        int64_t distance = std::max(*cost, distances[j]);
        if (distance < distances[i]) {
          distances[i] = distance;
          steps[i] = j;
        }
      }
    }

    if (steps[in_size->at(dim)] == 0) {
      steps[in_size->at(dim)] = out_size[dim];
    }

    if (VLOG_IS_ON(1)) {
      std::string debug_path = "Resize Calculated Path: ";
      for (int64_t curr_step = in_size->at(dim); curr_step < out_size[dim];
           curr_step = steps[curr_step]) {
        absl::StrAppend(&debug_path, " (", curr_step, " -> ", steps[curr_step],
                        ") ");
      }
      VLOG(1) << debug_path;
    }

    while (in_size->at(dim) < out_size[dim]) {
      std::array<int64_t, 2> next_size = *in_size;
      next_size[dim] = steps[in_size->at(dim)];
      StatusOr<nvinfer1::ITensor*> result =
          ResizeUsingDilationAndConvolution<2>(builder, input_tensor, next_size,
                                               false,
                                               params_->use_implicit_batch);
      TRT_ENSURE_PTR_OK(result);
      input_tensor = *result;
      in_size->at(dim) = next_size[dim];
    }
    return input_tensor;
  }

 private:
  ITensorProxyPtr input_tensor_;
  ResizeAttributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertResize>(),
                                  {"ResizeBilinear", "ResizeNearestNeighbor"});
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
