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

#include "tensorflow/compiler/tf2tensorrt/convert/ops/slice_ops.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <bitset>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

Status ConvertStridedSliceHelper(
    OpConverterParams* params, const TRT_TensorOrWeights& input,
    const PartialTensorShape& input_dims, const SliceDims& begin,
    const SliceDims& stride, const SliceDims& end,
    absl::optional<nvinfer1::Dims> final_shape, absl::optional<int> op_instance,
    absl::optional<StridedSliceShapeSpec> strided_slice_spec) {
  const auto& node_def = params->node_def;

  nvinfer1::Dims begin_dims, stride_dims, end_dims;
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(begin, &begin_dims, params->use_implicit_batch));
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(stride, &stride_dims, params->use_implicit_batch));
  TF_RETURN_IF_ERROR(
      ContainerToTrtDims(end, &end_dims, params->use_implicit_batch));

  // Try to calculate size dimensions. In the implict batch case, not having
  // static slice size is an error.
  nvinfer1::Dims size_dims = begin_dims;
  absl::InlinedVector<int64, 4> static_input_size_indices;
  absl::InlinedVector<int64, 4> dynamic_input_size_indices;
  for (int i = 0; i < begin_dims.nbDims; i++) {
    size_dims.d[i] = (std::abs(end_dims.d[i] - begin_dims.d[i]) +
                      std::abs(stride_dims.d[i]) - 1) /
                     std::abs(stride_dims.d[i]);

    if (input_dims.dim_size(i) < 0) {
      // In this case, ValidateStridedSliceOp did not appropriate set the begin
      // and end dims. We need to calculate based on the masks below.
      dynamic_input_size_indices.push_back(i);
    } else {
      static_input_size_indices.push_back(i);
      if (end_dims.d[i] < begin_dims.d[i] && stride_dims.d[i] > 0) {
        return errors::InvalidArgument(
            "\"size\" cannot be negative for StridedSlice");
      }
    }
  }

  if (params->validation_only) return Status::OK();

  NetworkFactoryContext ctx(params->converter->network(), params->weight_store);

  VLOG(2) << "strided slice helper:"
          << " begin:" << DebugString(begin_dims)
          << "\n stride: " << DebugString(stride_dims)
          << "\n end: " << DebugString(end_dims)
          << "\n size: " << DebugString(size_dims)
          << "\n Dynamic indices: " << DebugString(dynamic_input_size_indices)
          << "\n Static indices: " << DebugString(static_input_size_indices);

  // Create the slice operation. This is all that is required in the case of
  // static dims, but for dynamic dims, we override these paramters below by
  // assigning inputs to the returned layer.
  auto slice = ctx.Slice(input.tensor()->trt_tensor(), begin_dims, size_dims,
                         stride_dims);

  if (!dynamic_input_size_indices.empty()) {
    // Require strided slice spec when dynamic shape is present.
    TRT_EXPECT(strided_slice_spec != absl::nullopt);

    if (params->use_implicit_batch) {
      return errors::InvalidArgument(
          "In implicit batch mode, dynamic input size is not supported.");
    }

    // Check if any of our dynamic indices has begin/end mask.
    absl::InlinedVector<int64, 4> dynamic_begin_indices;
    absl::InlinedVector<int64, 4> dynamic_end_indices;
    const auto begin_mask =
        std::bitset<32>(strided_slice_spec->begin_dense_mask);
    const auto end_mask = std::bitset<32>(strided_slice_spec->end_dense_mask);
    for (int i = 0; i < dynamic_input_size_indices.size(); i++) {
      auto dynamic_idx = dynamic_input_size_indices[i];
      // When stride is negative:
      // - If "begin_mask[dynamic_idx]" is set, then we need to adjust the slice
      // start of dimension[i] to the dynamic size.
      // - If "end_mask[dynamic_idx]" is set, it suffices to set
      // end_dims[dynamic_idx] to -1.
      // When stride is positive:
      // - If "begin_mask[dynamic_idx]" is set, it suffices to set
      // begin_dims[dynamic_idx] to zero.
      // - If "end_mask[dynamic_idx]" is set, we need to adjust slice end to the
      // dynamic size of dimension "dynamic_idx".
      if (begin_mask[dynamic_idx]) {
        // DLOG("begin_dims->d[" << dynamic_idx << "] = 0");
        begin_dims.d[dynamic_idx] = 0;
        if (stride_dims.d[dynamic_idx] < 0) {
          dynamic_begin_indices.push_back(dynamic_idx);
        }
      }
      if (end_mask[dynamic_idx]) {
        end_dims.d[dynamic_idx] = stride_dims.d[dynamic_idx] > 0 ? 0 : -1;
        if (stride_dims.d[dynamic_idx] > 0) {
          dynamic_end_indices.push_back(dynamic_idx);
        }
      }
    }

    VLOG(2) << " Dynamic begin indices: " << DebugString(dynamic_begin_indices)
            << " Dynamic end indices: " << DebugString(dynamic_end_indices);

    // Create ITensors for each of the begin/stride/end constants.
    auto begin_const = ctx.Constant(begin_dims);
    TRT_ENSURE_OK(begin_const);
    nvinfer1::ITensor* begin_tensor = begin_const->output;
    auto stride_const = ctx.Constant(stride_dims);
    TRT_ENSURE_OK(stride_const);
    auto end_const = ctx.Constant(end_dims);
    TRT_ENSURE_OK(end_const);
    nvinfer1::ITensor* end_tensor = end_const->output;

    // Make corrections based on the begin_mask/end_mask values.
    if (dynamic_end_indices.size() > 0) {
      auto dynamic_end_masked_tensor = ctx.GetPartialShapeOf(
          input.tensor()->trt_tensor(), dynamic_end_indices);
      TRT_ENSURE_OK(dynamic_end_masked_tensor);
      auto end_corrected =
          ctx.Add(dynamic_end_masked_tensor->output, end_tensor);
      TRT_ENSURE_OK(end_corrected);
      end_tensor = end_corrected->output;
    }
    if (dynamic_begin_indices.size() > 0) {
      auto dynamic_begin_masked_tensor = ctx.GetPartialShapeOf(
          input.tensor()->trt_tensor(), dynamic_begin_indices);
      TRT_ENSURE_OK(dynamic_begin_masked_tensor);
      // We need to subtract one from the size in order to get a valid index.
      auto nonzero_mask = ctx.NonZeroInt(dynamic_begin_masked_tensor->output);
      TRT_ENSURE_OK(nonzero_mask);
      auto sub_one =
          ctx.Sub(dynamic_begin_masked_tensor->output, nonzero_mask->output);
      TRT_ENSURE_OK(sub_one)
      auto begin_corrected = ctx.Add(sub_one->output, begin_tensor);
      TRT_ENSURE_OK(begin_corrected);
      begin_tensor = begin_corrected->output;
    }

    // Calculate the final size of the slice dynamicly.
    nvinfer1::ITensor* size_tensor;
    {
      auto num = ctx.Sub(end_tensor, begin_tensor);
      TRT_ENSURE_OK(num);
      auto ceil_div = ctx.AbsCeilDivInt(num->output, stride_const->output);
      TRT_ENSURE_OK(ceil_div);
      size_tensor = ceil_div->output;
    }

    TRT_ENSURE_OK(slice);
    nvinfer1::ITensor* trt_final_size;
    slice->layer->setInput(1, *begin_tensor);
    slice->layer->setInput(2, *size_tensor);
    slice->layer->setInput(3, *stride_const->output);
  }

  params->converter->SetLayerName(slice->layer, params->node_def, "slice",
                                  op_instance);
  ITensorProxyPtr tensor = slice->output;

  // Reshape for shrink_axis.
  if (final_shape) {
    TF_RETURN_IF_ERROR(PrepareTensorForShape(
        params->converter, TRT_TensorOrWeights(tensor), *final_shape,
        /*validation_only=*/false, &tensor, node_def, op_instance));
  }
  params->outputs->push_back(TRT_TensorOrWeights(tensor));
  return Status::OK();
}
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT