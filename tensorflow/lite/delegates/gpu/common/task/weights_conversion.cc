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

#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"

namespace tflite {
namespace gpu {
uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWI& shape) {
  if (weight_desc.layout == WeightsLayout::kOHWIOGroupI4O4 ||
      weight_desc.layout == WeightsLayout::kOHWIOGroupO4I4 ||
      weight_desc.layout == WeightsLayout::k2DX4I4YIsHWIAndXIsOOGroupO4 ||
      weight_desc.layout == WeightsLayout::k2DX4O4YIsHWIAndXIsOOGroupI4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4 * weight_desc.output_group_size);
    return i_aligned * o_aligned * shape.h * shape.w;
  } else if (weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
             weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4);
    return i_aligned * o_aligned * weight_desc.spatial_remap.size();
  } else {
    return -1;
  }
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, DataType dst_type,
    absl::Span<uint8_t> dst) {
  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  if (dst_weight_desc.layout == WeightsLayout::kOHWIOGroupI4O4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOHWIOGroupO4I4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4I4YIsHWIAndXIsOOGroupO4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4O4YIsHWIAndXIsOOGroupI4) {
    if (dst_type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  }
}

}  // namespace gpu
}  // namespace tflite
