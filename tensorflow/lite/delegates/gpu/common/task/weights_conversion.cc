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

#include <cassert>
#include <cstdint>
#include <limits>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWDI& shape) {
  if (weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      weight_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    uint i_aligned = static_cast<uint>(AlignByN(shape.i, 4));
    uint o_aligned =
        static_cast<uint>(AlignByN(shape.o, 4 * weight_desc.output_group_size));
    uint64_t total = static_cast<uint64_t>(i_aligned) *
                     static_cast<uint64_t>(o_aligned) *
                     static_cast<uint64_t>(static_cast<uint>(shape.h)) *
                     static_cast<uint64_t>(static_cast<uint>(shape.w)) *
                     static_cast<uint64_t>(static_cast<uint>(shape.d));
    if (total > std::numeric_limits<uint>::max()) {
      return 0;
    }
    return static_cast<uint>(total);
  } else if (weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
             weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    uint i_aligned = static_cast<uint>(AlignByN(shape.i, 4));
    uint o_aligned = static_cast<uint>(AlignByN(shape.o, 4));
    uint64_t total = static_cast<uint64_t>(i_aligned) *
                     static_cast<uint64_t>(o_aligned) *
                     weight_desc.spatial_remap.size();
    if (total > std::numeric_limits<uint>::max()) {
      return 0;
    }
    return static_cast<uint>(total);
  } else {
    return static_cast<uint>(-1);
  }
}

uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWI& shape) {
  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return GetTotalElementsCountForLayout(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWI& shape) {
  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return Get2dResourceSize(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWDI& shape) {
  const int dst_depth =
      AlignByN(DivideRoundUp(shape.o, 4), weight_desc.output_group_size);
  const int src_depth = DivideRoundUp(shape.i, 4);

  return uint2(dst_depth, src_depth * shape.h * shape.w * shape.d);
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst) {
  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  if (flt_count == 0) return;
  if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  }
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWDI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst) {
  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  if (flt_count == 0) return;
  if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToODHWIOGroupI4O4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToODHWIOGroupI4O4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToODHWIOGroupO4I4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToODHWIOGroupO4I4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToI4DHWIOOGroupO4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToI4DHWIOOGroupO4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToO4DHWIOOGroupI4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToO4DHWIOOGroupI4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  }
}

}  // namespace gpu
}  // namespace tflite
