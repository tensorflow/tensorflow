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
                                    const OHWDI& shape) {
  if (weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weight_desc.layout == WeightsLayout::kPowerVRConvF16OSpatialIOGroupI4O4 ||
      weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      weight_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4 * weight_desc.output_group_size);
    return i_aligned * o_aligned * shape.h * shape.w * shape.d;
  } else if (weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
             weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4);
    return i_aligned * o_aligned * weight_desc.spatial_remap.size();
  } else {
    return -1;
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
  } else if (dst_weight_desc.layout ==
             WeightsLayout::kPowerVRConvF16OSpatialIOGroupI4O4) {
    assert(dst_weight_desc.type == DataType::FLOAT16);
    half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
    RearrangeWeightsToImgConvF16OHWIOGroupI4O4(
        weights, dst_weight_desc.output_group_size,
        absl::MakeSpan(f16_ptr, flt_count / 4));
  }
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWDI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst) {
  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
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

void RearrangeWeightsToImgConvF16OHWIOGroupI4O4(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    int out_group_size, absl::Span<half4> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d_group = 0; d_group < out_group_size; ++d_group) {
            for (int j = 0; j < 4; ++j) {
              half4 filter;
              for (int i = 0; i < 4; ++i) {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * out_group_size + d_group) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filter[i] = weights.data[f_index];
                } else {
                  filter[i] = 0.0f;
                }
              }
              dst[counter++] = filter;
            }
            // [row, column] = [4, 2x2] (16 half)
            // B0_f16.xy, B0_f16.zw,
            // B1_f16.xy, B1_f16.zw
            // B2_f16.xy, B2_f16.zw,
            // B3_f16.xy, B3_f16.zw
            //
            // Transpose => [2x2, 4]
            // B0_f16.xy, B1_f16.xy,
            // B2_f16.xy, B3_f16.xy,
            // B0_f16.zw, B1_f16.zw,
            // B2_f16.zw, B3_f16.zw
            half2 b0zw(dst[counter - 4].z, dst[counter - 4].w);
            half2 b1xy(dst[counter - 3].x, dst[counter - 3].y);
            half2 b1zw(dst[counter - 3].z, dst[counter - 3].w);
            half2 b2xy(dst[counter - 2].x, dst[counter - 2].y);
            half2 b2zw(dst[counter - 2].z, dst[counter - 2].w);
            half2 b3xy(dst[counter - 1].x, dst[counter - 1].y);
            dst[counter - 4].z = b1xy[0];
            dst[counter - 4].w = b1xy[1];
            dst[counter - 3].x = b2xy[0];
            dst[counter - 3].y = b2xy[1];
            dst[counter - 3].z = b3xy[0];
            dst[counter - 3].w = b3xy[1];
            dst[counter - 2].x = b0zw[0];
            dst[counter - 2].y = b0zw[1];
            dst[counter - 2].z = b1zw[0];
            dst[counter - 2].w = b1zw[1];
            dst[counter - 1].x = b2zw[0];
            dst[counter - 1].y = b2zw[1];
          }
        }
      }
    }
  }
}

}  // namespace gpu
}  // namespace tflite
