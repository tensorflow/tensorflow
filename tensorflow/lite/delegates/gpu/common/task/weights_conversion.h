/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_CONVERSION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_CONVERSION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
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
              T filter;
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
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupO4I4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
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
              T filter;
              for (int i = 0; i < 4; ++i) {
                const int s_ch = s * 4 + i;
                const int d_ch = (d * out_group_size + d_group) * 4 + j;
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
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToODHWIOGroupI4O4(
    const tflite::gpu::Tensor<OHWDI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d) {
    for (int z = 0; z < weights.shape.d; ++z) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          for (int s = 0; s < src_slices; ++s) {
            for (int d_group = 0; d_group < out_group_size; ++d_group) {
              for (int j = 0; j < 4; ++j) {
                T filter;
                for (int i = 0; i < 4; ++i) {
                  const int s_ch = s * 4 + j;
                  const int d_ch = (d * out_group_size + d_group) * 4 + i;
                  if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                    const int f_index =
                        weights.shape.LinearIndex({d_ch, y, x, z, s_ch});
                    filter[i] = weights.data[f_index];
                  } else {
                    filter[i] = 0.0f;
                  }
                }
                dst[counter++] = filter;
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToI4HWIOOGroupO4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int j = 0; j < 4; ++j) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d = 0; d < dst_groups; ++d) {
            for (int d_group = 0; d_group < out_group_size; ++d_group) {
              T filter;
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
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToO4HWIOOGroupI4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int j = 0; j < 4; ++j) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int s = 0; s < src_slices; ++s) {
          for (int d = 0; d < dst_groups; ++d) {
            for (int d_group = 0; d_group < out_group_size; ++d_group) {
              T filter;
              for (int i = 0; i < 4; ++i) {
                const int s_ch = s * 4 + i;
                const int d_ch = (d * out_group_size + d_group) * 4 + j;
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
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToI4DHWIOOGroupO4(
    const tflite::gpu::Tensor<OHWDI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int j = 0; j < 4; ++j) {
    for (int z = 0; z < weights.shape.d; ++z) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          for (int s = 0; s < src_slices; ++s) {
            for (int d = 0; d < dst_groups; ++d) {
              for (int d_group = 0; d_group < out_group_size; ++d_group) {
                T filter;
                for (int i = 0; i < 4; ++i) {
                  const int s_ch = s * 4 + j;
                  const int d_ch = (d * out_group_size + d_group) * 4 + i;
                  if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                    const int f_index =
                        weights.shape.LinearIndex({d_ch, y, x, z, s_ch});
                    filter[i] = weights.data[f_index];
                  } else {
                    filter[i] = 0.0f;
                  }
                }
                dst[counter++] = filter;
              }
            }
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialI4O4(
    const tflite::gpu::Tensor<OHWI, S>& weights,
    const std::vector<int>& spatial_remap, absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          const int kernel_index = spatial_remap[y * weights.shape.w + x];
          const int kernel_index_x = kernel_index % weights.shape.w;
          const int kernel_index_y = kernel_index / weights.shape.w;
          for (int i = 0; i < 4; ++i) {
            T filter;
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                filter[j] = weights.data[f_index];
              } else {
                filter[j] = 0.0f;
              }
            }
            dst[counter++] = filter;
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsToOICustomSpatialO4I4(
    const tflite::gpu::Tensor<OHWI, S>& weights,
    const std::vector<int>& spatial_remap, absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int y = 0; y < weights.shape.h; ++y) {
        for (int x = 0; x < weights.shape.w; ++x) {
          const int kernel_index = spatial_remap[y * weights.shape.w + x];
          const int kernel_index_x = kernel_index % weights.shape.w;
          const int kernel_index_y = kernel_index / weights.shape.w;
          for (int i = 0; i < 4; ++i) {
            T filter;
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                filter[j] = weights.data[f_index];
              } else {
                filter[j] = 0.0f;
              }
            }
            dst[counter++] = filter;
          }
        }
      }
    }
  }
}

uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWI& shape);

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, DataType dst_type,
    absl::Span<uint8_t> dst);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_CONVERSION_H_
