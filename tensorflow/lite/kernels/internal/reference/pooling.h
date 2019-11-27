/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  TFLITE_DCHECK(input_shape.DimensionsCount() == 4 ||
                input_shape.DimensionsCount() == 5);
  TFLITE_DCHECK(output_shape.DimensionsCount() == 4 ||
                output_shape.DimensionsCount() == 5);
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(5, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  const int batches = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int channels = MatchingDim(ext_input_shape, 4, ext_output_shape, 4);
  const int input_depth = ext_input_shape.Dims(1);
  const int input_height = ext_input_shape.Dims(2);
  const int input_width = ext_input_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_z = 0; out_z < output_depth; ++out_z) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < channels; ++channel) {
            const int in_x_origin =
                (out_x * stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * stride_height) - params.padding_values.height;
            const int in_z_origin =
                (out_z * stride_depth) - params.padding_values.depth;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            const int filter_z_start = std::max(0, -in_z_origin);
            const int filter_z_end =
                std::min(params.filter_depth, input_depth - in_z_origin);
            float total = 0.f;
            float filter_count = 0;
            for (int filter_z = filter_z_start; filter_z < filter_z_end;
                ++filter_z) {
              for (int filter_y = filter_y_start; filter_y < filter_y_end;
                  ++filter_y) {
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                    ++filter_x) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  const int in_z = in_z_origin + filter_z;
                  total +=
                      input_data[Offset(ext_input_shape, batch, in_z, in_y, in_x, channel)];
                  filter_count++;
                }
              }
            }
            const float average = total / filter_count;
            output_data[Offset(ext_output_shape, batch, out_z, out_y, out_x, channel)] =
                ActivationFunctionWithMinMax(average, params.float_activation_min,
                                             params.float_activation_max);
          }
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8* input_data,
                        const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK(input_shape.DimensionsCount() == 4 ||
                input_shape.DimensionsCount() == 5);
  TFLITE_DCHECK(output_shape.DimensionsCount() == 4 ||
                output_shape.DimensionsCount() == 5);
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(5, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  const int batches = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int channels = MatchingDim(ext_input_shape, 4, ext_output_shape, 4);
  const int input_depth = ext_input_shape.Dims(1);
  const int input_height = ext_input_shape.Dims(2);
  const int input_width = ext_input_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_z = 0; out_z < output_depth; ++out_z) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < channels; ++channel) {
            const int in_x_origin =
                (out_x * stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * stride_height) - params.padding_values.height;
            const int in_z_origin =
                (out_z * stride_depth) - params.padding_values.depth;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            const int filter_z_start = std::max(0, -in_z_origin);
            const int filter_z_end =
                std::min(params.filter_depth, input_depth - in_z_origin);
            int32 acc = 0;
            int filter_count = 0;
            for (int filter_z = filter_z_start; filter_z < filter_z_end;
                ++filter_z) {
              for (int filter_y = filter_y_start; filter_y < filter_y_end;
                  ++filter_y) {
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                    ++filter_x) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  const int in_z = in_z_origin + filter_z;
                  acc +=
                      input_data[Offset(ext_input_shape, batch, in_z, in_y, in_x, channel)];
                  filter_count++;
                }
              }
            }
            acc = (acc + filter_count / 2) / filter_count;
            acc = std::max(acc, params.quantized_activation_min);
            acc = std::min(acc, params.quantized_activation_max);
            output_data[Offset(ext_output_shape, batch, out_z, out_y, out_x, channel)] =
                static_cast<uint8>(acc);
          }
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  TFLITE_DCHECK(input_shape.DimensionsCount() == 4 ||
                input_shape.DimensionsCount() == 5);
  TFLITE_DCHECK(output_shape.DimensionsCount() == 4 ||
                output_shape.DimensionsCount() == 5);
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(5, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  const int batches = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int channels = MatchingDim(ext_input_shape, 4, ext_output_shape, 4);
  const int input_depth = ext_input_shape.Dims(1);
  const int input_height = ext_input_shape.Dims(2);
  const int input_width = ext_input_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_z = 0; out_z < output_depth; ++out_z) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < channels; ++channel) {
            const int in_x_origin =
                (out_x * stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * stride_height) - params.padding_values.height;
            const int in_z_origin =
                (out_z * stride_depth) - params.padding_values.depth;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            const int filter_z_start = std::max(0, -in_z_origin);
            const int filter_z_end =
                std::min(params.filter_depth, input_depth - in_z_origin);
            float sum_squares = 0.f;
            int filter_count = 0;
            for (int filter_z = filter_z_start; filter_z < filter_z_end;
                ++filter_z) {
              for (int filter_y = filter_y_start; filter_y < filter_y_end;
                  ++filter_y) {
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                    ++filter_x) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  const int in_z = in_z_origin + filter_z;
                  const float val =
                      input_data[Offset(ext_input_shape, batch, in_z, in_y, in_x, channel)];
                  sum_squares += val * val;
                  filter_count++;
                }
              }
            }
            const float l2pool_result = std::sqrt(sum_squares / filter_count);
            output_data[Offset(ext_output_shape, batch, out_z, out_y, out_x, channel)] =
                ActivationFunctionWithMinMax(l2pool_result,
                                            params.float_activation_min,
                                            params.float_activation_max);
          }
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  TFLITE_DCHECK(input_shape.DimensionsCount() == 4 ||
                input_shape.DimensionsCount() == 5);
  TFLITE_DCHECK(output_shape.DimensionsCount() == 4 ||
                output_shape.DimensionsCount() == 5);
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(5, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  const int batches = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int channels = MatchingDim(ext_input_shape, 4, ext_output_shape, 4);
  const int input_depth = ext_input_shape.Dims(1);
  const int input_height = ext_input_shape.Dims(2);
  const int input_width = ext_input_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_z = 0; out_z < output_depth; ++out_z) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < channels; ++channel) {
            const int in_x_origin =
                (out_x * stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * stride_height) - params.padding_values.height;
            const int in_z_origin =
                (out_z * stride_depth) - params.padding_values.depth;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            const int filter_z_start = std::max(0, -in_z_origin);
            const int filter_z_end =
                std::min(params.filter_depth, input_depth - in_z_origin);
            float max = std::numeric_limits<float>::lowest();
            for (int filter_z = filter_z_start; filter_z < filter_z_end;
                ++filter_z) {
              for (int filter_y = filter_y_start; filter_y < filter_y_end;
                  ++filter_y) {
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                    ++filter_x) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  const int in_z = in_z_origin + filter_z;
                  max = std::max(
                      max,
                      input_data[Offset(ext_input_shape, batch, in_z, in_y, in_x, channel)]);
                }
              }
            }
            output_data[Offset(ext_output_shape, batch, out_z, out_y, out_x, channel)] =
                ActivationFunctionWithMinMax(max, params.float_activation_min,
                                            params.float_activation_max);
          }
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8* input_data, const RuntimeShape& output_shape,
                    uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, 0);
  TFLITE_DCHECK_LE(params.quantized_activation_max, 255);
  TFLITE_DCHECK(input_shape.DimensionsCount() == 4 ||
                input_shape.DimensionsCount() == 5);
  TFLITE_DCHECK(output_shape.DimensionsCount() == 4 ||
                output_shape.DimensionsCount() == 5);
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(5, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  const int batches = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int channels = MatchingDim(ext_input_shape, 4, ext_output_shape, 4);
  const int input_depth = ext_input_shape.Dims(1);
  const int input_height = ext_input_shape.Dims(2);
  const int input_width = ext_input_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_z = 0; out_z < output_depth; ++out_z) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int channel = 0; channel < channels; ++channel) {
            const int in_x_origin =
                (out_x * stride_width) - params.padding_values.width;
            const int in_y_origin =
                (out_y * stride_height) - params.padding_values.height;
            const int in_z_origin =
                (out_z * stride_depth) - params.padding_values.depth;
            // Compute the boundaries of the filter region clamped so as to
            // ensure that the filter window fits in the input array.
            const int filter_x_start = std::max(0, -in_x_origin);
            const int filter_x_end =
                std::min(params.filter_width, input_width - in_x_origin);
            const int filter_y_start = std::max(0, -in_y_origin);
            const int filter_y_end =
                std::min(params.filter_height, input_height - in_y_origin);
            const int filter_z_start = std::max(0, -in_z_origin);
            const int filter_z_end =
                std::min(params.filter_depth, input_depth - in_z_origin);
            uint8 max = 0;
            for (int filter_z = filter_z_start; filter_z < filter_z_end;
                ++filter_z) {
              for (int filter_y = filter_y_start; filter_y < filter_y_end;
                  ++filter_y) {
                for (int filter_x = filter_x_start; filter_x < filter_x_end;
                    ++filter_x) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  const int in_z = in_z_origin + filter_z;
                  max = std::max(
                      max,
                      input_data[Offset(ext_input_shape, batch, in_z, in_y, in_x, channel)]);
                }
              }
            }
            max = std::max<uint8>(max, params.quantized_activation_min);
            max = std::min<uint8>(max, params.quantized_activation_max);
            output_data[Offset(ext_output_shape, batch, out_z, out_y, out_x, channel)] =
                static_cast<uint8>(max);
          }
        }
      }
    }
  }
}
}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
