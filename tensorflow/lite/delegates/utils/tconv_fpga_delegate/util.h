#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_TCONV_FPGA_DELEGATE_TCONV_FPGA_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_TCONV_FPGA_DELEGATE_TCONV_FPGA_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

// This file mostly contains generic tflite code to prepare CONV2D tensors /
// IM2COL

constexpr int kOutputShapeTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kDataInputTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kOutputTensor = 0;
const int kTensorNotAllocated = -1;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int col2im_id = kTensorNotAllocated;
  int transposed_weights_id = kTensorNotAllocated;
  int scratch_tensor_id = kTensorNotAllocated;

  // col2im is the temporary tensor allocated and used in optimized path for
  // storing col2im data:gemm result for input_matrix x filter_matrix.
  int32_t col2im_index;

  // TfLiteConverter will transpose weights from HWOI to OHWI order.
  // In optimized path, we will transpose them back to HWOI, this temporary
  // tensor is allocated for storing transposed weights.
  int32_t transposed_weights_index;

  // Scratch tensor is used in the quantized path for storing accumulation
  // results.
  int32_t scratch_tensor_index;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int32_t> per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  bool has_col2im = false;
  bool weights_are_transposed = false;
};

inline TfLiteTensor* GetTensorAtIndex(const TfLiteContext* context,
                                      int tensor_index) {
  return &context->tensors[tensor_index];
}

inline TfLiteStatus GetMutableInputSafe(const TfLiteContext* context,
                                        int tensor_index,
                                        const TfLiteTensor** tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

TfLiteStatus GetInputSafe(const TfLiteContext* context, int tensor_index,
                          const TfLiteTensor** tensor) {
  return GetMutableInputSafe(context, tensor_index, tensor);
}

TfLiteStatus GetOutputSafe(const TfLiteContext* context, int tensor_index,
                           TfLiteTensor** tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

namespace tflite {

// From im2col_utils.h
// template <typename T>
// inline void ExtractPatchIntoBufferColumn(const RuntimeShape& input_shape, int w,
//                                          int h, int b, int kheight, int kwidth,
//                                          int stride_width, int stride_height,
//                                          int pad_width, int pad_height,
//                                          int in_width, int in_height,
//                                          int in_depth, int single_buffer_length,
//                                          int buffer_id, const T* in_data,
//                                          T* conv_buffer_data, uint8 zero_byte) {
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   // This chunk of code reshapes all the inputs corresponding to
//   // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
//   const int kwidth_times_indepth = kwidth * in_depth;
//   const int inwidth_times_indepth = in_width * in_depth;
//   const int ih_ungated_start = h * stride_height - pad_height;
//   const int ih_ungated_end = (ih_ungated_start + kheight);
//   const int ih_end = std::min(ih_ungated_end, in_height);
//   const int iw_ungated_start = w * stride_width - pad_width;
//   const int iw_ungated_end = (iw_ungated_start + kwidth);
//   const int iw_end = std::min(iw_ungated_end, in_width);
//   // If the patch is off the edge of the input image, skip writing those rows
//   // and columns from the patch into the output array.
//   const int h_offset = std::max(0, -ih_ungated_start);
//   const int w_offset = std::max(0, -iw_ungated_start);
//   const int ih_start = std::max(0, ih_ungated_start);
//   const int iw_start = std::max(0, iw_ungated_start);
//   const int single_row_num =
//       std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
//   const int output_row_offset = (buffer_id * single_buffer_length);
//   int out_offset =
//       output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
//   int in_offset = Offset(input_shape, b, ih_start, iw_start, 0);

//   // Express all of the calculations as padding around the input patch.
//   const int top_padding = h_offset;
//   const int bottom_padding = (ih_ungated_end - ih_end);
//   const int left_padding = w_offset;
//   const int right_padding = (iw_ungated_end - iw_end);
//   assert(single_row_num ==
//          ((kwidth - (left_padding + right_padding)) * in_depth));

//   // Write out zeroes to the elements representing the top rows of the input
//   // patch that are off the edge of the input image.
//   if (top_padding > 0) {
//     const int top_row_elements = (top_padding * kwidth * in_depth);
//     memset(conv_buffer_data + output_row_offset, zero_byte,
//            (top_row_elements * sizeof(T)));
//   }

//   // If the patch is on the interior of the input image horizontally, just copy
//   // over the rows sequentially, otherwise add zero padding at the start or end.
//   if ((left_padding == 0) && (right_padding == 0)) {
//     for (int ih = ih_start; ih < ih_end; ++ih) {
//       memcpy(conv_buffer_data + out_offset, in_data + in_offset,
//              single_row_num * sizeof(T));
//       out_offset += kwidth_times_indepth;
//       in_offset += inwidth_times_indepth;
//     }
//   } else {
//     for (int ih = ih_start; ih < ih_end; ++ih) {
//       if (left_padding > 0) {
//         const int left_start = (out_offset - (left_padding * in_depth));
//         memset(conv_buffer_data + left_start, zero_byte,
//                (left_padding * in_depth * sizeof(T)));
//       }
//       memcpy(conv_buffer_data + out_offset, in_data + in_offset,
//              single_row_num * sizeof(T));
//       if (right_padding > 0) {
//         const int right_start = (out_offset + single_row_num);
//         memset(conv_buffer_data + right_start, zero_byte,
//                (right_padding * in_depth * sizeof(T)));
//       }
//       out_offset += kwidth_times_indepth;
//       in_offset += inwidth_times_indepth;
//     }
//   }

//   // If the bottom of the patch falls off the input image, pad the values
//   // representing those input rows with zeroes.
//   if (bottom_padding > 0) {
//     const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
//     const int bottom_start =
//         output_row_offset +
//         ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
//     memset(conv_buffer_data + bottom_start, zero_byte,
//            (bottom_row_elements * sizeof(T)));
//   }
// }

// template <typename T>
// void Im2col(ConvParams& params, int kheight, int kwidth, uint8 zero_byte,
//             const RuntimeShape& input_shape, const T* input_data,
//             const RuntimeShape& output_shape, T* output_data) {
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = input_shape.Dims(3);
//   const int input_width = input_shape.Dims(2);
//   const int input_height = input_shape.Dims(1);
//   const int output_depth = output_shape.Dims(3);
//   const int output_width = output_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);

//   int buffer_id = 0;
//   // Loop over the output nodes.
//   for (int b = 0; b < batches; ++b) {
//     for (int h = 0; h < output_height; ++h) {
//       for (int w = 0; w < output_width; ++w) {
//         ExtractPatchIntoBufferColumn(
//             input_shape, w, h, b, kheight, kwidth, stride_width, stride_height,
//             pad_width, pad_height, input_width, input_height, input_depth,
//             output_depth, buffer_id, input_data, output_data, zero_byte);
//         ++buffer_id;
//       }
//     }
//   }
// }

// // Supports per-batch zero_byte for per-batch asymmetric quantized inputs.
// template <typename T>
// void DilatedIm2col(ConvParams& params, const RuntimeShape& input_shape,
//                    const T* input_data, const RuntimeShape& filter_shape,
//                    const RuntimeShape& output_shape, T* im2col_data,
//                    const int32_t* zero_bytes, const int zero_bytes_len) {
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   // For dilated convolution, the input pixels are not contiguous therefore we
//   // can't use the same optimizations as Im2Col(). Though note this code would
//   // work fine for the non-dilated case too (though likely a bit slower).
//   TFLITE_DCHECK(dilation_width_factor != 1 || dilation_height_factor != 1);
//   TFLITE_DCHECK(im2col_data);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   MatchingDim(output_shape, 3, filter_shape, 0);

//   // Construct the MxN sized im2col matrix.
//   // The rows M, are sub-ordered B x H x W
//   const RuntimeShape row_shape({1, batches, output_height, output_width});
//   // The columns, N, are sub-ordered Kh x Kw x Din
//   const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
//   // Use dimensions M and N to construct dims for indexing directly into im2col
//   const RuntimeShape im2col_shape(
//       {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

//   // Loop through the output rows (B x H x W)
//   for (int batch = 0; batch < batches; ++batch) {
//     const T zero_byte = zero_bytes_len > 1 ? static_cast<T>(zero_bytes[batch])
//                                            : static_cast<T>(zero_bytes[0]);
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         // Each im2col row is an output pixel. Arrange the input data in this
//         // row in an order we can conveniently multiply with the filter data.
//         int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
//         const int in_x_origin = (out_x * stride_width) - pad_width;
//         const int in_y_origin = (out_y * stride_height) - pad_height;
//         // Loop through all the pixels of the filter (Kh x Kw)
//         for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//           const int in_y = in_y_origin + dilation_height_factor * filter_y;
//           if ((in_y >= 0) && (in_y < input_height)) {
//             // Filter row is within the input data.
//             // Loop through all the filter pixels in this row.
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;
//               int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
//               T* dst = im2col_data +
//                        Offset(im2col_shape, 0, 0, row_offset, col_offset);
//               if ((in_x >= 0) && (in_x < input_width)) {
//                 // Filter pixel is within the input, copy the input data.
//                 T const* src =
//                     input_data + Offset(input_shape, batch, in_y, in_x, 0);
//                 memcpy(dst, src, input_depth * sizeof(T));
//               } else {
//                 // Filter pixel is outside the input, zero it out.
//                 memset(dst, zero_byte, input_depth * sizeof(T));
//               }
//             }
//           } else {
//             // Filter row is outside the input, zero out the entire filter row.
//             int col_offset = Offset(col_shape, 0, filter_y, 0, 0);
//             T* dst = im2col_data +
//                      Offset(im2col_shape, 0, 0, row_offset, col_offset);
//             memset(dst, zero_byte, filter_width * input_depth * sizeof(T));
//           }
//         }
//       }
//     }
//   }
// }

// template <typename T>
// void DilatedIm2col(ConvParams& params, uint8 zero_byte,
//                    const RuntimeShape& input_shape, const T* input_data,
//                    const RuntimeShape& filter_shape,
//                    const RuntimeShape& output_shape, T* im2col_data) {
//   const int32_t zero_point = static_cast<int32_t>(zero_byte);
//   DilatedIm2col<T>(params, input_shape, input_data, filter_shape, output_shape,
//                    im2col_data, &zero_point, 1);
// }


// From im2col_utils.h End
TfLiteStatus ResizeAndTransposeWeights(TfLiteContext* context,
                                       const TfLiteTensor* weights,
                                       TfLiteTensor* transposed_weights) {
  TfLiteIntArray* transposed_weights_shape_array = TfLiteIntArrayCreate(4);
  const RuntimeShape& input_shape = GetTensorShape(weights);
  transposed_weights_shape_array->data[0] = input_shape.Dims(1);
  transposed_weights_shape_array->data[1] = input_shape.Dims(2);
  transposed_weights_shape_array->data[2] = input_shape.Dims(0);
  transposed_weights_shape_array->data[3] = input_shape.Dims(3);

  transposed_weights->type = weights->type;
  transposed_weights->allocation_type = kTfLiteDynamic;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, transposed_weights,
                                              transposed_weights_shape_array));

  // Transpose the weights from OHWI order to HWOI order.
  TransposeParams transpose_params;
  transpose_params.perm_count = 4;
  transpose_params.perm[0] = 1;
  transpose_params.perm[1] = 2;
  transpose_params.perm[2] = 0;
  transpose_params.perm[3] = 3;

  if (weights->type == kTfLiteFloat32) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<float>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<float>(transposed_weights));
  } else if (weights->type == kTfLiteUInt8) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<uint8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<uint8>(transposed_weights));
  } else if (weights->type == kTfLiteInt8) {
    // int16 transpose_conv also with int8 weights
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<int8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<int8>(transposed_weights));
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "Only float32, uint8, int8, int16 is supported currently, got %s.",
        TfLiteTypeGetName(weights->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}


}  // namespace tflite

bool IsIm2ColRequired(const TfLiteTensor* input, TfLiteConvParams* params,
                      const TfLiteTensor* filter, OpData* data,
                      bool is_hybrid) {
  // If HWCN weights are required, Im2Col not required
  // if (data->need_hwcn_weights) return false;

  // segregate based on dilated conv & non-dialated conv
  const bool need_dilated_im2col =
      params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_width != 1 || params->stride_height != 1 ||
      filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

  // Return early as basic requirement is not met
  if (!need_im2col) return false;

  // Special case for Hybrid, as it supports only non-dilated im2col currently
  const bool is_hybrid_non_dilated = is_hybrid && need_non_dilated_im2col;
  const bool is_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;

  if (is_hybrid && !need_non_dilated_im2col) {
    return false;
  } else {
    return true;
  }
}

TfLiteStatus ResizeTensor(TfLiteContext* context,
                          const TfLiteTensor* shape_tensor,
                          TfLiteTensor* tensor_to_resize) {
  // Currently only support int32 for output shape.
  if (shape_tensor->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Output shape is %s, not int32.",
                       TfLiteTypeGetName(shape_tensor->type));
    return kTfLiteError;
  }

  TfLiteIntArray* shape = TfLiteIntArrayCreate(tflite::NumElements(shape_tensor));
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = shape_tensor->data.i32[i];
  }

  return context->ResizeTensor(context, tensor_to_resize, shape);
}

TfLiteStatus ResizeCol2ImTensor(TfLiteContext* context,
                                const TfLiteTensor* output_shape,
                                const TfLiteTensor* weights,
                                const TfLiteTensor* input,
                                TfLiteTensor* col2im) {
  if (output_shape->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "col2im shape is %s, not int32.",
                       TfLiteTypeGetName(output_shape->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, tflite::NumElements(output_shape), 4);
  TfLiteIntArray* col2im_shape_array = TfLiteIntArrayCreate(2);
  const tflite::RuntimeShape& input_shape = tflite::GetTensorShape(input);
  const tflite::RuntimeShape& weights_shape = tflite::GetTensorShape(weights);
  col2im_shape_array->data[0] = input_shape.Dims(1) * input_shape.Dims(2);
  col2im_shape_array->data[1] =
      weights_shape.Dims(0) * weights_shape.Dims(1) * weights_shape.Dims(2);

  col2im->type = input->type == kTfLiteFloat32 ? kTfLiteFloat32 : kTfLiteInt32;
  col2im->allocation_type = kTfLiteDynamic;
  return context->ResizeTensor(context, col2im, col2im_shape_array);
}

static TfLiteStatus AllocateTemporaryTensorsIfRequired(
    TfLiteContext* context, TfLiteNode* node, TfLiteTransposeConvParams* params,
    OpData* data, bool req_temp_out, int temp_out_tid, int& temp_out_id) {
  int temporaries_count = node->temporaries->size;
  if (data->col2im_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->col2im_id);
  }
  data->col2im_index = temporaries_count;
  data->has_col2im = true;
  ++temporaries_count;

  if (data->transposed_weights_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->transposed_weights_id);
  }
  data->transposed_weights_index = temporaries_count;
  data->weights_are_transposed = true;
  ++temporaries_count;

  if (data->scratch_tensor_id == kTensorNotAllocated) {
    context->AddTensors(context, 1, &data->scratch_tensor_id);
  }
  data->scratch_tensor_index = temporaries_count;
  ++temporaries_count;

  if (req_temp_out) {
    temp_out_id = temporaries_count;
    if (temp_out_tid == kTensorNotAllocated) {
      context->AddTensors(context, 1, &temp_out_tid);
    }
    ++temporaries_count;
  }

  auto temp_array = TfLiteIntArrayCreate(temporaries_count);
  for (int i = 0; i < node->temporaries->size; i++)
    temp_array->data[i] = node->temporaries->data[i];

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = temp_array;

  return kTfLiteOk;
}

#endif