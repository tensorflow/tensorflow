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
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  if (!buffer) {
    return nullptr;
  }
#define RET_ENSURE(context, condition)                                  \
  do {                                                                  \
    if (!(condition)) {                                                 \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #condition);                         \
      return nullptr;                                                   \
    }                                                                   \
  } while (0)
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  RET_ENSURE(context, m["height"].IsInt());
  RET_ENSURE(context, m["width"].IsInt());
  RET_ENSURE(context, m["num_images"].IsInt());
  RET_ENSURE(context, m["channels"].IsInt());
  OpData* op_data = new OpData();
  op_data->height = m["height"].AsInt32();
  op_data->width = m["width"].AsInt32();
  op_data->num_images = m["num_images"].AsInt32();
  op_data->channels = m["channels"].AsInt32();
  return op_data;
#undef RET_ENSURE
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data);
  TF_LITE_ENSURE(context, op_data->height > 0);
  TF_LITE_ENSURE(context, op_data->width > 0);
  TF_LITE_ENSURE(context, op_data->num_images > 0);
  // TODO(b/172544567): Support grayscale images.
  TF_LITE_ENSURE(context, op_data->channels == 3 || op_data->channels == 4);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input_buffer;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, /*index=*/0, &input_buffer));

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output_tensor));

  TF_LITE_ENSURE_TYPES_EQ(context, input_buffer->type, kTfLiteString);
  TF_LITE_ENSURE_TYPES_EQ(context, output_tensor->type, kTfLiteUInt8);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input_buffer), 1);
  TF_LITE_ENSURE_EQ(context, input_buffer->dims->data[0], op_data->num_images);

  // Resize output.
  // Output shape is determined as {num_images, height, width, channels}.
  TfLiteIntArray* new_dims = TfLiteIntArrayCreate(4);
  new_dims->data[0] = op_data->num_images;
  new_dims->data[1] = op_data->height;
  new_dims->data[2] = op_data->width;
  new_dims->data[3] = op_data->channels;
  output_tensor->type = kTfLiteUInt8;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output_tensor, new_dims));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Decodes a batch of JPEG images.

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input_buffer;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, /*index=*/0, &input_buffer));
  TF_LITE_ENSURE(context, input_buffer);
  TF_LITE_ENSURE(context, input_buffer->data.raw);
  const int channels = op_data->channels;
  // TODO(b/172544567): Support grayscale images.
  const int decode_channels = 3;
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output_tensor));
  // kTfliteUInt8 corresponds to unsigned char as shown in
  // "tensorflow/lite/portable_type_to_tflitetype.h".
  unsigned char* output_arr = GetTensorData<unsigned char>(output_tensor);
  Status decoder_status;
  std::unique_ptr<LibjpegDecoder> decoder =
      LibjpegDecoder::Create(decoder_status);
  if (decoder_status.code != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, decoder_status.error_message.c_str());
    return kTfLiteError;
  }

  const int kDecodedImageSize =
      op_data->width * op_data->height * decode_channels;
  const int kOutputImageSize = op_data->width * op_data->height * channels;

  int output_array_offset = 0;
  for (int img = 0; img < op_data->num_images; ++img) {
    tflite::StringRef inputref =
        tflite::GetString(input_buffer, /*string_index=*/img);
    unsigned char* decoded = output_arr + output_array_offset;

    Status decode_status = decoder->DecodeImage(
        inputref, {op_data->height, op_data->width, decode_channels}, decoded,
        kDecodedImageSize);

    if (channels == 4) {
      // Reorganize the decoded buffer from 3 channels to 4 channels.
      size_t height = op_data->height;
      size_t src_offset = kDecodedImageSize;
      size_t dst_offset = kOutputImageSize;
      while (height--) {
        size_t width = op_data->width;
        while (width--) {
          src_offset -= decode_channels;
          dst_offset -= channels;
          std::copy_n(decoded + src_offset, decode_channels,
                      decoded + dst_offset);
          // Add an alpha channel value of 255 (fully opaque) to the
          // current pixel if the target output channels is provided as 4. This
          // is a workaround to allow jpeg decoder to work with 4 channel input
          // models.
          decoded[dst_offset + 3] = 255;
        }
      }
    }

    output_array_offset += kOutputImageSize;

    if (decode_status.code != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, decode_status.error_message.c_str());
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_DECODE_JPEG() {
  static TfLiteRegistration r = {
      decode_jpeg_kernel::Init, decode_jpeg_kernel::Free,
      decode_jpeg_kernel::Prepare, decode_jpeg_kernel::Eval};
  return &r;
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
