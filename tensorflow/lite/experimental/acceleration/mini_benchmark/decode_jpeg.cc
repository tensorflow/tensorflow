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
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
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
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  // TODO(b/172544567): Add error handling for incorrect/missing attributes.
  OpData* op_data = new OpData();
  op_data->height = m["height"].AsInt32();
  op_data->width = m["width"].AsInt32();
  op_data->num_images = m["num_images"].AsInt32();
  return op_data;
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
  // TODO(b/172544567): Support grayscale images.
  new_dims->data[3] = 3;  // Channels.
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

  const int kImageSize = op_data->width * op_data->height * 3;
  int output_array_offset = 0;
  for (int img = 0; img < op_data->num_images; ++img) {
    tflite::StringRef inputref =
        tflite::GetString(input_buffer, /*string_index=*/img);

    Status decode_status = decoder->DecodeImage(
        inputref, {op_data->height, op_data->width, /*channels=*/3},
        output_arr + output_array_offset, kImageSize);

    output_array_offset += kImageSize;

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
