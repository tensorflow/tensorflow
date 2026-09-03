/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// Functions to read images in JPEG XL format.

#include "tensorflow/core/lib/jxl/jxl_io.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "lib/include/jxl/codestream_header.h"  // from @jpegxl
#include "lib/include/jxl/color_encoding.h"  // from @jpegxl
#include "lib/include/jxl/decode.h"  // from @jpegxl
#include "lib/include/jxl/decode_cxx.h"  // from @jpegxl
#include "lib/include/jxl/encode.h"  // from @jpegxl
#include "lib/include/jxl/encode_cxx.h"  // from @jpegxl
#include "lib/include/jxl/resizable_parallel_runner.h"  // from @jpegxl
#include "lib/include/jxl/resizable_parallel_runner_cxx.h"  // from @jpegxl
#include "lib/include/jxl/types.h"  // from @jpegxl
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace jxl {

bool HasJxlHeader(absl::string_view encoded) {
  JxlSignature signature = JxlSignatureCheck(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
  return signature == JXL_SIG_CODESTREAM || signature == JXL_SIG_CONTAINER;
}

bool DecodeHeader(absl::string_view encoded, int* width, int* height,
                  int* channels, int* bit_depth) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(encoded.data());
  const int data_size = encoded.size();

  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO)) {
    return false;
  }

  JxlDecoderSetInput(dec.get(), data, data_size);
  JxlDecoderCloseInput(dec.get());

  JxlBasicInfo info;

  JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

  if (status == JXL_DEC_ERROR) {
    return false;
  } else if (status == JXL_DEC_NEED_MORE_INPUT) {
    return false;
  } else if (status == JXL_DEC_BASIC_INFO) {
    if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
      return false;
    }
    if (width != nullptr) {
      *width = info.xsize;
    }
    if (height != nullptr) {
      *height = info.ysize;
    }
    if (channels != nullptr) {
      *channels = info.num_color_channels + (info.alpha_bits != 0);
    }
    if (bit_depth != nullptr) {
      *bit_depth = info.bits_per_sample;
    }
    return true;
  }
  return false;
}

bool DecodeImage(absl::string_view encoded, int channels, int channel_bits,
                 void* output, size_t output_size_bytes) {
  if (output == nullptr) return false;
  if (channel_bits != 8 && channel_bits != 16 && channel_bits != 32) {
    return false;
  }
  // Multi-threaded parallel runner.
  auto runner = JxlResizableParallelRunnerMake(nullptr);

  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(),
                                JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE)) {
    return false;
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlResizableParallelRunner,
                                                     runner.get())) {
    return false;
  }

  JxlBasicInfo info;
  JxlDataType data_type;
  if (channel_bits == 8) {
    data_type = JXL_TYPE_UINT8;
  } else if (channel_bits == 16) {
    data_type = JXL_TYPE_UINT16;
  } else {
    data_type = JXL_TYPE_FLOAT;
  }
  JxlPixelFormat format = {static_cast<uint32_t>(channels), data_type,
                           JXL_NATIVE_ENDIAN, 0};
  size_t xsize = 0, ysize = 0;
  const size_t bytes_per_sample = channel_bits / 8;

  JxlDecoderSetInput(dec.get(),
                     reinterpret_cast<const uint8_t*>(encoded.data()),
                     encoded.size());
  JxlDecoderCloseInput(dec.get());

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_ERROR) {
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      return false;
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
        return false;
      }
      xsize = info.xsize;
      ysize = info.ysize;
      if (xsize == 0 || ysize == 0) return false;
      if (output_size_bytes < xsize * ysize * channels * bytes_per_sample) {
        return false;
      }

      JxlResizableParallelRunnerSetThreads(
          runner.get(),
          JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
        return false;
      }
      if (buffer_size != xsize * ysize * channels * bytes_per_sample) {
        return false;
      }
      if (output_size_bytes < buffer_size) {
        return false;
      }

      if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                         output, buffer_size)) {
        return false;
      }
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
      // Nothing to do. If the image is an animation, more
      // full frames may be decoded.
      return true;
    } else {
      return false;
    }
  }
}

bool DecodeImage(absl::string_view encoded, int channels, uint8_t* output,
                 size_t output_size_bytes) {
  return DecodeImage(encoded, channels, 8, output, output_size_bytes);
}

bool DecodeImage16(absl::string_view encoded, int channels, uint16_t* output,
                   size_t output_size_bytes) {
  return DecodeImage(encoded, channels, 16, output, output_size_bytes);
}

bool DecodeImageFloat(absl::string_view encoded, int channels, float* output,
                      size_t output_size_bytes) {
  return DecodeImage(encoded, channels, 32, output, output_size_bytes);
}

float DistanceFromQuality(float quality) {
  return JxlEncoderDistanceFromQuality(quality);
}

template <typename T>
bool WriteImageToBuffer(const void* image_data, int width, int height,
                        int channels, int channel_bits, float distance,
                        int effort, T* output) {
  if (image_data == nullptr || output == nullptr) return false;
  if (width <= 0 || height <= 0) return false;
  if (channels != 1 && channels != 3 && channels != 4) return false;
  if (channel_bits != 8 && channel_bits != 16) return false;
  if (!(distance >= 0.0f && distance <= 25.0f)) return false;

  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  if (enc == nullptr) return false;

  // A distance of exactly 0.0 means lossless. libjxl requires
  // uses_original_profile to be set before lossless can be enabled, so it must
  // be decided here, before JxlEncoderSetBasicInfo below.
  const bool is_lossless = (distance == 0.0f);

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = width;
  basic_info.ysize = height;
  basic_info.bits_per_sample = channel_bits;
  basic_info.uses_original_profile = is_lossless ? JXL_TRUE : JXL_FALSE;

  if (channels == 1) {
    basic_info.num_color_channels = 1;
    basic_info.alpha_bits = 0;
    basic_info.num_extra_channels = 0;
  } else if (channels == 3) {
    basic_info.num_color_channels = 3;
    basic_info.alpha_bits = 0;
    basic_info.num_extra_channels = 0;
  } else if (channels == 4) {
    basic_info.num_color_channels = 3;
    basic_info.alpha_bits = channel_bits;
    basic_info.num_extra_channels = 1;
  }

  if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(enc.get(), &basic_info)) {
    return false;
  }

  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding, /*is_gray=*/channels == 1);
  if (JXL_ENC_SUCCESS !=
      JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
    return false;
  }

  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  if (frame_settings == nullptr) {
    return false;
  }

  // JxlEncoderSetFrameDistance enables lossless encoding itself when distance
  // is 0.0, so no separate lossless flag is needed here.
  if (JXL_ENC_SUCCESS != JxlEncoderSetFrameDistance(frame_settings, distance)) {
    return false;
  }

  if (effort >= 1 && effort <= 9) {
    if (JXL_ENC_SUCCESS !=
        JxlEncoderFrameSettingsSetOption(
            frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, effort)) {
      return false;
    }
  }

  JxlDataType data_type =
      (channel_bits == 8) ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
  JxlPixelFormat pixel_format = {static_cast<uint32_t>(channels), data_type,
                                 JXL_NATIVE_ENDIAN, 0};
  const size_t bytes_per_sample = channel_bits / 8;
  const size_t buffer_size =
      static_cast<size_t>(width) * height * channels * bytes_per_sample;

  if (JXL_ENC_SUCCESS != JxlEncoderAddImageFrame(frame_settings, &pixel_format,
                                                 image_data, buffer_size)) {
    return false;
  }
  JxlEncoderCloseInput(enc.get());

  output->clear();
  constexpr size_t kInitialBufferSize = 65536;
  size_t total_written = 0;
  output->resize(kInitialBufferSize);

  for (;;) {
    uint8_t* next_out =
        reinterpret_cast<uint8_t*>(output->data()) + total_written;
    size_t avail_out = output->size() - total_written;
    JxlEncoderStatus process_result =
        JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    total_written = next_out - reinterpret_cast<uint8_t*>(output->data());

    if (process_result == JXL_ENC_SUCCESS) {
      output->resize(total_written);
      return true;
    }
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      output->resize(output->size() * 2);
    } else {
      return false;
    }
  }
}

template bool WriteImageToBuffer<std::string>(const void* image_data, int width,
                                              int height, int channels,
                                              int channel_bits, float distance,
                                              int effort, std::string* output);

template bool WriteImageToBuffer<tstring>(const void* image_data, int width,
                                          int height, int channels,
                                          int channel_bits, float distance,
                                          int effort, tstring* output);

}  // namespace jxl
}  // namespace tensorflow
