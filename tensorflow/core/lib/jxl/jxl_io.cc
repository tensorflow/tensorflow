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

#include "absl/strings/string_view.h"
#include "lib/include/jxl/codestream_header.h"  // from @jpegxl
#include "lib/include/jxl/decode.h"  // from @jpegxl
#include "lib/include/jxl/decode_cxx.h"  // from @jpegxl
#include "lib/include/jxl/resizable_parallel_runner.h"  // from @jpegxl
#include "lib/include/jxl/resizable_parallel_runner_cxx.h"  // from @jpegxl
#include "lib/include/jxl/types.h"  // from @jpegxl

namespace tensorflow {
namespace jxl {

bool HasJxlHeader(absl::string_view encoded) {
  JxlSignature signature = JxlSignatureCheck(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
  return signature == JXL_SIG_CODESTREAM || signature == JXL_SIG_CONTAINER;
}

bool DecodeHeader(absl::string_view encoded, int* width, int* height,
                  int* channels) {
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
    *width = info.xsize;
    *height = info.ysize;
    if (channels != nullptr) {
      *channels = info.num_color_channels + (info.alpha_bits != 0);
    }
    return true;
  }
  return false;
}

bool DecodeImage(absl::string_view encoded, int channels, uint8_t* output,
                 size_t output_size) {
  if (output == nullptr) return false;
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
  JxlPixelFormat format = {(uint32_t)channels, JXL_TYPE_UINT8,
                           JXL_NATIVE_ENDIAN, 0};
  size_t xsize = 0, ysize = 0;

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
      if (output_size < xsize * ysize * channels) {
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
      if (buffer_size != xsize * ysize * channels) {
        return false;
      }
      if (output_size < buffer_size) {
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

}  // namespace jxl
}  // namespace tensorflow
