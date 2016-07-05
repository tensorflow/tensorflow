/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Functions to read and write images in GIF format.

#include "tensorflow/core/lib/gif/gif_io.h"
#include "tensorflow/core/platform/gif.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gif {

int input_callback(GifFileType* gif_file, GifByteType* buf, int size) {
  if (gif_file->UserData && memcpy(buf, gif_file->UserData, size)) {
    gif_file->UserData = ((uint8_t*)gif_file->UserData) + size;
    return size;
  }
  return 0;
}

uint8* Decode(const void* srcdata, int datasize,
              std::function<uint8*(int, int, int)> allocate_output) {
  int error_code = D_GIF_SUCCEEDED;
  GifFileType* gif_file = DGifOpen(const_cast<void *>(srcdata),
                                   &input_callback,
                                   &error_code);
  if (error_code != D_GIF_SUCCEEDED) {
    LOG(ERROR) << "Fail to open gif file, reason: "
        << GifErrorString(error_code);
    return nullptr;
  }
  if (DGifSlurp(gif_file) != GIF_OK) {
    LOG(ERROR) << "Fail to slurp gif file, reason: "
        << GifErrorString(gif_file->Error);
    return nullptr;
  }
  if (gif_file->ImageCount <= 0) {
    LOG(ERROR) << "Gif file does not contain any image";
    return nullptr;
  }

  SavedImage* first_image = &gif_file->SavedImages[0];
  ColorMapObject* color_map = first_image->ImageDesc.ColorMap ?
      first_image->ImageDesc.ColorMap : gif_file->SColorMap;
  int width = first_image->ImageDesc.Width;
  int height = first_image->ImageDesc.Height;
  int channel = 3;

  uint8* dstdata = allocate_output(width, height, channel);
  for (int i = 0; i < height; ++i) {
    uint8* p_dst = dstdata + i * width * channel;
    for (int j = 0; j < width; ++j) {
      GifByteType color_index = first_image->RasterBits[i * width + j];
      const GifColorType& gif_color = color_map->Colors[color_index];
      p_dst[j * channel + 0] = gif_color.Red;
      p_dst[j * channel + 1] = gif_color.Green;
      p_dst[j * channel + 2] = gif_color.Blue;
    }
  }

  if (DGifCloseFile(gif_file, &error_code) != GIF_OK) {
    LOG(WARNING) << "Fail to close gif file, reason: "
        << GifErrorString(error_code);
  }
  return dstdata;
}

}  // namespace gif
}  // namespace tensorflow
