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

// Functions to read images in GIF format.

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
              std::function<uint8*(int, int, int, int)> allocate_output) {
  int error_code = D_GIF_SUCCEEDED;
  GifFileType* gif_file =
      DGifOpen(const_cast<void*>(srcdata), &input_callback, &error_code);
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

  int num_frames = gif_file->ImageCount;
  int width = gif_file->SWidth;
  int height = gif_file->SHeight;
  int channel = 3;

  uint8* dstdata = allocate_output(num_frames, width, height, channel);
  for (int k = 0; k < num_frames; k++) {
    SavedImage* this_image = &gif_file->SavedImages[k];
    GifImageDesc* img_desc = &this_image->ImageDesc;
    if (img_desc->Left != 0 || img_desc->Top != 0 || img_desc->Width != width ||
        img_desc->Height != height) {
      LOG(ERROR) << "Can't process optimized gif.";
      return nullptr;
    }

    ColorMapObject* color_map = this_image->ImageDesc.ColorMap
                                    ? this_image->ImageDesc.ColorMap
                                    : gif_file->SColorMap;

    uint8* this_dst = dstdata + k * width * channel * height;
    for (int i = 0; i < height; ++i) {
      uint8* p_dst = this_dst + i * width * channel;
      for (int j = 0; j < width; ++j) {
        GifByteType color_index = this_image->RasterBits[i * width + j];
        const GifColorType& gif_color = color_map->Colors[color_index];
        p_dst[j * channel + 0] = gif_color.Red;
        p_dst[j * channel + 1] = gif_color.Green;
        p_dst[j * channel + 2] = gif_color.Blue;
      }
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
