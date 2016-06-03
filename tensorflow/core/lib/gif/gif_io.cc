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
#include "tensorflow/core/platform/freeimage.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gif {

uint8* Decode(const void* srcdata, int datasize,
              std::function<uint8*(int, int, int)> allocate_output) {
    FIMEMORY* fi_mem = nullptr;
    FIBITMAP* fi_bmp = nullptr;
    RGBQUAD* fi_palette = nullptr;
    uint8* dstdata = nullptr;

    FreeImage_Initialise();

    fi_mem = FreeImage_OpenMemory((BYTE*)srcdata, datasize);
    if (!fi_mem) {
        LOG(ERROR) << "Fail to load gif data to memory";
        FreeImage_CloseMemory(fi_mem);
        FreeImage_DeInitialise();
        return nullptr;
    }

    FREE_IMAGE_FORMAT fif_type = FreeImage_GetFileTypeFromMemory(fi_mem);
    if (fif_type != FIF_GIF) {
        LOG(ERROR) << "File type is not gif, got " << fif_type;
        FreeImage_CloseMemory(fi_mem);
        FreeImage_DeInitialise();
        return nullptr;
    }

    fi_bmp = FreeImage_LoadFromMemory(FIF_GIF, fi_mem, GIF_DEFAULT);
    if (!fi_bmp) {
        LOG(ERROR) << "Fail to load from memory";
        FreeImage_CloseMemory(fi_mem);
        FreeImage_DeInitialise();
        return nullptr;
    }

    int width = FreeImage_GetWidth(fi_bmp);
    int height = FreeImage_GetHeight(fi_bmp);
    int channel = 3;
    if (FreeImage_GetBPP(fi_bmp) != 8) {
        fi_bmp = FreeImage_ConvertTo8Bits(fi_bmp);
    }
    fi_palette = FreeImage_GetPalette(fi_bmp);
    if (!fi_palette) {
        LOG(ERROR) << "Fail to get palette";
        FreeImage_CloseMemory(fi_mem);
        FreeImage_Unload(fi_bmp);
        FreeImage_DeInitialise();
        return nullptr;
    }

    dstdata = allocate_output(width, height, channel);
    for (int i = 0; i < height; ++i) {
        uint8* p_dst = dstdata + i * width * channel;
        for (int j = 0; j < width; ++j) {
            BYTE intensity = 0;
            FreeImage_GetPixelIndex(fi_bmp, j, i, &intensity);
            p_dst[channel * j + 0] = fi_palette[intensity].rgbBlue;
            p_dst[channel * j + 1] = fi_palette[intensity].rgbGreen;
            p_dst[channel * j + 2] = fi_palette[intensity].rgbRed;
        }
    }

    FreeImage_CloseMemory(fi_mem);
    FreeImage_Unload(fi_bmp);
    FreeImage_DeInitialise();
    return dstdata;
}

}  // namespace gif
}  // namespace tensorflow
