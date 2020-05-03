// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ios_image_load.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

std::vector<uint8_t> LoadImageFromFile(const char* file_name, int* out_width, int* out_height,
                                       int* out_channels) {
  FILE* file_handle = fopen(file_name, "rb");
  fseek(file_handle, 0, SEEK_END);
  const size_t bytes_in_file = ftell(file_handle);
  fseek(file_handle, 0, SEEK_SET);
  std::vector<uint8_t> file_data(bytes_in_file);
  fread(file_data.data(), 1, bytes_in_file, file_handle);
  fclose(file_handle);

  CFDataRef file_data_ref =
      CFDataCreateWithBytesNoCopy(NULL, file_data.data(), bytes_in_file, kCFAllocatorNull);
  CGDataProviderRef image_provider = CGDataProviderCreateWithCFData(file_data_ref);

  const char* suffix = strrchr(file_name, '.');
  if (!suffix || suffix == file_name) {
    suffix = "";
  }
  CGImageRef image;
  if (strcasecmp(suffix, ".png") == 0) {
    image = CGImageCreateWithPNGDataProvider(image_provider, NULL, true, kCGRenderingIntentDefault);
  } else if ((strcasecmp(suffix, ".jpg") == 0) || (strcasecmp(suffix, ".jpeg") == 0)) {
    image =
        CGImageCreateWithJPEGDataProvider(image_provider, NULL, true, kCGRenderingIntentDefault);
  } else {
    CFRelease(image_provider);
    CFRelease(file_data_ref);
    fprintf(stderr, "Unknown suffix for file '%s'\n", file_name);
    *out_width = 0;
    *out_height = 0;
    *out_channels = 0;
    return std::vector<uint8_t>();
  }

  const int width = (int)CGImageGetWidth(image);
  const int height = (int)CGImageGetHeight(image);
  const int channels = 4;
  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  const int bytes_per_row = (width * channels);
  const int bytes_in_image = (bytes_per_row * height);
  std::vector<uint8_t> result(bytes_in_image);
  const int bits_per_component = 8;

  CGContextRef context =
      CGBitmapContextCreate(result.data(), width, height, bits_per_component, bytes_per_row,
                            color_space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(color_space);
  CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
  CGContextRelease(context);
  CFRelease(image);
  CFRelease(image_provider);
  CFRelease(file_data_ref);

  *out_width = width;
  *out_height = height;
  *out_channels = channels;
  return result;
}
