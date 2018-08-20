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
#import <CoreGraphics/CoreGraphics.h>

std::vector<uint8_t> LoadImageFromUIImage(UIImage *img) {
    CGImageRef image = img.CGImage;
    
    const int width = (int)CGImageGetWidth(image);
    const int height = (int)CGImageGetHeight(image);
    const int channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = width * channels;
    const int bytes_in_image = bytes_per_row * height;
    std::vector<uint8_t> result(bytes_in_image);
    const int bits_per_component = 8;
    CGContextRef context = CGBitmapContextCreate(result.data(),
                                                 width,
                                                 height,
                                                 bits_per_component,
                                                 bytes_per_row, 
                                                 color_space,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    
    return result;
}
