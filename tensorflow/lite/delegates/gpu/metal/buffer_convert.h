/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_CONVERT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_CONVERT_H_

#import <Metal/Metal.h>

#include "tensorflow/lite/delegates/gpu/common/shape.h"

@interface TFLBufferConvert : NSObject

/// Constructs converter from/to BHWC <-> BPHWC4
/// @param isFloat16 the BPHWC4 buffer is in float16 format.
/// @param convertToPBHWC4 convert BHWC -> BPHWC4 if true or BPHWC4 -> BHWC instead.
- (id)initWithDevice:(id<MTLDevice>)device
           isFloat16:(bool)isFloat16
     convertToPBHWC4:(bool)convertToPBHWC4;

/// Converts from/to BHWC <-> BPHWC4
/// @param shape shape of BHWC tensor.
- (void)convertWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                     shape:(const ::tflite::gpu::BHWC&)shape
              sourceBuffer:(id<MTLBuffer>)sourceBuffer
           convertedBuffer:(id<MTLBuffer>)convertedBuffer;

@end

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_CONVERT_H_
