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

#import "tensorflow/lite/delegates/gpu/metal/buffer_convert.h"

#import <Metal/Metal.h>

#include <string>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::DivideRoundUp;
using ::tflite::gpu::metal::CreateComputeProgram;

@implementation TFLBufferConvert {
  id<MTLComputePipelineState> _program;
}

- (id)initWithDevice:(id<MTLDevice>)device
           isFloat16:(bool)isFloat16
     convertToPBHWC4:(bool)convertToPBHWC4 {
  if (self = [super init]) {
    std::string shaderSource;
    if (convertToPBHWC4) {
      shaderSource = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void ComputeFunction(device float* const input_buffer [[buffer(0)]],
                                    device FLT4* output_buffer [[buffer(1)]],
                                    constant int4& size [[buffer(2)]],
                                    uint3 gid[[thread_position_in_grid]]) {
          int linear_id = static_cast<int>(gid.x);
          int X = linear_id / size.w;
          int B = linear_id % size.w;
          int Y = static_cast<int>(gid.y);
          int S = static_cast<int>(gid.z);
          if (X >= size.x || Y >= size.y) {
            return;
          }
          FLT4 value = FLT4(0.0);
          for (int i = 0; i < 4; i++) {
            int channel = S * 4 + i;
            if (channel >= size.z) break;
            const int bhwc_index = ((B * size.y + Y) * size.x + X) * size.z + channel;
            value[i] = input_buffer[bhwc_index];
          }
          const int shwbc4_index = ((S * size.y + Y) * size.x + X) * size.w + B;
          output_buffer[shwbc4_index] = value;
        }
      )";
    } else {
      shaderSource = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void ComputeFunction(device FLT4* const input_buffer [[buffer(0)]],
                                    device float* output_buffer [[buffer(1)]],
                                    constant int4& size [[buffer(2)]],
                                    uint3 gid[[thread_position_in_grid]]) {
          int linear_id = static_cast<int>(gid.x);
          int X = linear_id / size.w;
          int B = linear_id % size.w;
          int Y = static_cast<int>(gid.y);
          int S = static_cast<int>(gid.z);
          if (X >= size.x || Y >= size.y) {
            return;
          }
          const int shwbc4_index = ((S * size.y + Y) * size.x + X) * size.w + B;
          FLT4 value = input_buffer[shwbc4_index];
          for (int i = 0; i < 4; i++) {
            int channel = S * 4 + i;
            if (channel >= size.z) break;
            const int bhwc_index = ((B * size.y + Y) * size.x + X) * size.z + channel;
            output_buffer[bhwc_index] = value[i];
          }
        }
      )";
    }
    const std::map<std::string, std::string> macros = {{"FLT4", isFloat16 ? "half4" : "float4"}};
    id<MTLComputePipelineState> program;
    if (CreateComputeProgram(device, shaderSource, "ComputeFunction", macros, &program).ok()) {
      _program = program;
      return self;
    }
  }
  return nil;
}

- (void)convertWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                     shape:(const BHWC&)shape
              sourceBuffer:(id<MTLBuffer>)sourceBuffer
           convertedBuffer:(id<MTLBuffer>)convertedBuffer {
  [encoder setComputePipelineState:_program];
  [encoder setBuffer:sourceBuffer offset:0 atIndex:0];
  [encoder setBuffer:convertedBuffer offset:0 atIndex:1];

  std::vector<int> uniforms = {shape.w, shape.h, shape.c, shape.b};
  [encoder setBytes:uniforms.data() length:uniforms.size() * sizeof(int) atIndex:2];

  MTLSize group_size = MTLSizeMake(16, 8, 1);
  int slices = DivideRoundUp(shape.c, 4);
  int groups_x = DivideRoundUp(shape.w * shape.b, group_size.width);
  int groups_y = DivideRoundUp(shape.h, group_size.height);
  int groups_z = DivideRoundUp(slices, group_size.depth);
  MTLSize groups_count = MTLSizeMake(groups_x, groups_y, groups_z);
  [encoder dispatchThreadgroups:groups_count threadsPerThreadgroup:group_size];
}

@end
