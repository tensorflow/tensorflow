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

#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "absl/strings/match.h"

#import <Metal/Metal.h>

#include <Availability.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"

// Compile-time message: print define name and value.
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "=" VALUE(var)

namespace tflite {
namespace gpu {
namespace metal {

id<MTLDevice> GetBestSupportedMetalDevice() { return MTLCreateSystemDefaultDevice(); }

absl::Status CreateComputeProgram(id<MTLDevice> device, const std::string& code,
                                  const std::string& function_name,
                                  const std::map<std::string, std::string>& macros,
                                  id<MTLComputePipelineState>* program) {
  id<MTLFunction> function;
  RETURN_IF_ERROR(CreateFunction(device, code, function_name, macros, &function));

  NSError* error = nil;
  *program = [device newComputePipelineStateWithFunction:function error:&error];
  if (!*program) {
    NSString* errorString =
        [NSString stringWithFormat:@"newComputePipelineStateWithFunction error: %@",
                                   [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }
  return absl::OkStatus();
}

absl::Status CreateComputeProgramWithArgumentBuffer(
    id<MTLDevice> device, const std::string& code, const std::string& function_name,
    const std::map<std::string, std::string>& macros, id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder) {
  if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    id<MTLFunction> function;
    RETURN_IF_ERROR(CreateFunction(device, code, "ComputeFunction", macros, &function));
    *arguments_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!*arguments_encoder) {
      return absl::InternalError("Failed to get MTLArgumentEncoder.");
    }
    MTLComputePipelineDescriptor* pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = function;
    NSError* error = nil;
    *program = [device newComputePipelineStateWithDescriptor:pipeline_desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nullptr
                                                       error:&error];
    if (!*program) {
      NSString* error_string =
          [NSString stringWithFormat:@"newComputePipelineStateWithDescriptor: %@",
                                     [error localizedDescription]];
      return absl::InternalError([error_string UTF8String]);
    }
    return absl::OkStatus();
  } else {
    return absl::InternalError("Metal argument buffers available since ios 11, tvos 11 or macos "
                               "10.13.");
  }
}

absl::Status CreateComputeProgramWithICBSupport(id<MTLDevice> device, const std::string& code,
                                                const std::string& function_name,
                                                const std::map<std::string, std::string>& macros,
                                                id<MTLComputePipelineState>* program,
                                                id<MTLArgumentEncoder>* arguments_encoder) {
  if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
    id<MTLFunction> function;
    RETURN_IF_ERROR(CreateFunction(device, code, "ComputeFunction", macros, &function));
    *arguments_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!*arguments_encoder) {
      return absl::InternalError("Failed to get MTLArgumentEncoder.");
    }
    MTLComputePipelineDescriptor* pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = function;
    pipeline_desc.supportIndirectCommandBuffers = TRUE;
    NSError* error = nil;
    *program = [device newComputePipelineStateWithDescriptor:pipeline_desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nullptr
                                                       error:&error];
    if (!*program) {
      NSString* error_string =
          [NSString stringWithFormat:@"newComputePipelineStateWithDescriptor: %@",
                                     [error localizedDescription]];
      return absl::InternalError([error_string UTF8String]);
    }
    return absl::OkStatus();
  } else {
    return absl::InternalError("Indirect compute command buffer available since ios 13, tvos 13 "
                               "or macos 11.00");
  }
}

absl::Status CreateFunction(id<MTLDevice> device, const std::string& code,
                            const std::string& function_name,
                            const std::map<std::string, std::string>& macros,
                            id<MTLFunction>* function) {
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

  // Runtime checks for the iOS version independently of minimum target iOS.
  if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion2_3];
  } else if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion2_2];
  } else if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion2_1];
  } else if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion2_0];
  } else if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion1_2];
  } else if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    [options setLanguageVersion:MTLLanguageVersion1_1];
  }
#if (defined(__MAC_10_11) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_11) ||    \
    (defined(__IPHONE_9_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0) || \
    (defined(__TVOS_9_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_9_0)
  // Minimum target OS version is able to support Metal.
#else
#pragma message(VAR_NAME_VALUE(__MAC_OS_X_VERSION_MIN_REQUIRED))
#pragma message(VAR_NAME_VALUE(__IPHONE_OS_VERSION_MIN_REQUIRED))
#pragma message(VAR_NAME_VALUE(__TV_OS_VERSION_MIN_REQUIRED))
// NOLINTBEGIN
#error \
    "The Metal delegate is not supported on current target SDK. Minimum supported os: iOS/tvOS 9.0, macOS 10.11"
// NOLINTEND
#endif

  NSMutableDictionary<NSString*, NSString*>* macros_dict = [NSMutableDictionary dictionary];
  for (const auto& pair : macros) {
    std::string key = pair.first;
    std::string value = pair.second;
    if (absl::StrContains(key, ' ')) {
      key = "\"" + key + "\"";
    }
    if (absl::StrContains(value, ' ')) {
      value = "\"" + value + "\"";
    }
    [macros_dict setObject:[NSString stringWithCString:value.c_str()
                                              encoding:[NSString defaultCStringEncoding]]
                    forKey:[NSString stringWithCString:key.c_str()
                                              encoding:[NSString defaultCStringEncoding]]];
  }

  [options setFastMathEnabled:YES];
  [options setPreprocessorMacros:macros_dict];
  NSError* error = nil;
  NSString* code_ns = [NSString stringWithCString:code.c_str()
                                         encoding:[NSString defaultCStringEncoding]];
  id<MTLLibrary> library = [device newLibraryWithSource:code_ns options:options error:&error];
  if (!library) {
    NSString* errorString =
        [NSString stringWithFormat:@"newLibraryWithSource: %@", [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }

  NSString* function_name_ns = [NSString stringWithCString:function_name.c_str()
                                                  encoding:[NSString defaultCStringEncoding]];
  *function = [library newFunctionWithName:function_name_ns];
  if (!*function) {
    NSString* errorString =
        [NSString stringWithFormat:@"newFunctionWithName: %@", [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }
  return absl::OkStatus();
}

int PixelFormatToSizeInBytes(MTLPixelFormat pixel_format) {
  if (pixel_format == MTLPixelFormatRGBA32Uint ||
      pixel_format == MTLPixelFormatRGBA32Sint ||
      pixel_format == MTLPixelFormatRGBA32Float) {
    return 16;
  } else if (pixel_format == MTLPixelFormatRGBA16Unorm ||
             pixel_format == MTLPixelFormatRGBA16Snorm ||
             pixel_format == MTLPixelFormatRGBA16Uint ||
             pixel_format == MTLPixelFormatRGBA16Sint ||
             pixel_format == MTLPixelFormatRGBA16Float) {
    return 8;
  } else if (pixel_format == MTLPixelFormatRGBA8Unorm ||
             pixel_format == MTLPixelFormatRGBA8Snorm ||
             pixel_format == MTLPixelFormatRGBA8Uint ||
             pixel_format == MTLPixelFormatRGBA8Sint) {
    return 4;
  }
  return -1;
}

MTLPixelFormat DataTypeToRGBAPixelFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::FLOAT32:
      return MTLPixelFormatRGBA32Float;
    case DataType::FLOAT16:
      return MTLPixelFormatRGBA16Float;
    case DataType::INT8:
      return normalized ? MTLPixelFormatRGBA8Snorm : MTLPixelFormatRGBA8Sint;
    case DataType::UINT8:
      return normalized ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA8Uint;
    case DataType::INT16:
      return normalized ? MTLPixelFormatRGBA16Snorm : MTLPixelFormatRGBA16Sint;
    case DataType::UINT16:
      return normalized ? MTLPixelFormatRGBA16Unorm : MTLPixelFormatRGBA16Uint;
    case DataType::INT32:
      return MTLPixelFormatRGBA32Sint;
    case DataType::UINT32:
      return MTLPixelFormatRGBA32Uint;
    case DataType::BOOL:
      return MTLPixelFormatRGBA8Uint;
    default:
      return MTLPixelFormatInvalid;
  }
}

void WriteDataToTexture2D(id<MTLTexture> texture, id<MTLDevice> device, const void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  id<MTLBuffer> temp_buffer = [device newBufferWithBytes:data
                                                  length:pixel_size * texture.width * texture.height
                                                 options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
  [blitCommandEncoder copyFromBuffer:temp_buffer
                        sourceOffset:0
                   sourceBytesPerRow:pixel_size * texture.width
                 sourceBytesPerImage:pixel_size * texture.width * texture.height
                          sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                           toTexture:texture
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blitCommandEncoder endEncoding];

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
}

void ReadDataFromTexture2D(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  const int buffer_size = pixel_size * texture.width * texture.height;
  id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                  options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
  [blitCommandEncoder copyFromTexture:texture
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                             toBuffer:temp_buffer
                    destinationOffset:0
               destinationBytesPerRow:pixel_size * texture.width
             destinationBytesPerImage:pixel_size * texture.width * texture.height];
  [blitCommandEncoder endEncoding];

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::memcpy(data, [temp_buffer contents], buffer_size);
}

void WriteDataToTexture3D(id<MTLTexture> texture, id<MTLDevice> device, const void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  id<MTLBuffer> temp_buffer =
      [device newBufferWithBytes:data
                          length:pixel_size * texture.width * texture.height * texture.depth
                         options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
  [blitCommandEncoder copyFromBuffer:temp_buffer
                        sourceOffset:0
                   sourceBytesPerRow:pixel_size * texture.width
                 sourceBytesPerImage:pixel_size * texture.width * texture.height
                          sourceSize:MTLSizeMake(texture.width, texture.height, texture.depth)
                           toTexture:texture
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blitCommandEncoder endEncoding];

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
}

void ReadDataFromTexture3D(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  const int buffer_size = pixel_size * texture.width * texture.height * texture.depth;
  id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                  options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
  [blitCommandEncoder copyFromTexture:texture
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(texture.width, texture.height, texture.depth)
                             toBuffer:temp_buffer
                    destinationOffset:0
               destinationBytesPerRow:pixel_size * texture.width
             destinationBytesPerImage:pixel_size * texture.width * texture.height];
  [blitCommandEncoder endEncoding];

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::memcpy(data, [temp_buffer contents], buffer_size);
}

void WriteDataToTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device, const void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  id<MTLBuffer> temp_buffer =
      [device newBufferWithBytes:data
                          length:pixel_size * texture.width * texture.height * texture.arrayLength
                         options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  for (int i = 0; i < texture.arrayLength; ++i) {
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromBuffer:temp_buffer
                          sourceOffset:pixel_size * texture.width * texture.height * i
                     sourceBytesPerRow:pixel_size * texture.width
                   sourceBytesPerImage:pixel_size * texture.width * texture.height
                            sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                             toTexture:texture
                      destinationSlice:i
                      destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blitCommandEncoder endEncoding];
  }

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
}

void ReadDataFromTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
  const int buffer_size = pixel_size * texture.width * texture.height * texture.arrayLength;
  id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                  options:MTLResourceStorageModeShared];

  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

  for (int i = 0; i < texture.arrayLength; ++i) {
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromTexture:texture
                            sourceSlice:i
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                               toBuffer:temp_buffer
                      destinationOffset:pixel_size * texture.width * texture.height * i
                 destinationBytesPerRow:pixel_size * texture.width
               destinationBytesPerImage:pixel_size * texture.width * texture.height];
    [blitCommandEncoder endEncoding];
  }

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::memcpy(data, [temp_buffer contents], buffer_size);
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
