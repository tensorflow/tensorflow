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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_

#import <Metal/Metal.h>

#include <map>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Returns system default device on iOS or Intel GPU on macOS.
id<MTLDevice> GetBestSupportedMetalDevice();

absl::Status CreateComputeProgram(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program);

absl::Status CreateComputeProgramWithArgumentBuffer(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder);

// ICB - indirect command buffer
absl::Status CreateComputeProgramWithICBSupport(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder);

absl::Status CreateFunction(id<MTLDevice> device, const std::string& code,
                            const std::string& function_name,
                            const std::map<std::string, std::string>& macros,
                            id<MTLFunction>* function);

int PixelFormatToSizeInBytes(MTLPixelFormat pixel_format);
MTLPixelFormat DataTypeToRGBAPixelFormat(DataType type, bool normalized = false);

void WriteDataToTexture2D(id<MTLTexture> texture, id<MTLDevice> device, const void* data);
void ReadDataFromTexture2D(id<MTLTexture> texture, id<MTLDevice> device, void* data);

void WriteDataToTexture3D(id<MTLTexture> texture, id<MTLDevice> device, const void* data);
void ReadDataFromTexture3D(id<MTLTexture> texture, id<MTLDevice> device, void* data);

void WriteDataToTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device, const void* data);
void ReadDataFromTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device, void* data);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_
