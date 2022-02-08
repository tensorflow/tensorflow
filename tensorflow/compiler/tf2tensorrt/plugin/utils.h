/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_UTILS_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"


namespace tensorflow {
namespace tensorrt {

// Get a plugin creator from the TFTRT plugin registry.
nvinfer1::IPluginCreator* GetPluginCreator(const char* name,
                                           const char* version);

// Initializes the TensorRT plugin registry if this hasn't been done yet.
void MaybeInitializeTrtPlugins(nvinfer1::ILogger* trt_logger);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_PLUGIN_UTILS_H_
