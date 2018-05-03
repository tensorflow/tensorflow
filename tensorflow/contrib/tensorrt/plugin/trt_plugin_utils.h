/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_UTILS
#define TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_UTILS

#include <functional>

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

typedef std::function<PluginTensorRT*(const void*, size_t)>
    PluginDeserializeFunc;

typedef std::function<PluginTensorRT*(void)> PluginConstructFunc;

// TODO(jie): work on error handling here
string ExtractOpName(const void* serial_data, size_t serial_length,
                     size_t* incremental);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_PLUGIN_TRT_PLUGIN_UTILS
