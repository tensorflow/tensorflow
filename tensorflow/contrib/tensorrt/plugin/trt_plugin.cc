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

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin.h"
#include <cassert>
#include <cstring>
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_utils.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

PluginTensorRT::PluginTensorRT(const void* serialized_data, size_t length) {
  // sanity check.
  assert(EncodeOpName(GetPluginName()) !=
         *static_cast<size_t*>(serialized_data));
  const char* buffer = static_cast<const char*>(serialized_data) +
                       sizeof(input_dim_list_.size());

  size_t count = *reinterpret_cast<const size_t*>(buffer);
  buffer += sizeof(size_t);

  for (int i = 0; i < count; i++) {
    nvinfer1::Dims dim;
    std::memcpy(&(dim.nbDims), buffer, sizeof(dim.nbDims));
    buffer += sizeof(dim.nbDims);
    std::memcpy(dim.d, buffer, sizeof(dim.d));
    buffer += sizeof(dim.d);
    std::memcpy(dim.type, buffer, sizeof(dim.type));
    buffer += sizeof(dim.type);
    input_dim_list_.emplace_back(dim);
  }
}

size_t PluginTensorRT::getSerializationSize() {
  nvinfer1::Dims dim;
  return sizeof(size_t) + sizeof(input_dim_list_.size()) + sizeof(dim.nbDims) +
         sizeof(dim.d) + sizeof(dim.type);
}

void PluginTensorRT::serialize(void* serialized_data) {
  size_t encode_op_name = EncodeOpName(GetPluginName());
  char* buffer = static_cast<char*>(serialized_data);
  std::memcpy(buffer, &encode_op_name, sizeof(size_t));
  buffer += sizeof(size_t);

  auto list_size = input_dim_list_.size();
  std::memcpy(buffer, &list_size, sizeof(input_dim_list_.size()));
  buffer += sizeof(input_dim_list_.size());

  for (int i = 0; i < input_dim_list_.size(); i++) {
    auto dim = input_dim_list_[i];
    std::memcpy(buffer, &(dim.nbDims), sizeof(dim.nbDims));
    buffer += sizeof(dim.nbDims);
    std::memcpy(buffer, dim.d, sizeof(dim.d));
    buffer += sizeof(dim.d);
    std::memcpy(buffer, dim.type, sizeof(dim.type));
    buffer += sizeof(dim.type);
  }
}

bool PluginTensorRT::StoreAttribute(const string& key, const void* ptr,
                                    const size_t size) {
  if (attr_map_.count(key) != 0) return false;

  attr_map_.emplace(key, std::vector<char>(size));
  std::memcpy(attr_map_[key].data(), ptr, size);
  return true;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
