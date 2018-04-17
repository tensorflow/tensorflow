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
  const char* buffer = static_cast<const char*>(serialized_data);
  size_t op_name_char_count = *reinterpret_cast<const size_t*>(buffer);
  buffer += sizeof(size_t);
  buffer += op_name_char_count;

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

void PluginTensorRT::configure(const nvinfer1::Dims* inputs, int num_inputs,
                               const nvinfer1::Dims* outputs, int num_outputs,
                               int max_batch_size) {
  for (int index = 0; index < num_inputs; index++) {
    nvinfer1::Dims dim;
    dim.nbDims = inputs[index].nbDims;
    for (int i = 0; i < dim.nbDims; i++) {
      dim.d[i] = inputs[index].d[i];
      dim.type[i] = inputs[index].type[i];
    }
    input_dim_list_.emplace_back(dim);
  }
  return;
}

size_t PluginTensorRT::getSerializationSize() {
  nvinfer1::Dims dim;
  return sizeof(size_t) + GetPluginName().size() +
         sizeof(input_dim_list_.size()) + sizeof(dim.nbDims) + sizeof(dim.d) +
         sizeof(dim.type);
}

void PluginTensorRT::serialize(void* serialized_data) {
  size_t op_name_size = GetPluginName().size();
  char* buffer = static_cast<char*>(serialized_data);
  std::memcpy(buffer, &op_name_size, sizeof(size_t));
  buffer += sizeof(size_t);

  std::memcpy(buffer, GetPluginName().data(), op_name_size);
  buffer += op_name_size;

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

bool PluginTensorRT::StoreAttribute(const std::string& key, const void* ptr,
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
