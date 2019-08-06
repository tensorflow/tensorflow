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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/plugin/trt_plugin.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
using nvinfer1::DataType;
using nvinfer1::Dims;
using nvinfer1::IPluginCreator;
using nvinfer1::IPluginV2;
using nvinfer1::IPluginV2Ext;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PluginFormat;

template <typename SrcT, typename DstT>
__global__ void Cast(const SrcT* input, int num_elements, DstT* output) {
  for (int i : CudaGridRangeX(num_elements)) {
    output[i] = static_cast<DstT>(input[i]);
  }
}

template <typename SrcT, typename DstT>
void RunCast(const SrcT* d_input, int num_elements, DstT* d_output,
             cudaStream_t stream) {
  const int threads_per_block = 256;
  const int blocks_per_grid =
      (num_elements + threads_per_block - 1) / threads_per_block;
  TF_CHECK_OK(CudaLaunchKernel(Cast<SrcT, DstT>, threads_per_block,
                               blocks_per_grid, 0, stream, d_input,
                               num_elements, d_output));
}

const char* kPluginName = "TfTrtPluginCast";

class CastPlugin : public TrtPlugin {
 public:
  CastPlugin(DataType src_type, DataType dst_type)
      : src_type_(src_type), dst_type_(dst_type) {}

  CastPlugin(const void* serialized_data, size_t length)
      : TrtPlugin(serialized_data, length) {
    const char* buffer = static_cast<const char*>(serialized_data);
    src_type_ = ReadFromBuffer<DataType>(&buffer);
    dst_type_ = ReadFromBuffer<DataType>(&buffer);
    src_dims_ = ReadFromBuffer<Dims>(&buffer);
  }

  CastPlugin(const CastPlugin& rhs)
      : TrtPlugin(rhs),
        src_type_(rhs.src_type_),
        dst_type_(rhs.dst_type_),
        src_dims_(rhs.src_dims_) {}

  // Methods from IPluginV2Ext.

  DataType getOutputDataType(int index, const DataType* input_types,
                             int num_inputs) const override {
    DCHECK_EQ(0, index);
    DCHECK_EQ(1, num_inputs);
    return dst_type_;
  }

  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcasted,
                                    int num_inputs) const override {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int input_index) const override {
    return false;
  }

  void configurePlugin(const Dims* input_dims, int num_inputs,
                       const Dims* output_dims, int num_outputs,
                       const DataType* input_types,
                       const DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       PluginFormat float_format, int max_batch_size) override {
    DCHECK_EQ(1, num_inputs);
    DCHECK_EQ(1, num_outputs);
    DCHECK(src_type_ == input_types[0]);
    DCHECK(dst_type_ == output_types[0]);
    src_dims_ = input_dims[0];
  }

  IPluginV2Ext* clone() const override { return new CastPlugin(*this); }

  // Methods from IPluginV2.

  const char* getPluginType() const override { return kPluginName; };

  const char* getPluginVersion() const override { return kTfTrtPluginVersion; };

  int getNbOutputs() const override { return 1; }

  Dims getOutputDimensions(int index, const Dims* inputs,
                           int num_input_dims) override {
    DCHECK_EQ(0, index);
    DCHECK_EQ(1, num_input_dims);
    return inputs[0];
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return type == DataType::kFLOAT || type == DataType::kINT32;
  }

  size_t getWorkspaceSize(int max_batch_size) const override { return 0; }

  int enqueue(int batch_size, const void* const* inputs, void** outputs, void*,
              cudaStream_t stream) override {
    int num_elements = batch_size;
    for (int i = 0; i < src_dims_.nbDims; i++) {
      num_elements *= src_dims_.d[i];
    }
    const void* input = inputs[0];
    void* output = outputs[0];
    DCHECK_NE(static_cast<int>(src_type_), static_cast<int>(dst_type_));

    switch (src_type_) {
      case DataType::kFLOAT:
        RunCast(reinterpret_cast<const float*>(input), num_elements,
                reinterpret_cast<int32*>(output), stream);
        break;
      case DataType::kINT32:
        RunCast(reinterpret_cast<const int32*>(input), num_elements,
                reinterpret_cast<float*>(output), stream);
        break;
      default:
        return 1;  // Indicates a failure.
    }
    return 0;
  }

  size_t getSerializationSize() const override {
    return 2 * sizeof(DataType) + sizeof(Dims);
  }

  void serialize(void* serialized_data) const override {
    char* buffer = static_cast<char*>(serialized_data);
    WriteToBuffer(src_type_, &buffer);
    WriteToBuffer(dst_type_, &buffer);
    WriteToBuffer(src_dims_, &buffer);
  }

 private:
  DataType src_type_;
  DataType dst_type_;
  Dims src_dims_;
};

class CastPluginCreator : public IPluginCreator {
 public:
  CastPluginCreator() {
    setPluginNamespace(kTfTrtPluginNamespace);
    plugin_fields_.emplace_back(
        PluginField("SrcT", nullptr, PluginFieldType::kINT32, 1));
    plugin_fields_.emplace_back(
        PluginField("DstT", nullptr, PluginFieldType::kINT32, 1));

    field_collection_.nbFields = plugin_fields_.size();
    field_collection_.fields = plugin_fields_.data();
  }

  const char* getPluginName() const override { return kPluginName; }

  const char* getPluginVersion() const override { return kTfTrtPluginVersion; }

  const PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  IPluginV2* createPlugin(
      const char* name,
      const PluginFieldCollection* field_collection) override {
    const PluginField* fields = field_collection->fields;
    DataType src_type, dst_type;
    for (int i = 0; i < field_collection->nbFields; ++i) {
      const char* attr_name = fields[i].name;
      if (!strcmp(attr_name, "SrcT")) {
        src_type = *static_cast<const DataType*>(fields[i].data);
      } else if (!strcmp(attr_name, "DstT")) {
        dst_type = *static_cast<const DataType*>(fields[i].data);
      } else {
        return nullptr;
      }
    }
    return new CastPlugin(src_type, dst_type);
  }

  IPluginV2* deserializePlugin(const char* name, const void* serial_data,
                               size_t serial_len) override {
    return new CastPlugin(serial_data, serial_len);
  }

  void setPluginNamespace(const char* plugin_namespace) override {
    namespace_ = plugin_namespace;
  }

  const char* getPluginNamespace() const override { return namespace_.c_str(); }

 private:
  PluginFieldCollection field_collection_;
  std::vector<PluginField> plugin_fields_;
  std::string namespace_;
};

REGISTER_TFTRT_PLUGIN(CastPluginCreator);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
