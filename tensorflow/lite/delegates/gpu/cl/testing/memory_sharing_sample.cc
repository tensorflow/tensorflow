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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <ostream>
#include <set>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// returns storage types that can be used with shared common buffer
std::set<TensorStorageType> GetSupportedStorages(const GpuInfo& gpu_info) {
  std::set<TensorStorageType> supported_storages;
  if (gpu_info.IsCL11OrHigher()) {
    supported_storages.insert(TensorStorageType::BUFFER);
  }
  if (CanUseSubBufferForImage2d(gpu_info)) {
    supported_storages.insert(TensorStorageType::TEXTURE_2D);
    supported_storages.insert(TensorStorageType::IMAGE_BUFFER);
  }
  return supported_storages;
}

absl::Status RunSample(const std::string& model_name_mv1,
                       const std::string& model_name_mv2) {
  // mv1 postfix here and later is for mobilenet_v1
  // mv2 postfix here and later is for mobilenet_v2
  auto flatbuffer_mv1 =
      tflite::FlatBufferModel::BuildFromFile(model_name_mv1.c_str());
  GraphFloat32 graph_mv1;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer_mv1, op_resolver, &graph_mv1,
                                      /*allow_quant_ops*/ true));

  auto flatbuffer_mv2 =
      tflite::FlatBufferModel::BuildFromFile(model_name_mv2.c_str());
  GraphFloat32 graph_mv2;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer_mv2, op_resolver, &graph_mv2,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));
  const auto& gpu_info = env.GetDevicePtr()->GetInfo();

  auto supported_storages = GetSupportedStorages(gpu_info);
  if (supported_storages.empty()) {
    return absl::UnimplementedError("No solution for this device");
  }
  TensorStorageType storage_type = *supported_storages.begin();
  if (gpu_info.IsAdreno()) {
    if (supported_storages.find(TensorStorageType::TEXTURE_2D) !=
        supported_storages.end()) {
      storage_type = TensorStorageType::TEXTURE_2D;
    } else if (supported_storages.find(TensorStorageType::IMAGE_BUFFER) !=
               supported_storages.end()) {
      storage_type = TensorStorageType::IMAGE_BUFFER;
    }
  }
  if (gpu_info.IsMali()) {
    if (supported_storages.find(TensorStorageType::TEXTURE_2D) !=
        supported_storages.end()) {
      storage_type = TensorStorageType::TEXTURE_2D;
    } else if (supported_storages.find(TensorStorageType::BUFFER) !=
               supported_storages.end()) {
      storage_type = TensorStorageType::BUFFER;
    }
  }

  CreateGpuModelInfo create_info_mv1;
  create_info_mv1.precision = env.IsSupported(CalculationsPrecision::F16)
                                  ? CalculationsPrecision::F16
                                  : CalculationsPrecision::F32;
  create_info_mv1.storage_type = storage_type;
  create_info_mv1.hints.Add(ModelHints::kAllowSpecialKernels);

  CreateGpuModelInfo create_info_mv2 = create_info_mv1;

  Tensor input_224_224, output_mv1, output_mv2;
  auto data_type = DeduceDataTypeFromPrecision(create_info_mv1.precision);
  RETURN_IF_ERROR(CreateTensor(
      env.context(),
      CreateHwcTensorDescriptor(data_type, TensorStorageType::TEXTURE_2D,
                                HWC(224, 224, 3)),
      &input_224_224));
  RETURN_IF_ERROR(
      CreateTensor(env.context(),
                   CreateHwcTensorDescriptor(
                       data_type, TensorStorageType::BUFFER, HWC(1, 1, 1001)),
                   &output_mv1));
  RETURN_IF_ERROR(
      CreateTensor(env.context(),
                   CreateHwcTensorDescriptor(
                       data_type, TensorStorageType::BUFFER, HWC(1, 1, 1001)),
                   &output_mv2));

  create_info_mv1.external_immutable_tensors = {
      {graph_mv1.inputs()[0]->id, &input_224_224},
      {graph_mv1.outputs()[0]->id, &output_mv1},
  };
  create_info_mv2.external_immutable_tensors = {
      {graph_mv2.inputs()[0]->id, &input_224_224},
      {graph_mv2.outputs()[0]->id, &output_mv2},
  };

  RETURN_IF_ERROR(RunGraphTransformsForGpuModel(&graph_mv1));
  GpuModel gpu_model_mv1;
  RETURN_IF_ERROR(
      GraphToGpuModel(graph_mv1, create_info_mv1, gpu_info, &gpu_model_mv1));
  uint64_t total_size_mv1 = 0;
  RETURN_IF_ERROR(GetTotalBufferSizeForTensors(gpu_model_mv1, create_info_mv1,
                                               gpu_info, &total_size_mv1));

  RETURN_IF_ERROR(RunGraphTransformsForGpuModel(&graph_mv2));
  GpuModel gpu_model_mv2;
  RETURN_IF_ERROR(
      GraphToGpuModel(graph_mv2, create_info_mv2, gpu_info, &gpu_model_mv2));
  uint64_t total_size_mv2 = 0;
  RETURN_IF_ERROR(GetTotalBufferSizeForTensors(gpu_model_mv2, create_info_mv2,
                                               gpu_info, &total_size_mv2));

  uint64_t total_size = std::max(total_size_mv1, total_size_mv2);

  Buffer shared_buffer;
  RETURN_IF_ERROR(
      CreateReadWriteBuffer(total_size, &env.context(), &shared_buffer));
  InferenceContext context_mv1;
  RETURN_IF_ERROR(context_mv1.InitFromGpuModel(create_info_mv1, &gpu_model_mv1,
                                               &env, nullptr, &shared_buffer));

  InferenceContext context_mv2;
  RETURN_IF_ERROR(context_mv2.InitFromGpuModel(create_info_mv2, &gpu_model_mv2,
                                               &env, nullptr, &shared_buffer));

  {  // profiling mv1
    auto* queue = env.profiling_queue();
    ProfilingInfo profiling_info;
    RETURN_IF_ERROR(context_mv1.Profile(queue, &profiling_info));
    std::cout << profiling_info.GetDetailedReport() << std::endl;
  }
  {  // profiling mv2
    auto* queue = env.profiling_queue();
    ProfilingInfo profiling_info;
    RETURN_IF_ERROR(context_mv2.Profile(queue, &profiling_info));
    std::cout << profiling_info.GetDetailedReport() << std::endl;
  }

  {
    const uint64_t runtime_mem_bytes =
        context_mv1.GetSizeOfMemoryAllocatedForIntermediateTensors();
    std::cout << "Memory for intermediate tensors - "
              << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    const uint64_t const_mem_bytes = context_mv1.GetConstantTensorsSize();
    std::cout << "Memory for constant tensors - "
              << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Total tensors memory(const + intermediate) - "
              << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0
              << " MB" << std::endl;
  }
  {
    const uint64_t runtime_mem_bytes =
        context_mv2.GetSizeOfMemoryAllocatedForIntermediateTensors();
    std::cout << "Memory for intermediate tensors - "
              << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    const uint64_t const_mem_bytes = context_mv2.GetConstantTensorsSize();
    std::cout << "Memory for constant tensors - "
              << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Total tensors memory(const + intermediate) - "
              << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0
              << " MB" << std::endl;
  }

  const uint64_t runtime_mem_bytes = shared_buffer.GetMemorySizeInBytes();
  const uint64_t inout_mem_bytes = input_224_224.GetMemorySizeInBytes() +
                                   output_mv1.GetMemorySizeInBytes() +
                                   output_mv2.GetMemorySizeInBytes();
  std::cout
      << "Total consumed memory size(2 models) for intermediate tensors - "
      << (runtime_mem_bytes + inout_mem_bytes) / 1024.0 / 1024.0 << " MB"
      << std::endl;
  const uint64_t total_constant_size = context_mv1.GetConstantTensorsSize() +
                                       context_mv2.GetConstantTensorsSize();
  std::cout << "Total consumed memory size(2 models, runtime + constant) - "
            << (runtime_mem_bytes + inout_mem_bytes + total_constant_size) /
                   1024.0 / 1024.0
            << " MB" << std::endl;

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
  if (argc <= 2) {
    std::cerr << "Expected 2 model path as arguments.";
    return -1;
  }

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cerr << load_status.message();
    return -1;
  }

  auto run_status = tflite::gpu::cl::RunSample(argv[1], argv[2]);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  return EXIT_SUCCESS;
}
