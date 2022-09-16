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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {

absl::Status RunPredefinedLayoutSample(const std::string& model_name) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops=*/true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  {
    // Example of adding predefined descriptor
    // Assumed that graph has first input with batch = 1.
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    create_info.predefined[graph_cl.inputs()[0]->id] =
        TensorDescriptor{data_type, TensorStorageType::BUFFER, Layout::HWC};
  }
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  // After initialization we can receive input tensor
  // in_ten will have TensorStorageType::BUFFER storage type
  Tensor* in_ten = context.GetTensor(graph_cl.inputs()[0]->id);
  if (in_ten->GetStorageType() != TensorStorageType::BUFFER) {
    return absl::InternalError("Failed preconditiion");
  }

  RETURN_IF_ERROR(context.AddToQueue(env.queue()));

  std::cout << "Finished RunPredefinedLayoutSample." << std::endl;

  return absl::OkStatus();
}

absl::Status RunExternalImmutableSample(const std::string& model_name) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  // Example of external immutable tensors:
  std::vector<Tensor> outputs(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    // Assumed that graph outputs have batch size = 1.
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    TensorDescriptor required_tensor_desc = TensorDescriptor{
        data_type, TensorStorageType::TEXTURE_ARRAY, Layout::HWC};
    required_tensor_desc.SetBHWCShape(graph_cl.outputs()[i]->tensor.shape);
    RETURN_IF_ERROR(
        CreateTensor(env.context(), required_tensor_desc, &outputs[i]));
    create_info.external_immutable_tensors[graph_cl.outputs()[i]->id] =
        &outputs[i];
  }
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  RETURN_IF_ERROR(context.AddToQueue(env.queue()));

  // outputs can be used here. But AddToQueue do not have cpu
  // syncronization.
  RETURN_IF_ERROR(env.queue()->WaitForCompletion());

  TensorDescriptor desc;
  RETURN_IF_ERROR(outputs[0].ToDescriptor(&desc, env.queue()));
  TensorFloat32 cpu_tensor;
  desc.DownloadData(&cpu_tensor);
  std::cout << "First tensor data at index 0 - " << cpu_tensor.data[0]
            << std::endl;

  return absl::OkStatus();
}

absl::Status RunSerializedTest(const std::string& model_name) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  {  // calculating time without building serialized model
    InferenceContext test_context;
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        test_context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Inference context initialization total time - "
              << total_time_ms << "ms" << std::endl;
  }
  InferenceContext context;
  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(context.InitFromGraphWithTransforms(create_info, &graph_cl,
                                                      &env, &serialized_model));

  std::vector<TensorFloat32> src_tensors(graph_cl.inputs().size());
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    src_tensors[i].id = graph_cl.inputs()[i]->id;
    src_tensors[i].shape = graph_cl.inputs()[i]->tensor.shape;
    src_tensors[i].data.resize(src_tensors[i].shape.DimensionsProduct());
    for (int j = 0; j < src_tensors[i].data.size(); ++j) {
      src_tensors[i].data[j] = std::sin(j);
    }
  }
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    RETURN_IF_ERROR(context.SetInputTensor(graph_cl.inputs()[i]->id,
                                           src_tensors[i], env.queue()));
  }
  RETURN_IF_ERROR(context.AddToQueue(env.queue()));
  RETURN_IF_ERROR(env.queue()->WaitForCompletion());

  std::vector<TensorFloat32> dst_tensors(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    RETURN_IF_ERROR(context.GetOutputTensor(graph_cl.outputs()[i]->id,
                                            env.queue(), &dst_tensors[i]));
  }

  Environment env_v2;
  RETURN_IF_ERROR(CreateEnvironment(&env_v2));
  InferenceContext serialized_context;
  {
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        serialized_context.RestoreDeserialized(serialized_model, &env_v2));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Serialized inference context initialization total time - "
              << total_time_ms << "ms" << std::endl;
  }
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    RETURN_IF_ERROR(serialized_context.SetInputTensor(
        graph_cl.inputs()[i]->id, src_tensors[i], env_v2.queue()));
  }

  RETURN_IF_ERROR(serialized_context.AddToQueue(env_v2.queue()));
  RETURN_IF_ERROR(env_v2.queue()->WaitForCompletion());

  std::vector<TensorFloat32> dst_tensors_v2(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    RETURN_IF_ERROR(serialized_context.GetOutputTensor(
        graph_cl.outputs()[i]->id, env_v2.queue(), &dst_tensors_v2[i]));
  }

  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    if (dst_tensors[i].data.size() != dst_tensors_v2[i].data.size()) {
      std::cout << "Different sizes for " << i << " output tensor" << std::endl;
      break;
    }
    for (int j = 0; j < dst_tensors[i].data.size(); ++j) {
      if (dst_tensors[i].data[j] != dst_tensors_v2[i].data[j]) {
        std::cout << "Different elements for " << j << " element in " << i
                  << " tensor: " << dst_tensors[i].data[j] << " - "
                  << dst_tensors_v2[i].data[j] << std::endl;
        break;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RunCommandBufferSample(int num_tests, int num_runs_per_sec,
                                    Environment* env,
                                    InferenceContext* context) {
  if (!env->device().GetInfo().SupportsExtension("cl_khr_command_buffer")) {
    return absl::OkStatus();
  }

  cl_command_queue command_queue = env->queue()->queue();
  cl_int errcode_ret;
  std::vector<cl_command_buffer_khr> cbs(num_runs_per_sec);
  for (auto& cb : cbs) {
    cb = clCreateCommandBufferKHR(1, &command_queue, nullptr, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
      return absl::InternalError("Failed clCreateCommandBufferKHR.");
    }
    RETURN_IF_ERROR(context->AddToCommanBuffer(cb));
    errcode_ret = clFinalizeCommandBufferKHR(cb);
    if (errcode_ret != CL_SUCCESS) {
      return absl::InternalError("Failed clFinalizeCommandBufferKHR.");
    }
  }

  for (int i = 0; i < num_tests; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (auto& cb : cbs) {
      cl_int error_code =
          clEnqueueCommandBufferKHR(1, &command_queue, cb, 0, nullptr, nullptr);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to clEnqueueCommandBufferKHR - ",
                         CLErrorCodeToString(error_code)));
      }
      clFlush(command_queue);
    }
    clFinish(command_queue);
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    const double average_inference_time = total_time_ms / num_runs_per_sec;
    std::cout << "Total time CB - " << average_inference_time << "ms"
              << std::endl;
  }
  for (auto& cb : cbs) {
    clReleaseCommandBufferKHR(cb);
  }
  return absl::OkStatus();
}

absl::Status RunModelSample(const std::string& model_name) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  auto* queue = env.profiling_queue();
  ProfilingInfo profiling_info;
  RETURN_IF_ERROR(context.Profile(queue, &profiling_info));
  std::cout << profiling_info.GetDetailedReport() << std::endl;
  const uint64_t runtime_mem_bytes =
      context.GetSizeOfMemoryAllocatedForIntermediateTensors();
  std::cout << "Memory for intermediate tensors - "
            << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  const uint64_t const_mem_bytes = context.GetConstantTensorsSize();
  std::cout << "Memory for constant tensors - "
            << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  std::cout << "Total tensors memory(const + intermediate) - "
            << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0 << " MB"
            << std::endl;

  const int num_runs_per_sec = std::max(
      1, static_cast<int>(1000.0f / absl::ToDoubleMilliseconds(
                                        profiling_info.GetTotalTime())));

  const int kNumRuns = 10;
  for (int i = 0; i < kNumRuns; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_runs_per_sec; ++k) {
      RETURN_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    const double average_inference_time = total_time_ms / num_runs_per_sec;
    std::cout << "Total time - " << average_inference_time << "ms" << std::endl;
  }

  RETURN_IF_ERROR(
      RunCommandBufferSample(kNumRuns, num_runs_per_sec, &env, &context));

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "Expected model path as second argument.";
    return -1;
  }

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cerr << load_status.message();
    return -1;
  }

  auto run_status = tflite::gpu::cl::RunModelSample(argv[1]);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  bool run_serialized_test = false;
  if (run_serialized_test) {
    run_status = tflite::gpu::cl::RunSerializedTest(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  bool run_with_external_immutable_tensors = false;
  if (run_with_external_immutable_tensors) {
    run_status = tflite::gpu::cl::RunExternalImmutableSample(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  bool run_with_predefined_layout = false;
  if (run_with_predefined_layout) {
    run_status = tflite::gpu::cl::RunPredefinedLayoutSample(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  return EXIT_SUCCESS;
}
