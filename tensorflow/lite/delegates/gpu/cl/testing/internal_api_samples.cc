/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/testing/tflite_model_reader.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
void FillInputTensors(tflite::Interpreter* interpreter) {
  for (int k = 0; k < interpreter->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter->typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i);
      }
    } else {
      std::cout << "No support of non Float32 input/output tensors"
                << std::endl;
    }
  }
}

void CompareCPUGPUResults(tflite::Interpreter* cpu,
                          const std::vector<int64_t>& outputs,
                          const std::vector<std::vector<float>>& gpu,
                          float eps) {
  for (int i = 0; i < gpu.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(outputs[i]);
    const float* cpu_out = tensor_ptr->data.f;
    const float* gpu_out = gpu[i].data();
    const int kMaxPrint = 10;
    int printed = 0;
    int total_different = 0;
    for (int k = 0; k < tensor_ptr->bytes / 4; ++k) {
      const float abs_diff = fabs(cpu_out[k] - gpu_out[k]);
      if (abs_diff > eps) {
        total_different++;
        if (printed < kMaxPrint) {
          std::cout << "Output #" << i << ": element #" << k << ": CPU value - "
                    << cpu_out[k] << ", GPU value - " << gpu_out[k]
                    << ", abs diff - " << abs_diff << std::endl;
          printed++;
        }
        if (printed == kMaxPrint) {
          std::cout << "Printed " << kMaxPrint
                    << " different elements, threshhold - " << eps
                    << ", next different elements skipped" << std::endl;
          printed++;
        }
      }
    }
    std::cout << "Total " << total_different
              << " different elements, for output #" << i << ", threshhold - "
              << eps << std::endl;
  }
}
}  // namespace

absl::Status RunModelSampleWithInternalAPISerializedKernels(
    const std::string& model_name, const std::vector<uint8_t>& kernel_cache);

absl::Status RunModelSampleWithInternalAPISerialized(
    tflite::Interpreter* cpu, const std::vector<uint8_t>& kernel_cache,
    const std::vector<uint8_t>& serialized_model);

// Run Jet with OpenCL internal API and compares correctness with TFLite CPU
absl::Status RunModelSampleWithInternalAPI(const std::string& model_name,
                                           std::vector<uint8_t>* kernel_cache) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());

  ops::builtin::BuiltinOpResolver op_resolver;
  InterpreterBuilder tfl_builder(*flatbuffer, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  tfl_builder(&cpu_inference);
  if (!cpu_inference) {
    return absl::InternalError("Failed to build CPU inference.");
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to AllocateTensors for CPU inference.");
  }
  for (int k = 0; k < cpu_inference->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->inputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 input tensors");
    }
  }
  for (int k = 0; k < cpu_inference->outputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->outputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 output tensors");
    }
  }
  FillInputTensors(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();
  GraphFloat32 graph_cl;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

  auto inputs = graph_cl.inputs();
  auto outputs = graph_cl.outputs();
  std::vector<int64_t> in_refs(inputs.size());
  std::vector<int64_t> out_refs(outputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    in_refs[i] = inputs[i]->tensor.ref;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    out_refs[i] = outputs[i]->tensor.ref;
  }

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  std::unique_ptr<InferenceBuilder> builder;
  // Initializes builder.
  InferenceOptions options;
  options.priority1 = InferencePriority::MIN_LATENCY;
  options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
  options.priority3 = InferencePriority::MAX_PRECISION;
  options.usage = InferenceUsage::SUSTAINED_SPEED;

  RETURN_IF_ERROR(
      inf_env->NewInferenceBuilder(options, std::move(graph_cl), &builder));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Initialization total time - " << (end - start).count() * 1e-6f
            << "ms" << std::endl;

  if (kernel_cache) {
    *kernel_cache = inf_env->GetSerializedBinaryCache();
    std::cout << "Kernel cache size - " << kernel_cache->size() << std::endl;
  }

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  CompareCPUGPUResults(cpu_inference.get(), out_refs, output_tensors, 1e-4f);

  return absl::OkStatus();
}

absl::Status RunModelSampleWithInternalAPISerializedKernels(
    const std::string& model_name, const std::vector<uint8_t>& kernel_cache) {
  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());

  ops::builtin::BuiltinOpResolver op_resolver;
  InterpreterBuilder tfl_builder(*flatbuffer, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  tfl_builder(&cpu_inference);
  if (!cpu_inference) {
    return absl::InternalError("Failed to build CPU inference.");
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to AllocateTensors for CPU inference.");
  }
  for (int k = 0; k < cpu_inference->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->inputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 input tensors");
    }
  }
  for (int k = 0; k < cpu_inference->outputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->outputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 output tensors");
    }
  }
  FillInputTensors(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();
  GraphFloat32 graph_cl;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

  auto inputs = graph_cl.inputs();
  auto outputs = graph_cl.outputs();
  std::vector<int64_t> in_refs(inputs.size());
  std::vector<int64_t> out_refs(outputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    in_refs[i] = inputs[i]->tensor.ref;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    out_refs[i] = outputs[i]->tensor.ref;
  }

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  env_options.serialized_binary_cache =
      absl::MakeSpan(kernel_cache.data(), kernel_cache.size());
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  InferenceOptions options;
  options.priority1 = InferencePriority::MIN_LATENCY;
  options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
  options.priority3 = InferencePriority::MAX_PRECISION;
  options.usage = InferenceUsage::SUSTAINED_SPEED;

  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(inf_env->BuildSerializedModel(options, std::move(graph_cl),
                                                &serialized_model));
  std::unique_ptr<InferenceBuilder> builder;
  RETURN_IF_ERROR(inf_env->NewInferenceBuilder(serialized_model, &builder,
                                               /*in_refs*/ nullptr,
                                               /*out_refs*/ nullptr));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Initialization total time(with kernel cache) - "
            << (end - start).count() * 1e-6f << "ms" << std::endl;

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  CompareCPUGPUResults(cpu_inference.get(), out_refs, output_tensors, 1e-4f);

  RETURN_IF_ERROR(RunModelSampleWithInternalAPISerialized(
      cpu_inference.get(), kernel_cache, serialized_model));

  return absl::OkStatus();
}

absl::Status RunModelSampleWithInternalAPISerialized(
    tflite::Interpreter* cpu, const std::vector<uint8_t>& kernel_cache,
    const std::vector<uint8_t>& serialized_model) {
  FillInputTensors(cpu);
  auto status = cpu->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  env_options.serialized_binary_cache =
      absl::MakeSpan(kernel_cache.data(), kernel_cache.size());
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  std::vector<int64_t> in_refs;
  std::vector<int64_t> out_refs;
  std::unique_ptr<InferenceBuilder> builder;
  RETURN_IF_ERROR(inf_env->NewInferenceBuilder(serialized_model, &builder,
                                               &in_refs, &out_refs));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Serialized initialization total time - "
            << (end - start).count() * 1e-6f << "ms" << std::endl;

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  std::cout << "Comparing results second time:" << std::endl;

  CompareCPUGPUResults(cpu, out_refs, output_tensors, 1e-4f);

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

  std::vector<uint8_t> kernel_cache;
  auto run_status =
      tflite::gpu::cl::RunModelSampleWithInternalAPI(argv[1], &kernel_cache);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }
  run_status = tflite::gpu::cl::RunModelSampleWithInternalAPISerializedKernels(
      argv[1], kernel_cache);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  return EXIT_SUCCESS;
}
