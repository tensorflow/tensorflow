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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_api_delegate.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/testing/tflite_model_reader.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace {

void FillInputTensor(tflite::Interpreter* interpreter) {
  for (int k = 0; k < interpreter->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter->typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i);
      }
    }
    if (tensor_ptr->type == kTfLiteInt32) {
      int* p = interpreter->typed_input_tensor<int>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 2;
      }
    }
  }
}

void CompareCPUGPUResults(tflite::Interpreter* cpu, tflite::Interpreter* gpu,
                          float eps) {
  for (int i = 0; i < cpu->outputs().size(); ++i) {
    const float* cpu_out = cpu->typed_output_tensor<float>(i);
    const float* gpu_out = gpu->typed_output_tensor<float>(i);
    auto out_n = tflite::NumElements(cpu->tensor(cpu->outputs()[i]));
    const int kMaxPrint = 10;
    int printed = 0;
    int total_different = 0;
    for (int k = 0; k < out_n; ++k) {
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

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << "Expected model path as second argument." << std::endl;
    return -1;
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (!model) {
    std::cerr << "FlatBufferModel::BuildFromFile failed, model path - "
              << argv[1] << std::endl;
    return -1;
  }
  tflite::ops::builtin::BuiltinOpResolver op_resolver;
  tflite::InterpreterBuilder builder(*model, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  builder(&cpu_inference);
  if (!cpu_inference) {
    std::cerr << "Failed to build CPU inference." << std::endl;
    return -1;
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to AllocateTensors for CPU inference." << std::endl;
    return -1;
  }
  FillInputTensor(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to Invoke CPU inference." << std::endl;
    return -1;
  }

  // GPU.
  std::unique_ptr<tflite::Interpreter> gpu_inference;
  builder(&gpu_inference);
  if (!gpu_inference) {
    std::cerr << "Failed to build GPU inference." << std::endl;
    return -1;
  }
  TfLiteGpuDelegateOptionsV2 options;
  options.is_precision_loss_allowed = -1;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
  options.max_delegated_partitions = 1;
  auto* gpu_delegate = TfLiteGpuDelegateV2Create(&options);
  status = gpu_inference->ModifyGraphWithDelegate(gpu_delegate);
  if (status != kTfLiteOk) {
    std::cerr << "ModifyGraphWithDelegate failed." << std::endl;
    return -1;
  }
  FillInputTensor(gpu_inference.get());
  status = gpu_inference->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to Invoke GPU inference." << std::endl;
    return -1;
  }

  CompareCPUGPUResults(cpu_inference.get(), gpu_inference.get(), 1e-4f);

  // CPU inference latency.
  auto start = std::chrono::high_resolution_clock::now();
  cpu_inference->Invoke();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU time - " << (end - start).count() * 1e-6f << "ms"
            << std::endl;

  // GPU inference latency.
  start = std::chrono::high_resolution_clock::now();
  gpu_inference->Invoke();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU time(CPU->GPU->CPU) - " << (end - start).count() * 1e-6f
            << "ms" << std::endl;

  TfLiteGpuDelegateV2Delete(gpu_delegate);
  return EXIT_SUCCESS;
}
