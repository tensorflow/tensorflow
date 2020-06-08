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

#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/gl/api.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/object_manager.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"

namespace tflite {
namespace gpu {
namespace gl {

SingleOpModel::SingleOpModel(Operation&& operation,
                             const std::vector<TensorRef<BHWC>>& inputs,
                             const std::vector<TensorRef<BHWC>>& outputs) {
  auto node = graph_.NewNode();
  node->operation = std::move(operation);

  for (int i = 0; i < inputs.size(); ++i) {
    auto input = graph_.NewValue();
    input->tensor = inputs[i];
    graph_.AddConsumer(node->id, input->id).IgnoreError();
    TensorFloat32 tensor;
    tensor.id = input->tensor.ref;
    tensor.shape = input->tensor.shape;
    inputs_.emplace_back(std::move(tensor));
  }

  for (int i = 0; i < outputs.size(); ++i) {
    auto output = graph_.NewValue();
    output->tensor = outputs[i];
    graph_.SetProducer(node->id, output->id).IgnoreError();
  }
}

bool SingleOpModel::PopulateTensor(int index, std::vector<float>&& data) {
  if (index >= inputs_.size() ||
      inputs_[index].shape.DimensionsProduct() != data.size()) {
    return false;
  }
  inputs_[index].data = std::move(data);
  return true;
}

absl::Status SingleOpModel::Invoke(const CompilationOptions& compile_options,
                                   const RuntimeOptions& runtime_options,
                                   const NodeShader& shader) {
  std::unique_ptr<EglEnvironment> env;
  RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env));

  ObjectManager objects;

  // Create buffers for input tensors.
  {
    std::unordered_map<int, uint32_t> tensor_to_id;
    for (const auto* input : graph_.inputs()) {
      tensor_to_id[input->tensor.ref] = input->id;
    }
    for (const auto& input : inputs_) {
      GlBuffer buffer;
      RETURN_IF_ERROR(CreatePHWC4BufferFromTensor(input, &buffer));
      RETURN_IF_ERROR(
          objects.RegisterBuffer(tensor_to_id[input.id], std::move(buffer)));
    }
  }

  // Create buffers for output tensors.
  for (const auto* output : graph_.outputs()) {
    GlBuffer buffer;
    RETURN_IF_ERROR(CreatePHWC4BufferFromTensorRef(output->tensor, &buffer));
    RETURN_IF_ERROR(objects.RegisterBuffer(output->id, std::move(buffer)));
  }

  // Compile model.
  GpuInfo gpu_info;
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
  std::unique_ptr<CompiledModel> compiled_model;
  RETURN_IF_ERROR(Compile(
      compile_options, graph_, /*tflite_graph_io=*/std::unordered_set<int>(),
      shader, *NewDefaultWorkgroupsCalculator(gpu_info), &compiled_model));

  // Get inference context.
  auto command_queue = NewCommandQueue(gpu_info);
  std::unique_ptr<InferenceContext> inference_context;
  RETURN_IF_ERROR(compiled_model->NewRun(
      runtime_options, &objects, command_queue.get(), &inference_context));
  RETURN_IF_ERROR(inference_context->Reset());

  // Run inference.
  RETURN_IF_ERROR(inference_context->Execute());

  // Copy output tensors to `output_`.
  for (const auto* output : graph_.outputs()) {
    TensorFloat32 tensor;
    tensor.id = output->tensor.ref;
    tensor.shape = output->tensor.shape;
    tensor.data.reserve(output->tensor.shape.DimensionsProduct());
    RETURN_IF_ERROR(
        CopyFromPHWC4Buffer(*objects.FindBuffer(output->id), &tensor));
    outputs_.push_back(std::move(tensor));
  }
  return absl::OkStatus();
}

absl::Status SingleOpModel::Invoke(const NodeShader& shader) {
  return Invoke(CompilationOptions(), RuntimeOptions(), shader);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
