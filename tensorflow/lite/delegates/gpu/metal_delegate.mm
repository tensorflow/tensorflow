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

#import "tensorflow/lite/delegates/gpu/metal_delegate.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/general_transformations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/metal/api.h"
#include "tensorflow/lite/delegates/gpu/metal/buffer_convert.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  struct ValueRef {
    BHWC shape;
    int64_t tensor_id;
  };

 public:
  explicit Delegate(const GpuDelegateOptions* options) {
    if (options) {
      options_ = *options;
    } else {
      // Default options.
      options_.allow_precision_loss = false;
      options_.wait_type = GpuDelegateOptions::WaitType::kPassive;
    }
    metal_device_ = MTLCreateSystemDefaultDevice();
  }

  Status BindBufferToTensor(id<MTLBuffer> buffer, int tensor_index) {
    for (auto& input : graph_inputs_) {
      if (input.tensor_id == tensor_index) {
        input_output_buffers_[input.id] = buffer;
        bphwc4_buffers_[input.id] = buffer;
        input.set_externally = true;
        return OkStatus();
      }
    }
    for (auto& output : graph_outputs_) {
      if (output.tensor_id == tensor_index) {
        input_output_buffers_[output.id] = buffer;
        bphwc4_buffers_[output.id] = buffer;
        output.set_externally = true;
        return OkStatus();
      }
    }
    return NotFoundError("Couldn't find tensor: " + std::to_string(tensor_index));
  }

  void SetCommandEncoder(id<MTLComputeCommandEncoder> encoder) {
    external_command_encoder_ = encoder;
  }

  Status Prepare(TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
    // Extract TFLite delegate execution plan from the context and convert it into FlowGraph32.
    GraphFloat32 graph;
    RETURN_IF_ERROR(BuildModel(context, delegate_params, &graph));

    // Apply general transformations on the graph.
    NullTransformationReporter reporter;
    ModelTransformer transformer(&graph, &reporter);
    if (!ApplyGeneralTransformations(&transformer)) {
      return InternalError("Graph general transformations failed");
    }

    // TODO(impjdi): Remove code duplication.
    auto values = graph.values();
    tensors_.reserve(values.back()->id + 1);
    for (const auto* value : values) {
      if (tensors_.size() <= value->id) tensors_.resize(value->id + 1);
      tensors_[value->id] = {
          value->tensor.shape,  // .shape
          value->tensor.ref,    // .tensor_id
      };
    }

    // Prepare graph inputs.
    inputs_.reserve(delegate_params->input_tensors->size);
    for (auto input : graph.inputs()) {
      inputs_.push_back(input->id);

      auto tensor = &context->tensors[input->tensor.ref];
      tensor->buffer_handle = input->id;
      tensor->delegate = &delegate_;
    }

    // Prepare graph outputs.
    outputs_.reserve(delegate_params->output_tensors->size);
    for (auto output : graph.outputs()) {
      outputs_.push_back(output->id);

      auto tensor = &context->tensors[output->tensor.ref];
      tensor->buffer_handle = output->id;
      tensor->delegate = &delegate_;
    }

    size_t storage_type_size;
    RuntimeOptions runtime_options;
    if (options_.allow_precision_loss) {
      storage_type_size = sizeof(HalfBits);
      runtime_options.storage_precision = RuntimeOptions::Precision::FP16;
      runtime_options.accumulator_precision = RuntimeOptions::Precision::FP16;
    } else {
      storage_type_size = sizeof(float);
      runtime_options.storage_precision = RuntimeOptions::Precision::FP32;
      runtime_options.accumulator_precision = RuntimeOptions::Precision::FP32;
    }
    command_queue_ = [metal_device_ newCommandQueue];

    // TODO(impjdi): Merge logic with above.
    // Pre-allocate input and output metal buffers
    std::vector<::tflite::gpu::ValueId> input_ids;
    input_ids.reserve(inputs_.size());
    std::map<::tflite::gpu::ValueId, BHWC> input_dimensions;
    graph_inputs_.reserve(inputs_.size());
    for (const ValueId input : inputs_) {
      const auto& input_tensor = tensors_[input];
      const auto tensor_id = input_tensor.tensor_id;
      input_ids.push_back(input);
      if (input_tensor.shape.b != 1) return UnimplementedError("Batching is not supported yet.");
      input_dimensions[input] = input_tensor.shape;
      graph_inputs_.push_back({
          input,               // .id
          tensor_id,           // .tensor_id
          input_tensor.shape,  // .shape
          false,               // .set_externally
      });
      int bhwc_length = static_cast<int>(sizeof(float) * input_tensor.shape.DimensionsProduct());
      int bphwc4_length =
          static_cast<int>(storage_type_size * GetElementsSizeForPHWC4(input_tensor.shape));
      id<MTLBuffer> buffer = [metal_device_ newBufferWithLength:bhwc_length
                                                        options:MTLResourceStorageModeShared];
      input_output_buffers_[input] = buffer;
      if (options_.allow_precision_loss || input_tensor.shape.c != 4) {
        bphwc4_buffers_[input] = [metal_device_ newBufferWithLength:bphwc4_length
                                                            options:MTLResourceStorageModeShared];
        if (converter_to_BPHWC4_ == nil) {
          converter_to_BPHWC4_ =
              [[TFLBufferConvert alloc] initWithDevice:metal_device_
                                             isFloat16:options_.allow_precision_loss
                                       convertToPBHWC4:true];
          if (converter_to_BPHWC4_ == nil) {
            return InternalError("Error initialization of input buffer converter");
          }
        }
      } else {
        bphwc4_buffers_[input] = buffer;
      }
    }

    std::vector<::tflite::gpu::ValueId> output_ids;
    output_ids.reserve(outputs_.size());
    graph_outputs_.reserve(outputs_.size());
    for (const ValueId output : outputs_) {
      const auto& output_tensor = tensors_[output];
      const auto tensor_id = output_tensor.tensor_id;
      output_ids.push_back(output);
      graph_outputs_.push_back({
          output,               // .id
          tensor_id,            // .tensor_id
          output_tensor.shape,  // .shape
          false,                // .set_externally
      });
      // Create BHWC buffer
      int bhwc_length = static_cast<int>(sizeof(float) * output_tensor.shape.DimensionsProduct());
      int bphwc4_length =
          static_cast<int>(storage_type_size * GetElementsSizeForPHWC4(output_tensor.shape));
      id<MTLBuffer> buffer = [metal_device_ newBufferWithLength:bhwc_length
                                                        options:MTLResourceStorageModeShared];
      input_output_buffers_[output] = buffer;
      if (options_.allow_precision_loss || output_tensor.shape.c != 4) {
        bphwc4_buffers_[output] = [metal_device_ newBufferWithLength:bphwc4_length
                                                             options:MTLResourceStorageModeShared];
        if (converter_from_BPHWC4_ == nil) {
          converter_from_BPHWC4_ =
              [[TFLBufferConvert alloc] initWithDevice:metal_device_
                                             isFloat16:options_.allow_precision_loss
                                       convertToPBHWC4:false];
          if (converter_from_BPHWC4_ == nil) {
            return InternalError("Error initialization of output buffer converter");
          }
        }
      } else {
        bphwc4_buffers_[output] = buffer;
      }
    }

    // TODO(impjdi): Merge these.
    CompiledModel compiled_model;
    RETURN_IF_ERROR(Compile(graph, runtime_options, &compiled_model));
    CompiledModel optimized_model;
    RETURN_IF_ERROR(ValidateOptimizeModel(input_ids, output_ids, compiled_model, &optimized_model));

    inference_context_ = [[TFLInferenceContext alloc] init];
    RETURN_IF_ERROR([inference_context_ compileModelWithDevice:metal_device_
                                               taskDescriptors:optimized_model
                                               outputBufferIDs:output_ids
                                                runtimeOptions:runtime_options]);
    std::map<::tflite::gpu::ValueId, BHWC> output_dimensions;
    RETURN_IF_ERROR([inference_context_ setInputDimensions:input_dimensions
                                          outputDimensions:&output_dimensions
                                           taskDescriptors:optimized_model]);
    return OkStatus();
  }

  Status Invoke(TfLiteContext* context) {
    // We need only synchronization so volatile works better than atomic which reads from global
    // memory each time.
    __block volatile bool buffer_completed = false;
    __block id<MTLCommandBuffer> command_buffer;
    __block id<MTLComputeCommandEncoder> encoder = external_command_encoder_;
    if (encoder == nil) {
      command_buffer = [command_queue_ commandBuffer];
      encoder = [command_buffer computeCommandEncoder];
    }

    // CPU HWC input data conversion to PHWC4 and fill the GPU buffer
    for (const auto& input : graph_inputs_) {
      if (input.set_externally) continue;
      // A user provides data on CPU memory for this buffer - need to copy to MTLBuffer

      TfLiteTensor* tensor = context->tensors + input.tensor_id;
      void* gpu_ptr = [input_output_buffers_[input.id] contents];
      std::memcpy(gpu_ptr, tensor->data.f, input.shape.DimensionsProduct() * sizeof(float));
      if (input_output_buffers_[input.id] == bphwc4_buffers_[input.id]) continue;
      [converter_to_BPHWC4_ convertWithEncoder:encoder
                                         shape:input.shape
                                  sourceBuffer:input_output_buffers_[input.id]
                               convertedBuffer:bphwc4_buffers_[input.id]];
    }

    [inference_context_ encodeWithEncoder:encoder
                       inputOutputBuffers:bphwc4_buffers_
                             encoderBlock:^(bool isLast) {
                               if (external_command_encoder_ != nil ||
                                   options_.wait_type == GpuDelegateOptions::WaitType::kPassive) {
                                 return encoder;
                               }
                               if (isLast) {
                                 if (options_.wait_type == GpuDelegateOptions::WaitType::kActive) {
                                   [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
                                     buffer_completed = true;
                                   }];
                                 }
                               } else {
                                 [encoder endEncoding];
                                 [command_buffer commit];
                                 command_buffer = [command_queue_ commandBuffer];
                                 encoder = [command_buffer computeCommandEncoder];
                               }
                               return encoder;
                             }];
    for (const auto& output : graph_outputs_) {
      if (output.set_externally) continue;
      if (bphwc4_buffers_[output.id] == input_output_buffers_[output.id]) continue;
      [converter_from_BPHWC4_ convertWithEncoder:encoder
                                           shape:output.shape
                                    sourceBuffer:bphwc4_buffers_[output.id]
                                 convertedBuffer:input_output_buffers_[output.id]];
    }

    if (external_command_encoder_ == nil) {
      [encoder endEncoding];
      [command_buffer commit];
      if (options_.wait_type == GpuDelegateOptions::WaitType::kActive) {
        while (!buffer_completed) {
          // Busy wait.
        }
      } else if (options_.wait_type == GpuDelegateOptions::WaitType::kPassive) {
        // passive wait: this thread sleeps until GPU finishes.
        [command_buffer waitUntilCompleted];
      }
    } else {
      // External command encoder must be set before every invoke call.
      external_command_encoder_ = nil;
      // External command encoder is assigned so all output buffers are controlled by a user.
      for (const auto& output : graph_outputs_) {
        if (!output.set_externally) {
          return InternalError(
              "External command encoder is used, but not all output buffers are bound.");
        }
      }
      return OkStatus();
    }

    // Retrieve data from GPU and convert from PHWC4 to HWC.
    for (const auto& output : graph_outputs_) {
      if (output.set_externally) continue;
      // A user retrieves data on CPU memory for this buffer - need to copy from MTLBuffer.
      TfLiteTensor* tensor = context->tensors + output.tensor_id;
      const void* gpu_ptr = [input_output_buffers_[output.id] contents];
      std::memcpy(tensor->data.f, gpu_ptr, output.shape.DimensionsProduct() * sizeof(float));
    }
    return OkStatus();
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  GpuDelegateOptions options_;

  id<MTLDevice> metal_device_;

  std::vector<ValueRef> tensors_;  // indexed by ValueId
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;

  TFLInferenceContext* inference_context_;
  // input and output buffers are passed into Metal inference engine
  std::map<::tflite::gpu::ValueId, id<MTLBuffer>> input_output_buffers_;
  std::map<::tflite::gpu::ValueId, id<MTLBuffer>> bphwc4_buffers_;
  TFLBufferConvert* converter_to_BPHWC4_ = nil;
  TFLBufferConvert* converter_from_BPHWC4_ = nil;

  struct BufferDescriptor {
    ValueId id;
    int64_t tensor_id;
    BHWC shape;
    bool set_externally;  // a user fills/retrieves data on this MTLBuffer buffer
  };
  std::vector<BufferDescriptor> graph_inputs_;
  std::vector<BufferDescriptor> graph_outputs_;

  id<MTLComputeCommandEncoder> external_command_encoder_;
  id<MTLCommandQueue> command_queue_;
};

inline Delegate* GetMetalDelegate(TfLiteNode* node) {
  return reinterpret_cast<Delegate*>(node->user_data);
}

inline Delegate* GetMetalDelegate(TfLiteDelegate* delegate) {
  return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const TfLiteRegistration kRegistration = {
      // .init
      [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        const auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* metal_delegate = GetMetalDelegate(params->delegate);
        // Everything below should happen in prepare function call, but TFLite for whatever reason
        // forbids that.
        const auto status = metal_delegate->Prepare(context, params);
        if (status.ok()) return metal_delegate;
        context->ReportError(context, "TfLiteGpuDelegate Prepare: %s",
                             status.error_message().c_str());
        return nullptr;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {},
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetMetalDelegate(node)->Invoke(context);
        if (status.ok()) return kTfLiteOk;
        context->ReportError(context, "TfLiteMetalDelegate Invoke: %s",
                             status.error_message().c_str());
        return kTfLiteError;
      },
      nullptr,                // .profiling_string
      0,                      // .builtin_code
      "TfLiteMetalDelegate",  // .custom_name
      1,                      // .version
  };
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(context, kRegistration,
                                                                     ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* NewGpuDelegate(const GpuDelegateOptions* options) {
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for Metal.");
  auto* metal_delegate = new ::tflite::gpu::metal::Delegate(options);
  return metal_delegate ? metal_delegate->tflite_delegate() : nullptr;
}

void DeleteGpuDelegate(TfLiteDelegate* delegate) {
  delete ::tflite::gpu::metal::GetMetalDelegate(delegate);
}

bool BindMetalBufferToTensor(TfLiteDelegate* delegate, int tensor_index, id<MTLBuffer> buffer) {
  auto* metal_delegate = ::tflite::gpu::metal::GetMetalDelegate(delegate);
  return metal_delegate && metal_delegate->BindBufferToTensor(buffer, tensor_index).ok();
}

bool SetCommandEncoder(TfLiteDelegate* delegate, id<MTLComputeCommandEncoder> encoder) {
  auto* metal_delegate = ::tflite::gpu::metal::GetMetalDelegate(delegate);
  if (!metal_delegate) return false;
  metal_delegate->SetCommandEncoder(encoder);
  return true;
}
