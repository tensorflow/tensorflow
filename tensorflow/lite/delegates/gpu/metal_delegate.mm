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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

// Multi-thread safe alarm clock for preventing GPU sleeping. It spawns lightweight compute tasks
// until no inference is performing on a device. It's reduces the CPU-to-CPU inference latency.
// The class is used only for kAggressive wait type.
class GpuAlarmClock {
 public:
  explicit GpuAlarmClock(id<MTLCommandQueue> command_queue) {
    auto device = [command_queue device];
    std::lock_guard<std::mutex> lock(alarms_mutex_);
    if (!alarms_) alarms_ = new std::map<id<MTLDevice>, GpuAlarmClockInternal*>();
    auto it = alarms_->find(device);
    if (it == alarms_->end()) {
      internal_ = new GpuAlarmClockInternal(command_queue);
      (*alarms_)[device] = internal_;
    } else {
      internal_ = it->second;
      internal_->total_alarms_++;
    }
  }
  ~GpuAlarmClock() {
    std::lock_guard<std::mutex> lock(alarms_mutex_);
    if (--internal_->total_alarms_ > 0) return;
    Stop();
    delete internal_;
    // Remove the alarm from the container to free-up device handle.
    for (auto it = alarms_->begin(); it != alarms_->end(); ++it) {
      if (it->second == internal_) {
        alarms_->erase(it);
        break;
      }
    }
    if (alarms_->empty()) {
      delete alarms_;
      alarms_ = nullptr;
    }
  }
  void Start() {
    if (started_) return;
    started_ = true;
    internal_->active_alarms_++;
  }
  void Stop() {
    if (!started_) return;
    started_ = false;
    internal_->active_alarms_--;
  }

 private:
  class GpuAlarmClockInternal {
   public:
    id<MTLComputePipelineState> stub_program_;
    id<MTLBuffer> stub_buffer_;
    explicit GpuAlarmClockInternal(id<MTLCommandQueue> command_queue) {
      command_queue_ = command_queue;
      device_ = [command_queue_ device];
      total_alarms_ = 1;
      NSString* error;
      id<MTLComputePipelineState> program;
      CreateComputeProgram(device_,
                           @"kernel void ComputeFunction(device int* output_buffer [[buffer(0)]]) "
                           @"{ output_buffer[0] = 0; }",
                           @"ComputeFunction", nullptr, &program);
      stub_program_ = program;
      stub_buffer_ = [device_ newBufferWithLength:sizeof(int) * 4
                                          options:MTLResourceHazardTrackingModeUntracked];
      alarm_thread_ = std::thread([this]() {
        id<MTLCommandBuffer> prev_command_buffer;
        while (!release_thread_) {
          if (active_alarms_ == total_alarms_) {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:stub_program_];
            [encoder setBuffer:stub_buffer_ offset:0 atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            if (prev_command_buffer != nil) [prev_command_buffer waitUntilScheduled];
            prev_command_buffer = command_buffer;
          } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
      });
    }
    ~GpuAlarmClockInternal() {
      release_thread_ = true;
      alarm_thread_.join();
    }

   private:
    friend class GpuAlarmClock;
    std::atomic<int> active_alarms_;
    std::thread alarm_thread_;
    id<MTLCommandQueue> command_queue_;
    id<MTLDevice> device_;
    volatile bool release_thread_ = false;
    int total_alarms_ = 0;
  };
  static std::map<id<MTLDevice>, GpuAlarmClockInternal*>* alarms_;
  std::mutex alarms_mutex_;
  GpuAlarmClockInternal* internal_;
  bool started_ = false;
};
std::map<id<MTLDevice>, GpuAlarmClock::GpuAlarmClockInternal*>* GpuAlarmClock::alarms_ = nullptr;

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
    command_queue_ = [metal_device_ newCommandQueue];
    if (options_.wait_type == GpuDelegateOptions::WaitType::kAggressive) {
      gpu_alarm_clock_ = std::unique_ptr<GpuAlarmClock>(new GpuAlarmClock(command_queue_));
      NSString* code = @R"(
          kernel void ComputeFunction(device int* output_buffer [[buffer(0)]],
                                      constant int& value [[buffer(1)]]) {
            output_buffer[0] = value;
          }
        )";
      NSString* error;
      id<MTLComputePipelineState> signal_program;
      CreateComputeProgram(metal_device_, code, @"ComputeFunction", nullptr, &signal_program);
      signal_program_ = signal_program;
      signal_buffer_ = [metal_device_ newBufferWithLength:sizeof(int) * 4
                                                  options:MTLResourceStorageModeShared |
                                                          MTLResourceHazardTrackingModeUntracked];
    }
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

  void SetCommandEncoder(
      id<MTLComputeCommandEncoder> encoder,
      std::function<id<MTLComputeCommandEncoder>(bool is_last)> control_encoder) {
    control_encoder_ = control_encoder;
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
    auto find_value = [&](int tensor_index) -> Value<TensorRefFloat32>* {
      for (auto value : values) {
        if (value->tensor.ref == tensor_index) return value;
      }
      return nullptr;
    };
    tensors_.reserve(values.back()->id + 1);
    for (const auto* value : values) {
      if (tensors_.size() <= value->id) tensors_.resize(value->id + 1);
      tensors_[value->id] = {
          value->tensor.shape,  // .shape
          value->tensor.ref,    // .tensor_id
      };
    }

    // Prepare graph inputs.
    //
    // Note that graph.inputs() cannot be used directly, as the notion of graph input has a
    // different meaning in public API and GPU-internal API.
    inputs_.reserve(delegate_params->input_tensors->size);
    for (int i = 0; i < delegate_params->input_tensors->size; ++i) {
      const int tensor_index = delegate_params->input_tensors->data[i];
      auto* tensor = context->tensors + tensor_index;
      if (tensor->allocation_type == TfLiteAllocationType::kTfLiteMmapRo) continue;
      const auto* input = find_value(tensor_index);
      if (!input || tensor->type != TfLiteType::kTfLiteFloat32) {
        return NotFoundError("Input tensor is not found in the graph.");
      }

      inputs_.push_back(input->id);
      tensor->buffer_handle = input->id;
      tensor->delegate = &delegate_;
    }

    // Prepare graph outputs.
    //
    // Note that graph.outputs() cannot be used directly, as the notion of graph output has a
    // different meaning in public API and GPU-internal API.
    outputs_.reserve(delegate_params->output_tensors->size);
    for (int i = 0; i < delegate_params->output_tensors->size; ++i) {
      const int tensor_index = delegate_params->output_tensors->data[i];
      auto* tensor = context->tensors + tensor_index;
      const auto* output = find_value(tensor_index);
      if (!output || tensor->type != TfLiteType::kTfLiteFloat32) {
        return NotFoundError("Output tensor is not found in the graph.");
      }

      outputs_.push_back(output->id);
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
    if (options_.wait_type == GpuDelegateOptions::WaitType::kAggressive) gpu_alarm_clock_->Stop();
    // We need only synchronization so volatile works better than atomic which reads from global
    // memory each time.
    __block volatile bool buffer_completed = false;
    __block id<MTLCommandBuffer> command_buffer;
    __block id<MTLComputeCommandEncoder> encoder = external_command_encoder_;
    if (external_command_encoder_ == nil) {
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
      if (external_command_encoder_ == nil) {
        [encoder endEncoding];
        [command_buffer commit];
        command_buffer = [command_queue_ commandBuffer];
        encoder = [command_buffer computeCommandEncoder];
      }
    }

    [inference_context_ encodeWithEncoder:encoder
                       inputOutputBuffers:bphwc4_buffers_
                             encoderBlock:^(bool isLast) {
                               if (control_encoder_ != nullptr) {
                                 return control_encoder_(isLast);
                               }
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
          // Busy wait. Use local variable. Volatile uses RAM access all the time.
          for (volatile int i = 0; i < 100; i++) {
          }
        }
      } else if (options_.wait_type == GpuDelegateOptions::WaitType::kPassive) {
        // passive wait: this thread sleeps until GPU finishes.
        [command_buffer waitUntilCompleted];
      } else if (options_.wait_type == GpuDelegateOptions::WaitType::kAggressive) {
        command_buffer = [command_queue_ commandBuffer];
        encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:signal_program_];
        [encoder setBuffer:signal_buffer_ offset:0 atIndex:0];
        signal_value_++;
        [encoder setBytes:&signal_value_ length:sizeof(int) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        gpu_alarm_clock_->Start();
        const int* signal_ptr = reinterpret_cast<const int*>([signal_buffer_ contents]);
        while (signal_ptr[0] != signal_value_) {
          // Busy wait. Spinning with local variable to avoid RAM pressure.
          for (volatile int i = 0; i < 100; i++) {
          }
        }
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
  std::function<id<MTLComputeCommandEncoder>(bool is_last)> control_encoder_;
  id<MTLCommandQueue> command_queue_;
  std::unique_ptr<GpuAlarmClock> gpu_alarm_clock_;
  id<MTLComputePipelineState> signal_program_;
  id<MTLBuffer> signal_buffer_;
  int signal_value_ = 0;
};

Delegate* GetMetalDelegate(TfLiteNode* node) {
  return reinterpret_cast<Delegate*>(node->user_data);
}

Delegate* GetMetalDelegate(TfLiteDelegate* delegate) {
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
        context->ReportError(context, "TfLiteGpuDelegate Prepare: %s", status.message().data());
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
        context->ReportError(context, "TfLiteMetalDelegate Invoke: %s", status.message().data());
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
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "Created TensorFlow Lite delegate for Metal.");
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

bool TFLSetCommandEncoder(
    TfLiteDelegate* delegate, id<MTLComputeCommandEncoder> encoder,
    std::function<id<MTLComputeCommandEncoder>(bool is_last)> control_encoder) {
  auto* metal_delegate = ::tflite::gpu::metal::GetMetalDelegate(delegate);
  if (!metal_delegate) return false;
  metal_delegate->SetCommandEncoder(encoder, control_encoder);
  return true;
}
