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

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/metal/buffer_convert.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/kernels/kernel_util.h"
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
      // TODO(impjdi): Properly handle returned status.
      CreateComputeProgram(device_,
                           @"kernel void ComputeFunction(device int* output_buffer [[buffer(0)]]) "
                           @"{ output_buffer[0] = 0; }",
                           @"ComputeFunction", nullptr, &program)
          .IgnoreError();
      stub_program_ = program;
      stub_buffer_ = [device_ newBufferWithLength:sizeof(int) * 4
                                          options:MTLResourceHazardTrackingModeUntracked];
      alarm_thread_ = std::thread([this]() {
        id<MTLCommandBuffer> prev_command_buffer;
        while (!release_thread_) {
          @autoreleasepool {
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
  explicit Delegate(const TFLGpuDelegateOptions* options) {
    if (options) {
      options_ = *options;
    } else {
      options_ = TFLGpuDelegateOptionsDefault();
    }
    metal_device_ = MTLCreateSystemDefaultDevice();
    command_queue_ = [metal_device_ newCommandQueue];
    if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeAggressive) {
      gpu_alarm_clock_ = std::unique_ptr<GpuAlarmClock>(new GpuAlarmClock(command_queue_));
      NSString* code = @R"(
          kernel void ComputeFunction(device int* output_buffer [[buffer(0)]],
                                      constant int& value [[buffer(1)]]) {
            output_buffer[0] = value;
          }
        )";
      NSString* error;
      id<MTLComputePipelineState> signal_program;
      // TODO(impjdi): Properly handle returned status.
      CreateComputeProgram(metal_device_, code, @"ComputeFunction", nullptr, &signal_program)
          .IgnoreError();
      signal_program_ = signal_program;
      signal_buffer_ = [metal_device_ newBufferWithLength:sizeof(int) * 4
                                                  options:MTLResourceStorageModeShared |
                                                          MTLResourceHazardTrackingModeUntracked];
    }
  }

  absl::Status BindBufferToTensor(id<MTLBuffer> buffer, int tensor_index) {
    // The tensor index is expected to be an input or output tensor of the interpreter.
    // For quantized model, the buffer should be linked with their dequantized counterpart.
    if (quant_conversion_map_.find(tensor_index) != quant_conversion_map_.end()) {
      tensor_index = quant_conversion_map_[tensor_index];
      // remove [dequantized tensor ID] -> [quantized tensor ID] mapping, to prevent extra
      // dequant/quant on in/outputs.
      quant_conversion_map_.erase(tensor_index);
    }
    for (auto& input : graph_inputs_) {
      if (input.tensor_id == tensor_index) {
        input_output_buffers_[input.id] = buffer;
        if (bphwc4_buffers_[input.id] != buffer) {
          bphwc_buffers_updated_ = true;
        }
        bphwc4_buffers_[input.id] = buffer;
        input.set_externally = true;
        return absl::OkStatus();
      }
    }
    for (auto& output : graph_outputs_) {
      if (output.tensor_id == tensor_index) {
        input_output_buffers_[output.id] = buffer;
        if (bphwc4_buffers_[output.id] != buffer) {
          bphwc_buffers_updated_ = true;
        }
        bphwc4_buffers_[output.id] = buffer;
        output.set_externally = true;
        return absl::OkStatus();
      }
    }
    return absl::NotFoundError("Couldn't find tensor: " + std::to_string(tensor_index));
  }

  void SetCommandBuffer(id<MTLCommandBuffer> command_buffer) {
    external_command_buffer_ = command_buffer;
  }

  // This directs the runtime to allocate memory for input/output temporary
  // tensors that require dequantization/quantization.
  absl::Status GetRequiredTemporaries(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteIntArray** temporaries_array_ptr) {
    if (quant_conversion_map_.empty()) return absl::OkStatus();

    std::vector<int> temporary_tensor_ids;
    for (auto index : input_tensor_ids_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensor_ids.push_back(index);
      }
    }
    for (auto index : output_tensor_ids_) {
      if (quant_conversion_map_.find(index) != quant_conversion_map_.end()) {
        temporary_tensor_ids.push_back(index);
      }
    }
    *temporaries_array_ptr = TfLiteIntArrayCreate(temporary_tensor_ids.size());
    for (int i = 0; i < temporary_tensor_ids.size(); ++i) {
      (*temporaries_array_ptr)->data[i] = temporary_tensor_ids[i];
    }
    return absl::OkStatus();
  }

  absl::Status Prepare(TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
    // Extract TFLite delegate execution plan from the context and convert it into GraphFloat32.
    GraphFloat32 graph;
    quant_conversion_map_.clear();
    if (options_.enable_quantization) {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, &graph, &quant_conversion_map_));
    } else {
      RETURN_IF_ERROR(BuildFinalModel(context, delegate_params, &graph));
    }

    // TODO(impjdi): Remove code duplication.
    auto values = graph.values();
    auto find_value = [&](int tensor_index) -> Value* {
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
    for (int tensor_index : TfLiteIntArrayView(delegate_params->input_tensors)) {
      auto* tensor = &context->tensors[tensor_index];
      if (IsConstantTensor(tensor)) continue;
      // For quantized models, actual inputs of GPU graph are float tensors, so the 8-bit inputs
      // to the delegate kernel need to be dequantized berfore feeding to the GPU graph.
      if (options_.enable_quantization &&
          quant_conversion_map_.find(tensor_index) != quant_conversion_map_.end()) {
        tensor_index = quant_conversion_map_[tensor_index];
        tensor = &context->tensors[tensor_index];
      }
      const auto* input = find_value(tensor_index);
      if (!input || tensor->type != TfLiteType::kTfLiteFloat32) {
        return absl::NotFoundError("Input tensor is not found in the graph.");
      }

      inputs_.push_back(input->id);
      input_tensor_ids_.push_back(tensor_index);
      tensor->buffer_handle = input->id;
      tensor->delegate = &delegate_;
    }

    // Prepare graph outputs.
    //
    // Note that graph.outputs() cannot be used directly, as the notion of graph output has a
    // different meaning in public API and GPU-internal API.
    for (int tensor_index : TfLiteIntArrayView(delegate_params->output_tensors)) {
      auto* tensor = &context->tensors[tensor_index];
      if (IsConstantTensor(tensor)) continue;
      // For quantized models, actual outputs of GPU graph are float tensors, so they should be
      // quantized to be the 8-bit outputs of delegate.
      if (options_.enable_quantization &&
          quant_conversion_map_.find(tensor_index) != quant_conversion_map_.end()) {
        tensor_index = quant_conversion_map_[tensor_index];
        tensor = &context->tensors[tensor_index];
      }
      const auto* output = find_value(tensor_index);
      if (!output || tensor->type != TfLiteType::kTfLiteFloat32) {
        return absl::NotFoundError("Output tensor is not found in the graph.");
      }

      outputs_.push_back(output->id);
      output_tensor_ids_.push_back(tensor_index);
      tensor->buffer_handle = output->id;
      tensor->delegate = &delegate_;
    }

    std::string device_name = std::string([[metal_device_ name] UTF8String]);
    GpuInfo gpu_info;
    GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
    size_t storage_type_size;
    CalculationsPrecision precision;
    if (options_.allow_precision_loss) {
      storage_type_size = sizeof(HalfBits);
      if (gpu_info.IsRoundToNearestSupported()) {
        precision = CalculationsPrecision::F16;
      } else {
        precision = CalculationsPrecision::F32_F16;
      }
    } else {
      storage_type_size = sizeof(float);
      precision = CalculationsPrecision::F32;
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
            return absl::InternalError("Error initialization of input buffer converter");
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
            return absl::InternalError("Error initialization of output buffer converter");
          }
        }
      } else {
        bphwc4_buffers_[output] = buffer;
      }
    }
    bphwc_buffers_updated_ = true;

    InferenceContext::CreateInferenceInfo create_info;
    create_info.precision = precision;
    create_info.storage_type = TensorStorageType::BUFFER;
    RETURN_IF_ERROR(
        inference_context_.InitFromGraphWithTransforms(create_info, &graph, metal_device_));
    return absl::OkStatus();
  }

  absl::Status Invoke(TfLiteContext* context) {
    if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeAggressive)
      gpu_alarm_clock_->Stop();
    // We need only synchronization so volatile works better than atomic which reads from global
    // memory each time.
    __block volatile bool buffer_completed = false;
    id<MTLCommandBuffer> command_buffer = external_command_buffer_;
    if (external_command_buffer_ == nil) {
      command_buffer = [command_queue_ commandBuffer];
    }
    const bool flush = external_command_buffer_ == nil &&
        (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeActive ||
         options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeAggressive);
    const int flush_period = 8;

    const bool is_quantized_model = !quant_conversion_map_.empty();
    if (is_quantized_model) {
      RETURN_IF_ERROR(DequantizeInputs(context, input_tensor_ids_, quant_conversion_map_));
    }

    // CPU HWC input data conversion to PHWC4 and fill the GPU buffer
    for (const auto& input : graph_inputs_) {
      if (input.set_externally) continue;
      // A user provides data on CPU memory for this buffer - need to copy to MTLBuffer

      TfLiteTensor* tensor = &context->tensors[input.tensor_id];
      void* gpu_ptr = [input_output_buffers_[input.id] contents];
      std::memcpy(gpu_ptr, tensor->data.f, input.shape.DimensionsProduct() * sizeof(float));
      if (input_output_buffers_[input.id] == bphwc4_buffers_[input.id]) continue;
      id<MTLComputeCommandEncoder> input_encoder = [command_buffer computeCommandEncoder];
      [converter_to_BPHWC4_ convertWithEncoder:input_encoder
                                         shape:input.shape
                                  sourceBuffer:input_output_buffers_[input.id]
                               convertedBuffer:bphwc4_buffers_[input.id]];
      [input_encoder endEncoding];
    }

    if (bphwc_buffers_updated_) {
      inference_context_.UpdatePreallocatedTensors(bphwc4_buffers_);
      bphwc_buffers_updated_ = false;
    }

    @autoreleasepool {
      if (flush) {
        [command_buffer commit];
        inference_context_.EncodeWithCommandQueue(command_queue_, flush_period);
        command_buffer = [command_queue_ commandBuffer];
      } else {
        inference_context_.EncodeWithCommandBuffer(command_buffer);
      }
    }

    for (const auto& output : graph_outputs_) {
      if (output.set_externally) continue;
      if (bphwc4_buffers_[output.id] == input_output_buffers_[output.id]) continue;
      id<MTLComputeCommandEncoder> output_encoder = [command_buffer computeCommandEncoder];
      [converter_from_BPHWC4_ convertWithEncoder:output_encoder
                                           shape:output.shape
                                    sourceBuffer:bphwc4_buffers_[output.id]
                                 convertedBuffer:input_output_buffers_[output.id]];
      [output_encoder endEncoding];
    }

    if (external_command_buffer_ == nil) {
      if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeActive) {
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          buffer_completed = true;
        }];
      }
      [command_buffer commit];
      if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeActive) {
        while (!buffer_completed) {
          // Busy wait. Use local variable. Volatile uses RAM access all the time.
          for (volatile int i = 0; i < 100; i++) {
          }
        }
      } else if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive) {
        // passive wait: this thread sleeps until GPU finishes.
        [command_buffer waitUntilCompleted];
      } else if (options_.wait_type == TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypeAggressive) {
        id<MTLCommandBuffer> signal_cb = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> signal_encoder = [signal_cb computeCommandEncoder];
        [signal_encoder setComputePipelineState:signal_program_];
        [signal_encoder setBuffer:signal_buffer_ offset:0 atIndex:0];
        signal_value_++;
        [signal_encoder setBytes:&signal_value_ length:sizeof(int) atIndex:1];
        [signal_encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [signal_encoder endEncoding];
        [signal_cb commit];
        gpu_alarm_clock_->Start();
        const int* signal_ptr = reinterpret_cast<const int*>([signal_buffer_ contents]);
        while (signal_ptr[0] != signal_value_) {
          // Busy wait. Spinning with local variable to avoid RAM pressure.
          for (volatile int i = 0; i < 100; i++) {
          }
        }
      }
    } else {
      // External command buffer must be set before every invoke call.
      external_command_buffer_ = nil;
      // External command buffer is assigned so all output buffers are controlled by a user.
      for (const auto& output : graph_outputs_) {
        if (!output.set_externally) {
          return absl::InternalError(
              "External command encoder is used, but not all output buffers are bound.");
        }
      }
      return absl::OkStatus();
    }

    // Retrieve data from GPU and convert from PHWC4 to HWC.
    for (const auto& output : graph_outputs_) {
      if (output.set_externally) continue;
      // A user retrieves data on CPU memory for this buffer - need to copy from MTLBuffer.
      TfLiteTensor* tensor = context->tensors + output.tensor_id;
      const void* gpu_ptr = [input_output_buffers_[output.id] contents];
      std::memcpy(tensor->data.f, gpu_ptr, output.shape.DimensionsProduct() * sizeof(float));
    }
    if (is_quantized_model) {
      RETURN_IF_ERROR(QuantizeOutputs(context, output_tensor_ids_, quant_conversion_map_));
    }
    return absl::OkStatus();
  }

  const TFLGpuDelegateOptions options() const { return options_; }

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

  TFLGpuDelegateOptions options_;

  id<MTLDevice> metal_device_;

  std::vector<ValueRef> tensors_;  // indexed by ValueId
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
  std::vector<int64_t> input_tensor_ids_;
  std::vector<int64_t> output_tensor_ids_;
  // Whenever quantized inference is enabled, this maps the tensor index of each
  // originally quantized (8-bit) tensor to its float version added in
  // model_builder - and vice versa.
  absl::flat_hash_map<int, int> quant_conversion_map_;

  InferenceContext inference_context_;
  // input and output buffers are passed into Metal inference engine
  std::map<::tflite::gpu::ValueId, id<MTLBuffer>> input_output_buffers_;
  std::map<::tflite::gpu::ValueId, id<MTLBuffer>> bphwc4_buffers_;
  bool bphwc_buffers_updated_ = true;
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

  id<MTLCommandBuffer> external_command_buffer_ = nil;
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
        TF_LITE_KERNEL_LOG(context, "TfLiteMetalDelegate Prepare: %s",
                           std::string(status.message()).c_str());
        return nullptr;
      },
      // .free
      [](TfLiteContext*, void* buffer) -> void {},
      // .prepare
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (!node->user_data) {
          return kTfLiteError;
        }

        auto* gpu_delegate_kernel = GetMetalDelegate(node);
        const auto status =
            gpu_delegate_kernel->GetRequiredTemporaries(context, node, &node->temporaries);
        if (!status.ok()) {
          TF_LITE_KERNEL_LOG(context, "TfLiteMetalDelegate Prepare: %s",
                             std::string(status.message()).c_str());
          return kTfLiteError;
        }
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      // .invoke
      [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        const auto status = GetMetalDelegate(node)->Invoke(context);
        if (status.ok()) return kTfLiteOk;
        TF_LITE_KERNEL_LOG(context, "TfLiteMetalDelegate Invoke: %s",
                           std::string(status.message()).c_str());
        return kTfLiteError;
      },
      nullptr,                // .profiling_string
      0,                      // .builtin_code
      "TfLiteMetalDelegate",  // .custom_name
      1,                      // .version
  };
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, GetMetalDelegate(delegate)->options().enable_quantization);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(context, kRegistration,
                                                                     ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace tflite

TfLiteDelegate* TFLGpuDelegateCreate(const TFLGpuDelegateOptions* options) {
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "Created TensorFlow Lite delegate for Metal.");
  auto* metal_delegate = new ::tflite::gpu::metal::Delegate(options);
  return metal_delegate ? metal_delegate->tflite_delegate() : nullptr;
}

void TFLGpuDelegateDelete(TfLiteDelegate* delegate) {
  delete ::tflite::gpu::metal::GetMetalDelegate(delegate);
}

bool TFLGpuDelegateBindMetalBufferToTensor(TfLiteDelegate* delegate, int tensor_index,
                                           id<MTLBuffer> buffer) {
  auto* metal_delegate = ::tflite::gpu::metal::GetMetalDelegate(delegate);
  return metal_delegate && metal_delegate->BindBufferToTensor(buffer, tensor_index).ok();
}

// Note: This function is not exposed in `metal_delegate.h`, but it's exposed in
// `metal_delegate_internal.h`.
bool TFLGpuDelegateSetCommandBuffer(TfLiteDelegate* delegate,
                                    id<MTLCommandBuffer> command_buffer) {
  auto* metal_delegate = ::tflite::gpu::metal::GetMetalDelegate(delegate);
  if (!metal_delegate) return false;
  metal_delegate->SetCommandBuffer(command_buffer);
  return true;
}

TFLGpuDelegateOptions TFLGpuDelegateOptionsDefault() {
  TFLGpuDelegateOptions options = {
      .allow_precision_loss = false,
      .wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive,
      .enable_quantization = true,
  };
  return options;
}
