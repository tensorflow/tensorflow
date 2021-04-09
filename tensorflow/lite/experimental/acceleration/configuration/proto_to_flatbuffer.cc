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
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;
using ::flatbuffers::String;
using ::flatbuffers::Vector;

ExecutionPreference ConvertExecutionPreference(
    proto::ExecutionPreference preference) {
  switch (preference) {
    case proto::ExecutionPreference::ANY:
      return ExecutionPreference_ANY;
    case proto::ExecutionPreference::LOW_LATENCY:
      return ExecutionPreference_LOW_LATENCY;
    case proto::ExecutionPreference::LOW_POWER:
      return ExecutionPreference_LOW_POWER;
    case proto::ExecutionPreference::FORCE_CPU:
      return ExecutionPreference_FORCE_CPU;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for ExecutionPreference: %d", preference);
  return ExecutionPreference_ANY;
}

Delegate ConvertDelegate(proto::Delegate delegate) {
  switch (delegate) {
    case proto::Delegate::NONE:
      return Delegate_NONE;
    case proto::Delegate::NNAPI:
      return Delegate_NNAPI;
    case proto::Delegate::GPU:
      return Delegate_GPU;
    case proto::Delegate::HEXAGON:
      return Delegate_HEXAGON;
    case proto::Delegate::XNNPACK:
      return Delegate_XNNPACK;
    case proto::Delegate::EDGETPU:
      return Delegate_EDGETPU;
    case proto::Delegate::EDGETPU_CORAL:
      return Delegate_EDGETPU_CORAL;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for Delegate: %d",
                  delegate);
  return Delegate_NONE;
}

NNAPIExecutionPreference ConvertNNAPIExecutionPreference(
    proto::NNAPIExecutionPreference preference) {
  switch (preference) {
    case proto::NNAPIExecutionPreference::UNDEFINED:
      return NNAPIExecutionPreference_UNDEFINED;
    case proto::NNAPIExecutionPreference::NNAPI_LOW_POWER:
      return NNAPIExecutionPreference_NNAPI_LOW_POWER;
    case proto::NNAPIExecutionPreference::NNAPI_FAST_SINGLE_ANSWER:
      return NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER;
    case proto::NNAPIExecutionPreference::NNAPI_SUSTAINED_SPEED:
      return NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPreference: %d",
                  preference);
  return NNAPIExecutionPreference_UNDEFINED;
}

NNAPIExecutionPriority ConvertNNAPIExecutionPriority(
    proto::NNAPIExecutionPriority priority) {
  switch (priority) {
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_LOW:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_LOW;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_MEDIUM:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_HIGH:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPriority: %d", priority);
  return NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED;
}

GPUBackend ConvertGPUBackend(proto::GPUBackend backend) {
  switch (backend) {
    case proto::GPUBackend::UNSET:
      return GPUBackend_UNSET;
    case proto::GPUBackend::OPENCL:
      return GPUBackend_OPENCL;
    case proto::GPUBackend::OPENGL:
      return GPUBackend_OPENGL;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for GPUBackend: %d",
                  backend);
  return GPUBackend_UNSET;
}

EdgeTpuPowerState ConvertEdgeTpuPowerState(proto::EdgeTpuPowerState state) {
  switch (state) {
    case proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE:
      return EdgeTpuPowerState_UNDEFINED_POWERSTATE;
    case proto::EdgeTpuPowerState::TPU_CORE_OFF:
      return EdgeTpuPowerState_TPU_CORE_OFF;
    case proto::EdgeTpuPowerState::READY:
      return EdgeTpuPowerState_READY;
    case proto::EdgeTpuPowerState::ACTIVE_MIN_POWER:
      return EdgeTpuPowerState_ACTIVE_MIN_POWER;
    case proto::EdgeTpuPowerState::ACTIVE_VERY_LOW_POWER:
      return EdgeTpuPowerState_ACTIVE_VERY_LOW_POWER;
    case proto::EdgeTpuPowerState::ACTIVE_LOW_POWER:
      return EdgeTpuPowerState_ACTIVE_LOW_POWER;
    case proto::EdgeTpuPowerState::ACTIVE:
      return EdgeTpuPowerState_ACTIVE;
    case proto::EdgeTpuPowerState::OVER_DRIVE:
      return EdgeTpuPowerState_OVER_DRIVE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for EdgeTpuSettings::PowerState: %d",
                  state);
  return EdgeTpuPowerState_UNDEFINED_POWERSTATE;
}

Offset<FallbackSettings> ConvertFallbackSettings(
    const proto::FallbackSettings& settings, FlatBufferBuilder* builder) {
  return CreateFallbackSettings(
      *builder, /*allow_automatic_fallback_on_compilation_error=*/
      settings.allow_automatic_fallback_on_compilation_error(),
      /*allow_automatic_fallback_on_execution_error=*/
      settings.allow_automatic_fallback_on_execution_error());
}

Offset<NNAPISettings> ConvertNNAPISettings(const proto::NNAPISettings& settings,
                                           FlatBufferBuilder* builder) {
  return CreateNNAPISettings(
      *builder,
      /*accelerator_name=*/builder->CreateString(settings.accelerator_name()),
      /*cache_directory=*/builder->CreateString(settings.cache_directory()),
      /*model_token=*/builder->CreateString(settings.model_token()),
      ConvertNNAPIExecutionPreference(settings.execution_preference()),
      /*no_of_nnapi_instances_to_cache=*/
      settings.no_of_nnapi_instances_to_cache(),
      ConvertFallbackSettings(settings.fallback_settings(), builder),
      /*allow_nnapi_cpu_on_android_10_plus=*/
      settings.allow_nnapi_cpu_on_android_10_plus(),
      ConvertNNAPIExecutionPriority(settings.execution_priority()),
      /*allow_dynamic_dimensions=*/
      settings.allow_dynamic_dimensions(),
      /*allow_fp16_precision_for_fp32=*/
      settings.allow_fp16_precision_for_fp32());
}

Offset<GPUSettings> ConvertGPUSettings(const proto::GPUSettings& settings,
                                       FlatBufferBuilder* builder) {
  return CreateGPUSettings(
      *builder,
      /*is_precision_loss_allowed=*/settings.is_precision_loss_allowed(),
      /*enable_quantized_inference=*/settings.enable_quantized_inference(),
      ConvertGPUBackend(settings.force_backend()));
}

Offset<HexagonSettings> ConvertHexagonSettings(
    const proto::HexagonSettings& settings, FlatBufferBuilder* builder) {
  return CreateHexagonSettings(
      *builder,
      /*debug_level=*/settings.debug_level(),
      /*powersave_level=*/settings.powersave_level(),
      /*print_graph_profile=*/settings.print_graph_profile(),
      /*print_graph_debug=*/settings.print_graph_debug());
}

Offset<XNNPackSettings> ConvertXNNPackSettings(
    const proto::XNNPackSettings& settings, FlatBufferBuilder* builder) {
  return CreateXNNPackSettings(*builder,
                               /*num_threads=*/settings.num_threads());
}

Offset<CPUSettings> ConvertCPUSettings(const proto::CPUSettings& settings,
                                       FlatBufferBuilder* builder) {
  return CreateCPUSettings(*builder,
                           /*num_threads=*/settings.num_threads());
}

Offset<tflite::EdgeTpuDeviceSpec> ConvertEdgeTpuDeviceSpec(
    FlatBufferBuilder* builder, const proto::EdgeTpuDeviceSpec& device_spec) {
  Offset<Vector<Offset<String>>> device_paths_fb = 0;
  if (device_spec.device_paths_size() > 0) {
    std::vector<Offset<String>> device_paths;
    for (const auto& device_path : device_spec.device_paths()) {
      auto device_path_fb = builder->CreateString(device_path);
      device_paths.push_back(device_path_fb);
    }
    device_paths_fb = builder->CreateVector(device_paths);
  }

  return tflite::CreateEdgeTpuDeviceSpec(
      *builder,
      static_cast<tflite::EdgeTpuDeviceSpec_::PlatformType>(
          device_spec.platform_type()),
      device_spec.num_chips(), device_paths_fb, device_spec.chip_family());
}

Offset<EdgeTpuSettings> ConvertEdgeTpuSettings(
    const proto::EdgeTpuSettings& settings, FlatBufferBuilder* builder) {
  Offset<Vector<Offset<tflite::EdgeTpuInactivePowerConfig>>>
      inactive_power_configs = 0;

  // Uses std vector to first construct the list and creates the flatbuffer
  // offset from it later.
  std::vector<Offset<tflite::EdgeTpuInactivePowerConfig>>
      inactive_power_configs_std;
  if (settings.inactive_power_configs_size() > 0) {
    for (const auto& config : settings.inactive_power_configs()) {
      inactive_power_configs_std.push_back(
          tflite::CreateEdgeTpuInactivePowerConfig(
              *builder,
              static_cast<tflite::EdgeTpuPowerState>(
                  config.inactive_power_state()),
              config.inactive_timeout_us()));
    }

    inactive_power_configs =
        builder->CreateVector<Offset<tflite::EdgeTpuInactivePowerConfig>>(
            inactive_power_configs_std);
  }

  Offset<tflite::EdgeTpuDeviceSpec> edgetpu_device_spec = 0;
  if (settings.has_edgetpu_device_spec()) {
    edgetpu_device_spec =
        ConvertEdgeTpuDeviceSpec(builder, settings.edgetpu_device_spec());
  }

  Offset<String> model_token = 0;
  if (settings.has_model_token()) {
    model_token = builder->CreateString(settings.model_token());
  }

  return CreateEdgeTpuSettings(
      *builder, ConvertEdgeTpuPowerState(settings.inference_power_state()),
      inactive_power_configs, settings.inference_priority(),
      edgetpu_device_spec, model_token);
}

Offset<CoralSettings> ConvertCoralSettings(const proto::CoralSettings& settings,
                                           FlatBufferBuilder* builder) {
  return CreateCoralSettings(
      *builder, builder->CreateString(settings.device()),
      static_cast<tflite::CoralSettings_::Performance>(settings.performance()),
      settings.usb_always_dfu(), settings.usb_max_bulk_in_queue_length());
}

Offset<TFLiteSettings> ConvertTfliteSettings(
    const proto::TFLiteSettings& settings, FlatBufferBuilder* builder) {
  return CreateTFLiteSettings(
      *builder, ConvertDelegate(settings.delegate()),
      ConvertNNAPISettings(settings.nnapi_settings(), builder),
      ConvertGPUSettings(settings.gpu_settings(), builder),
      ConvertHexagonSettings(settings.hexagon_settings(), builder),
      ConvertXNNPackSettings(settings.xnnpack_settings(), builder),
      ConvertCPUSettings(settings.cpu_settings(), builder),
      /*max_delegated_partitions=*/settings.max_delegated_partitions(),
      ConvertEdgeTpuSettings(settings.edgetpu_settings(), builder),
      ConvertCoralSettings(settings.coral_settings(), builder),
      ConvertFallbackSettings(settings.fallback_settings(), builder));
}

const ComputeSettings* ConvertFromProto(
    const proto::ComputeSettings& proto_settings, FlatBufferBuilder* builder) {
  auto settings = CreateComputeSettings(
      *builder, ConvertExecutionPreference(proto_settings.preference()),
      ConvertTfliteSettings(proto_settings.tflite_settings(), builder),
      builder->CreateString(proto_settings.model_namespace_for_statistics()),
      builder->CreateString(proto_settings.model_identifier_for_statistics()));
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

}  // namespace tflite
