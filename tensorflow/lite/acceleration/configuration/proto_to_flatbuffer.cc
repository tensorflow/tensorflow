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
#include "tensorflow/lite/acceleration/configuration/proto_to_flatbuffer.h"

#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
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
    case proto::Delegate::CORE_ML:
      return Delegate_CORE_ML;
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

GPUInferenceUsage ConvertGPUInferenceUsage(
    proto::GPUInferenceUsage preference) {
  switch (preference) {
    case proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    case proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferenceUsage: %d", preference);
  return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
}

GPUInferencePriority ConvertGPUInferencePriority(
    proto::GPUInferencePriority priority) {
  switch (priority) {
    case proto::GPUInferencePriority::GPU_PRIORITY_AUTO:
      return GPUInferencePriority_GPU_PRIORITY_AUTO;
    case proto::GPUInferencePriority::GPU_PRIORITY_MAX_PRECISION:
      return GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION;
    case proto::GPUInferencePriority::GPU_PRIORITY_MIN_LATENCY:
      return GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY;
    case proto::GPUInferencePriority::GPU_PRIORITY_MIN_MEMORY_USAGE:
      return GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferencePriority: %d", priority);
  return GPUInferencePriority_GPU_PRIORITY_AUTO;
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
    const proto::FallbackSettings& settings, FlatBufferBuilder& builder) {
  return CreateFallbackSettings(
      builder, /*allow_automatic_fallback_on_compilation_error=*/
      settings.allow_automatic_fallback_on_compilation_error(),
      /*allow_automatic_fallback_on_execution_error=*/
      settings.allow_automatic_fallback_on_execution_error());
}

Offset<NNAPISettings> ConvertNNAPISettings(const proto::NNAPISettings& settings,
                                           FlatBufferBuilder& builder) {
  return CreateNNAPISettings(
      builder,
      /*accelerator_name=*/builder.CreateString(settings.accelerator_name()),
      /*cache_directory=*/builder.CreateString(settings.cache_directory()),
      /*model_token=*/builder.CreateString(settings.model_token()),
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
      settings.allow_fp16_precision_for_fp32(),
      /*use_burst_computation=*/
      settings.use_burst_computation(),
      /*support_library_handle=*/
      settings.support_library_handle());
}

Offset<GPUSettings> ConvertGPUSettings(const proto::GPUSettings& settings,
                                       FlatBufferBuilder& builder) {
  return CreateGPUSettings(
      builder,
      /*is_precision_loss_allowed=*/settings.is_precision_loss_allowed(),
      /*enable_quantized_inference=*/settings.enable_quantized_inference(),
      ConvertGPUBackend(settings.force_backend()),
      ConvertGPUInferencePriority(settings.inference_priority1()),
      ConvertGPUInferencePriority(settings.inference_priority2()),
      ConvertGPUInferencePriority(settings.inference_priority3()),
      ConvertGPUInferenceUsage(settings.inference_preference()),
      /*cache_directory=*/builder.CreateString(settings.cache_directory()),
      /*model_token=*/builder.CreateString(settings.model_token()));
}

Offset<HexagonSettings> ConvertHexagonSettings(
    const proto::HexagonSettings& settings, FlatBufferBuilder& builder) {
  return CreateHexagonSettings(
      builder,
      /*debug_level=*/settings.debug_level(),
      /*powersave_level=*/settings.powersave_level(),
      /*print_graph_profile=*/settings.print_graph_profile(),
      /*print_graph_debug=*/settings.print_graph_debug());
}

Offset<XNNPackSettings> ConvertXNNPackSettings(
    const proto::XNNPackSettings& settings, FlatBufferBuilder& builder) {
  return CreateXNNPackSettings(
      builder,
      /*num_threads=*/settings.num_threads(),
      /*flags=*/tflite::XNNPackFlags(settings.flags()));
}

Offset<CoreMLSettings> ConvertCoreMLSettings(
    const proto::CoreMLSettings& settings, FlatBufferBuilder& builder) {
  tflite::CoreMLSettings_::EnabledDevices enabled_devices =
      tflite::CoreMLSettings_::EnabledDevices_DEVICES_ALL;
  switch (settings.enabled_devices()) {
    case proto::CoreMLSettings::DEVICES_ALL:
      enabled_devices = tflite::CoreMLSettings_::EnabledDevices_DEVICES_ALL;
      break;
    case proto::CoreMLSettings::DEVICES_WITH_NEURAL_ENGINE:
      enabled_devices =
          tflite::CoreMLSettings_::EnabledDevices_DEVICES_WITH_NEURAL_ENGINE;
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Invalid devices enum: %d",
                      settings.enabled_devices());
  }

  return CreateCoreMLSettings(
      builder, enabled_devices, settings.coreml_version(),
      settings.max_delegated_partitions(), settings.min_nodes_per_partition());
}

Offset<StableDelegateLoaderSettings> ConvertStableDelegateLoaderSettings(
    const proto::StableDelegateLoaderSettings& settings,
    FlatBufferBuilder& builder) {
  return CreateStableDelegateLoaderSettings(
      builder, builder.CreateString(settings.delegate_path()));
}

Offset<CPUSettings> ConvertCPUSettings(const proto::CPUSettings& settings,
                                       FlatBufferBuilder& builder) {
  return CreateCPUSettings(builder,
                           /*num_threads=*/settings.num_threads());
}

Offset<tflite::EdgeTpuDeviceSpec> ConvertEdgeTpuDeviceSpec(
    FlatBufferBuilder& builder, const proto::EdgeTpuDeviceSpec& device_spec) {
  Offset<Vector<Offset<String>>> device_paths_fb = 0;
  if (device_spec.device_paths_size() > 0) {
    std::vector<Offset<String>> device_paths;
    for (const auto& device_path : device_spec.device_paths()) {
      auto device_path_fb = builder.CreateString(device_path);
      device_paths.push_back(device_path_fb);
    }
    device_paths_fb = builder.CreateVector(device_paths);
  }

  return tflite::CreateEdgeTpuDeviceSpec(
      builder,
      static_cast<tflite::EdgeTpuDeviceSpec_::PlatformType>(
          device_spec.platform_type()),
      device_spec.num_chips(), device_paths_fb, device_spec.chip_family());
}

Offset<EdgeTpuSettings> ConvertEdgeTpuSettings(
    const proto::EdgeTpuSettings& settings, FlatBufferBuilder& builder) {
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
              builder,
              static_cast<tflite::EdgeTpuPowerState>(
                  config.inactive_power_state()),
              config.inactive_timeout_us()));
    }

    inactive_power_configs =
        builder.CreateVector<Offset<tflite::EdgeTpuInactivePowerConfig>>(
            inactive_power_configs_std);
  }

  Offset<tflite::EdgeTpuDeviceSpec> edgetpu_device_spec = 0;
  if (settings.has_edgetpu_device_spec()) {
    edgetpu_device_spec =
        ConvertEdgeTpuDeviceSpec(builder, settings.edgetpu_device_spec());
  }

  Offset<String> model_token = 0;
  if (settings.has_model_token()) {
    model_token = builder.CreateString(settings.model_token());
  }

  // First convert to std::vector, then convert to flatbuffer.
  std::vector<int32_t> hardware_cluster_ids_std{
      settings.hardware_cluster_ids().begin(),
      settings.hardware_cluster_ids().end()};
  auto hardware_cluster_ids_fb =
      builder.CreateVector<int32_t>(hardware_cluster_ids_std);

  Offset<String> public_model_id = 0;
  if (settings.has_public_model_id()) {
    public_model_id = builder.CreateString(settings.public_model_id());
  }

  return CreateEdgeTpuSettings(
      builder, ConvertEdgeTpuPowerState(settings.inference_power_state()),
      inactive_power_configs, settings.inference_priority(),
      edgetpu_device_spec, model_token,
      static_cast<tflite::EdgeTpuSettings_::FloatTruncationType>(
          settings.float_truncation_type()),
      static_cast<tflite::EdgeTpuSettings_::QosClass>(settings.qos_class()),
      hardware_cluster_ids_fb, public_model_id);
}

Offset<CoralSettings> ConvertCoralSettings(const proto::CoralSettings& settings,
                                           FlatBufferBuilder& builder) {
  return CreateCoralSettings(
      builder, builder.CreateString(settings.device()),
      static_cast<tflite::CoralSettings_::Performance>(settings.performance()),
      settings.usb_always_dfu(), settings.usb_max_bulk_in_queue_length());
}

Offset<TFLiteSettings> ConvertTfliteSettings(
    const proto::TFLiteSettings& settings, FlatBufferBuilder& builder) {
  return CreateTFLiteSettings(
      builder, ConvertDelegate(settings.delegate()),
      ConvertNNAPISettings(settings.nnapi_settings(), builder),
      ConvertGPUSettings(settings.gpu_settings(), builder),
      ConvertHexagonSettings(settings.hexagon_settings(), builder),
      ConvertXNNPackSettings(settings.xnnpack_settings(), builder),
      ConvertCoreMLSettings(settings.coreml_settings(), builder),
      ConvertCPUSettings(settings.cpu_settings(), builder),
      /*max_delegated_partitions=*/settings.max_delegated_partitions(),
      ConvertEdgeTpuSettings(settings.edgetpu_settings(), builder),
      ConvertCoralSettings(settings.coral_settings(), builder),
      ConvertFallbackSettings(settings.fallback_settings(), builder),
      settings.disable_default_delegates(),
      ConvertStableDelegateLoaderSettings(
          settings.stable_delegate_loader_settings(), builder));
}

Offset<ModelFile> ConvertModelFile(const proto::ModelFile& model_file,
                                   FlatBufferBuilder& builder) {
  return CreateModelFile(builder, builder.CreateString(model_file.filename()),
                         model_file.fd(), model_file.offset(),
                         model_file.length());
}

Offset<BenchmarkStoragePaths> ConvertBenchmarkStoragePaths(
    const proto::BenchmarkStoragePaths& storage_paths,
    FlatBufferBuilder& builder) {
  return CreateBenchmarkStoragePaths(
      builder, builder.CreateString(storage_paths.storage_file_path()),
      builder.CreateString(storage_paths.data_directory_path()));
}

Offset<MinibenchmarkSettings> ConvertMinibenchmarkSettings(
    const proto::MinibenchmarkSettings& settings, FlatBufferBuilder& builder) {
  Offset<Vector<Offset<TFLiteSettings>>> settings_to_test = 0;
  std::vector<Offset<TFLiteSettings>> settings_to_test_vec;
  if (settings.settings_to_test_size() > 0) {
    for (const auto& one : settings.settings_to_test()) {
      settings_to_test_vec.push_back(ConvertTfliteSettings(one, builder));
    }
    settings_to_test =
        builder.CreateVector<Offset<TFLiteSettings>>(settings_to_test_vec);
  }

  return CreateMinibenchmarkSettings(
      builder, settings_to_test,
      ConvertModelFile(settings.model_file(), builder),
      ConvertBenchmarkStoragePaths(settings.storage_paths(), builder));
}

const TFLiteSettings* ConvertFromProto(
    const proto::TFLiteSettings& proto_settings, FlatBufferBuilder* builder) {
  Offset<TFLiteSettings> settings =
      ConvertTfliteSettings(proto_settings, *builder);
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

const ComputeSettings* ConvertFromProto(
    const proto::ComputeSettings& proto_settings, FlatBufferBuilder* builder) {
  auto settings = CreateComputeSettings(
      *builder, ConvertExecutionPreference(proto_settings.preference()),
      ConvertTfliteSettings(proto_settings.tflite_settings(), *builder),
      builder->CreateString(proto_settings.model_namespace_for_statistics()),
      builder->CreateString(proto_settings.model_identifier_for_statistics()),
      ConvertMinibenchmarkSettings(proto_settings.settings_to_test_locally(),
                                   *builder));
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

const MinibenchmarkSettings* ConvertFromProto(
    const proto::MinibenchmarkSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder) {
  auto settings = ConvertMinibenchmarkSettings(proto_settings, *builder);
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

}  // namespace tflite
