/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.h"

#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
proto::ExecutionPreference ConvertExecutionPreference(
    ExecutionPreference preference) {
  switch (preference) {
    case ExecutionPreference_ANY:
      return proto::ExecutionPreference::ANY;
    case ExecutionPreference_LOW_LATENCY:
      return proto::ExecutionPreference::LOW_LATENCY;
    case ExecutionPreference_LOW_POWER:
      return proto::ExecutionPreference::LOW_POWER;
    case ExecutionPreference_FORCE_CPU:
      return proto::ExecutionPreference::FORCE_CPU;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for ExecutionPreference: %d", preference);
  return proto::ExecutionPreference::ANY;
}

proto::Delegate ConvertDelegate(Delegate delegate) {
  switch (delegate) {
    case Delegate_NONE:
      return proto::Delegate::NONE;
    case Delegate_NNAPI:
      return proto::Delegate::NNAPI;
    case Delegate_GPU:
      return proto::Delegate::GPU;
    case Delegate_HEXAGON:
      return proto::Delegate::HEXAGON;
    case Delegate_XNNPACK:
      return proto::Delegate::XNNPACK;
    case Delegate_EDGETPU:
      return proto::Delegate::EDGETPU;
    case Delegate_EDGETPU_CORAL:
      return proto::Delegate::EDGETPU_CORAL;
    case Delegate_CORE_ML:
      return proto::Delegate::CORE_ML;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for Delegate: %d",
                  delegate);
  return proto::Delegate::NONE;
}

proto::NNAPIExecutionPreference ConvertNNAPIExecutionPreference(
    NNAPIExecutionPreference preference) {
  switch (preference) {
    case NNAPIExecutionPreference_UNDEFINED:
      return proto::NNAPIExecutionPreference::UNDEFINED;
    case NNAPIExecutionPreference_NNAPI_LOW_POWER:
      return proto::NNAPIExecutionPreference::NNAPI_LOW_POWER;
    case NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER:
      return proto::NNAPIExecutionPreference::NNAPI_FAST_SINGLE_ANSWER;
    case NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED:
      return proto::NNAPIExecutionPreference::NNAPI_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPreference: %d",
                  preference);
  return proto::NNAPIExecutionPreference::UNDEFINED;
}

proto::NNAPIExecutionPriority ConvertNNAPIExecutionPriority(
    NNAPIExecutionPriority priority) {
  switch (priority) {
    case NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_LOW:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_LOW;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_MEDIUM;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_HIGH;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPriority: %d", priority);
  return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED;
}

proto::GPUBackend ConvertGPUBackend(GPUBackend backend) {
  switch (backend) {
    case GPUBackend_UNSET:
      return proto::GPUBackend::UNSET;
    case GPUBackend_OPENCL:
      return proto::GPUBackend::OPENCL;
    case GPUBackend_OPENGL:
      return proto::GPUBackend::OPENGL;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for GPUBackend: %d",
                  backend);
  return proto::GPUBackend::UNSET;
}

proto::GPUInferenceUsage ConvertGPUInferenceUsage(
    GPUInferenceUsage preference) {
  switch (preference) {
    case GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return proto::GPUInferenceUsage::
          GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    case GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferenceUsage: %d", preference);
  return proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
}

proto::GPUInferencePriority ConvertGPUInferencePriority(
    GPUInferencePriority priority) {
  switch (priority) {
    case GPUInferencePriority_GPU_PRIORITY_AUTO:
      return proto::GPUInferencePriority::GPU_PRIORITY_AUTO;
    case GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION:
      return proto::GPUInferencePriority::GPU_PRIORITY_MAX_PRECISION;
    case GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY:
      return proto::GPUInferencePriority::GPU_PRIORITY_MIN_LATENCY;
    case GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE:
      return proto::GPUInferencePriority::GPU_PRIORITY_MIN_MEMORY_USAGE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferencePriority: %d", priority);
  return proto::GPUInferencePriority::GPU_PRIORITY_AUTO;
}

proto::EdgeTpuPowerState ConvertEdgeTpuPowerState(EdgeTpuPowerState state) {
  switch (state) {
    case EdgeTpuPowerState_UNDEFINED_POWERSTATE:
      return proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE;
    case EdgeTpuPowerState_TPU_CORE_OFF:
      return proto::EdgeTpuPowerState::TPU_CORE_OFF;
    case EdgeTpuPowerState_READY:
      return proto::EdgeTpuPowerState::READY;
    case EdgeTpuPowerState_ACTIVE_MIN_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_MIN_POWER;
    case EdgeTpuPowerState_ACTIVE_VERY_LOW_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_VERY_LOW_POWER;
    case EdgeTpuPowerState_ACTIVE_LOW_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_LOW_POWER;
    case EdgeTpuPowerState_ACTIVE:
      return proto::EdgeTpuPowerState::ACTIVE;
    case EdgeTpuPowerState_OVER_DRIVE:
      return proto::EdgeTpuPowerState::OVER_DRIVE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for EdgeTpuSettings::PowerState: %d",
                  state);
  return proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE;
}

proto::FallbackSettings ConvertFallbackSettings(
    const FallbackSettings& settings) {
  proto::FallbackSettings proto_settings;
  proto_settings.set_allow_automatic_fallback_on_compilation_error(
      settings.allow_automatic_fallback_on_compilation_error());
  proto_settings.set_allow_automatic_fallback_on_execution_error(
      settings.allow_automatic_fallback_on_execution_error());
  return proto_settings;
}

proto::NNAPISettings ConvertNNAPISettings(const NNAPISettings& settings) {
  proto::NNAPISettings proto_settings;
  if (settings.accelerator_name() != nullptr) {
    proto_settings.set_accelerator_name(settings.accelerator_name()->str());
  }
  if (settings.cache_directory() != nullptr) {
    proto_settings.set_cache_directory(settings.cache_directory()->str());
  }
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  proto_settings.set_execution_preference(
      ConvertNNAPIExecutionPreference(settings.execution_preference()));
  proto_settings.set_no_of_nnapi_instances_to_cache(
      settings.no_of_nnapi_instances_to_cache());
  if (settings.fallback_settings() != nullptr) {
    *(proto_settings.mutable_fallback_settings()) =
        ConvertFallbackSettings(*settings.fallback_settings());
  }
  proto_settings.set_allow_nnapi_cpu_on_android_10_plus(
      settings.allow_nnapi_cpu_on_android_10_plus());
  proto_settings.set_execution_priority(
      ConvertNNAPIExecutionPriority(settings.execution_priority()));
  proto_settings.set_allow_dynamic_dimensions(
      settings.allow_dynamic_dimensions());
  proto_settings.set_allow_fp16_precision_for_fp32(
      settings.allow_fp16_precision_for_fp32());
  proto_settings.set_use_burst_computation(settings.use_burst_computation());
  proto_settings.set_support_library_handle(settings.support_library_handle());

  return proto_settings;
}

proto::GPUSettings ConvertGPUSettings(const GPUSettings& settings) {
  proto::GPUSettings proto_settings;
  proto_settings.set_is_precision_loss_allowed(
      settings.is_precision_loss_allowed());
  proto_settings.set_enable_quantized_inference(
      settings.enable_quantized_inference());
  proto_settings.set_force_backend(ConvertGPUBackend(settings.force_backend()));
  proto_settings.set_inference_priority1(
      ConvertGPUInferencePriority(settings.inference_priority1()));
  proto_settings.set_inference_priority2(
      ConvertGPUInferencePriority(settings.inference_priority2()));
  proto_settings.set_inference_priority3(
      ConvertGPUInferencePriority(settings.inference_priority3()));
  proto_settings.set_inference_preference(
      ConvertGPUInferenceUsage(settings.inference_preference()));
  if (settings.cache_directory() != nullptr) {
    proto_settings.set_cache_directory(settings.cache_directory()->str());
  }
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  return proto_settings;
}

proto::HexagonSettings ConvertHexagonSettings(const HexagonSettings& settings) {
  proto::HexagonSettings proto_settings;
  proto_settings.set_debug_level(settings.debug_level());
  proto_settings.set_powersave_level(settings.powersave_level());
  proto_settings.set_print_graph_profile(settings.print_graph_profile());
  proto_settings.set_print_graph_debug(settings.print_graph_debug());
  return proto_settings;
}

proto::XNNPackSettings ConvertXNNPackSettings(const XNNPackSettings& settings) {
  proto::XNNPackSettings proto_settings;
  proto_settings.set_num_threads(settings.num_threads());
  return proto_settings;
}

proto::CoreMLSettings ConvertCoreMLSettings(const CoreMLSettings& settings) {
  proto::CoreMLSettings proto_settings;
  switch (settings.enabled_devices()) {
    case CoreMLSettings_::EnabledDevices_DEVICES_ALL:
      proto_settings.set_enabled_devices(proto::CoreMLSettings::DEVICES_ALL);
      break;
    case CoreMLSettings_::EnabledDevices_DEVICES_WITH_NEURAL_ENGINE:
      proto_settings.set_enabled_devices(
          proto::CoreMLSettings::DEVICES_WITH_NEURAL_ENGINE);
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Invalid devices enum: %d",
                      settings.enabled_devices());
  }
  proto_settings.set_coreml_version(settings.coreml_version());
  proto_settings.set_max_delegated_partitions(
      settings.max_delegated_partitions());
  proto_settings.set_min_nodes_per_partition(
      settings.min_nodes_per_partition());
  return proto_settings;
}

proto::CPUSettings ConvertCPUSettings(const CPUSettings& settings) {
  proto::CPUSettings proto_settings;
  proto_settings.set_num_threads(settings.num_threads());
  return proto_settings;
}

proto::EdgeTpuDeviceSpec ConvertEdgeTpuDeviceSpec(
    const EdgeTpuDeviceSpec& device_spec) {
  proto::EdgeTpuDeviceSpec proto_settings;

  if (device_spec.device_paths() != nullptr) {
    for (int i = 0; i < device_spec.device_paths()->size(); ++i) {
      auto device_path = device_spec.device_paths()->Get(i);
      proto_settings.add_device_paths(device_path->str());
    }
  }

  proto_settings.set_platform_type(
      static_cast<proto::EdgeTpuDeviceSpec::PlatformType>(
          device_spec.platform_type()));
  proto_settings.set_num_chips(device_spec.num_chips());
  proto_settings.set_chip_family(device_spec.chip_family());

  return proto_settings;
}

proto::EdgeTpuSettings ConvertEdgeTpuSettings(const EdgeTpuSettings& settings) {
  proto::EdgeTpuSettings proto_settings;
  proto_settings.set_inference_power_state(
      ConvertEdgeTpuPowerState(settings.inference_power_state()));
  proto_settings.set_inference_priority(settings.inference_priority());
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  if (settings.edgetpu_device_spec() != nullptr) {
    *(proto_settings.mutable_edgetpu_device_spec()) =
        ConvertEdgeTpuDeviceSpec(*settings.edgetpu_device_spec());
  }
  proto_settings.set_float_truncation_type(
      static_cast<proto::EdgeTpuSettings::FloatTruncationType>(
          settings.float_truncation_type()));

  auto inactive_powre_configs = settings.inactive_power_configs();
  if (inactive_powre_configs != nullptr) {
    for (int i = 0; i < inactive_powre_configs->size(); ++i) {
      auto config = inactive_powre_configs->Get(i);
      auto proto_config = proto_settings.add_inactive_power_configs();
      proto_config->set_inactive_power_state(
          ConvertEdgeTpuPowerState(config->inactive_power_state()));
      proto_config->set_inactive_timeout_us(config->inactive_timeout_us());
    }
  }

  return proto_settings;
}

proto::CoralSettings ConvertCoralSettings(const CoralSettings& settings) {
  proto::CoralSettings proto_settings;
  if (settings.device() != nullptr) {
    proto_settings.set_device(settings.device()->str());
  }
  proto_settings.set_performance(
      static_cast<proto::CoralSettings::Performance>(settings.performance()));
  proto_settings.set_usb_always_dfu(settings.usb_always_dfu());
  proto_settings.set_usb_max_bulk_in_queue_length(
      settings.usb_max_bulk_in_queue_length());
  return proto_settings;
}

proto::TFLiteSettings ConvertTfliteSettings(const TFLiteSettings& settings) {
  proto::TFLiteSettings proto_settings;
  proto_settings.set_delegate(ConvertDelegate(settings.delegate()));
  if (settings.nnapi_settings() != nullptr) {
    *(proto_settings.mutable_nnapi_settings()) =
        ConvertNNAPISettings(*settings.nnapi_settings());
  }
  if (settings.gpu_settings() != nullptr) {
    *(proto_settings.mutable_gpu_settings()) =
        ConvertGPUSettings(*settings.gpu_settings());
  }
  if (settings.hexagon_settings() != nullptr) {
    *(proto_settings.mutable_hexagon_settings()) =
        ConvertHexagonSettings(*settings.hexagon_settings());
  }

  if (settings.xnnpack_settings() != nullptr) {
    *(proto_settings.mutable_xnnpack_settings()) =
        ConvertXNNPackSettings(*settings.xnnpack_settings());
  }

  if (settings.coreml_settings() != nullptr) {
    *(proto_settings.mutable_coreml_settings()) =
        ConvertCoreMLSettings(*settings.coreml_settings());
  }

  if (settings.cpu_settings() != nullptr) {
    *(proto_settings.mutable_cpu_settings()) =
        ConvertCPUSettings(*settings.cpu_settings());
  }

  proto_settings.set_max_delegated_partitions(
      settings.max_delegated_partitions());
  if (settings.edgetpu_settings() != nullptr) {
    *(proto_settings.mutable_edgetpu_settings()) =
        ConvertEdgeTpuSettings(*settings.edgetpu_settings());
  }
  if (settings.coral_settings() != nullptr) {
    *(proto_settings.mutable_coral_settings()) =
        ConvertCoralSettings(*settings.coral_settings());
  }
  if (settings.fallback_settings() != nullptr) {
    *(proto_settings.mutable_fallback_settings()) =
        ConvertFallbackSettings(*settings.fallback_settings());
  }
  return proto_settings;
}

proto::ModelFile ConvertModelFile(const ModelFile& model_file) {
  proto::ModelFile proto_settings;
  if (model_file.filename() != nullptr) {
    proto_settings.set_filename(model_file.filename()->str());
  }
  proto_settings.set_fd(model_file.fd());
  proto_settings.set_offset(model_file.offset());
  proto_settings.set_length(model_file.length());
  return proto_settings;
}

proto::BenchmarkStoragePaths ConvertBenchmarkStoragePaths(
    const BenchmarkStoragePaths& storage_paths) {
  proto::BenchmarkStoragePaths proto_settings;
  if (storage_paths.storage_file_path() != nullptr) {
    proto_settings.set_storage_file_path(
        storage_paths.storage_file_path()->str());
  }
  if (storage_paths.data_directory_path() != nullptr) {
    proto_settings.set_data_directory_path(
        storage_paths.data_directory_path()->str());
  }
  return proto_settings;
}

proto::MinibenchmarkSettings ConvertMinibenchmarkSettings(
    const MinibenchmarkSettings& settings) {
  proto::MinibenchmarkSettings proto_settings;
  if (settings.settings_to_test() != nullptr &&
      settings.settings_to_test()->size() > 0) {
    for (int i = 0; i < settings.settings_to_test()->size(); ++i) {
      auto tflite_setting = settings.settings_to_test()->Get(i);
      auto proto_tflite_setting = proto_settings.add_settings_to_test();
      *proto_tflite_setting = ConvertTfliteSettings(*tflite_setting);
    }
  }
  if (settings.model_file() != nullptr) {
    *(proto_settings.mutable_model_file()) =
        ConvertModelFile(*settings.model_file());
  }
  if (settings.storage_paths() != nullptr) {
    *(proto_settings.mutable_storage_paths()) =
        ConvertBenchmarkStoragePaths(*settings.storage_paths());
  }
  return proto_settings;
}

proto::BenchmarkEventType ConvertBenchmarkEventType(BenchmarkEventType type) {
  switch (type) {
    case BenchmarkEventType_UNDEFINED_BENCHMARK_EVENT_TYPE:
      return proto::BenchmarkEventType::UNDEFINED_BENCHMARK_EVENT_TYPE;
    case BenchmarkEventType_START:
      return proto::BenchmarkEventType::START;
    case BenchmarkEventType_END:
      return proto::BenchmarkEventType::END;
    case BenchmarkEventType_ERROR:
      return proto::BenchmarkEventType::ERROR;
    case BenchmarkEventType_LOGGED:
      return proto::BenchmarkEventType::LOGGED;
    case BenchmarkEventType_RECOVERED_ERROR:
      return proto::BenchmarkEventType::RECOVERED_ERROR;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for BenchmarkEventType: %d", type);
  return proto::BenchmarkEventType::UNDEFINED_BENCHMARK_EVENT_TYPE;
}

proto::BenchmarkMetric ConvertBenchmarkMetric(const BenchmarkMetric& metric) {
  proto::BenchmarkMetric proto_metric;
  if (metric.name() != nullptr) {
    proto_metric.set_name(metric.name()->str());
  }
  auto values = metric.values();
  if (values != nullptr) {
    for (int i = 0; i < values->size(); ++i) {
      proto_metric.add_values(values->Get(i));
    }
  }
  return proto_metric;
}

proto::BenchmarkResult ConvertBenchmarkResult(const BenchmarkResult& result) {
  proto::BenchmarkResult proto_result;
  auto initialization_time_us = result.initialization_time_us();
  if (initialization_time_us != nullptr) {
    for (int i = 0; i < initialization_time_us->size(); ++i) {
      proto_result.add_initialization_time_us(initialization_time_us->Get(i));
    }
  }
  auto inference_time_us = result.inference_time_us();
  if (inference_time_us != nullptr) {
    for (int i = 0; i < inference_time_us->size(); ++i) {
      proto_result.add_inference_time_us(inference_time_us->Get(i));
    }
  }
  proto_result.set_max_memory_kb(result.max_memory_kb());
  proto_result.set_ok(result.ok());
  auto metrics = result.metrics();
  if (metrics != nullptr) {
    for (int i = 0; i < metrics->size(); ++i) {
      *proto_result.add_metrics() = ConvertBenchmarkMetric(*metrics->Get(i));
    }
  }
  return proto_result;
}

proto::BenchmarkStage ConvertBenchmarkStage(BenchmarkStage stage) {
  switch (stage) {
    case BenchmarkStage_UNKNOWN:
      return proto::BenchmarkStage::UNKNOWN;
    case BenchmarkStage_INITIALIZATION:
      return proto::BenchmarkStage::INITIALIZATION;
    case BenchmarkStage_INFERENCE:
      return proto::BenchmarkStage::INFERENCE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for BenchmarkStage: %d",
                  stage);
  return proto::BenchmarkStage::UNKNOWN;
}

proto::ErrorCode ConvertBenchmarkErrorCode(const ErrorCode& code) {
  proto::ErrorCode proto_code;
  proto_code.set_source(ConvertDelegate(code.source()));
  proto_code.set_tflite_error(code.tflite_error());
  proto_code.set_underlying_api_error(code.underlying_api_error());
  return proto_code;
}

proto::BenchmarkError ConvertBenchmarkError(const BenchmarkError& error) {
  proto::BenchmarkError proto_error;
  proto_error.set_stage(ConvertBenchmarkStage(error.stage()));
  proto_error.set_exit_code(error.exit_code());
  proto_error.set_signal(error.signal());
  auto error_codes = error.error_code();
  if (error_codes != nullptr) {
    for (int i = 0; i < error_codes->size(); ++i) {
      *proto_error.add_error_code() =
          ConvertBenchmarkErrorCode(*error_codes->Get(i));
    }
  }
  proto_error.set_mini_benchmark_error_code(error.mini_benchmark_error_code());
  return proto_error;
}

proto::BenchmarkEvent ConvertBenchmarkEvent(const BenchmarkEvent& event) {
  proto::BenchmarkEvent proto_event;
  if (event.tflite_settings() != nullptr) {
    *proto_event.mutable_tflite_settings() =
        ConvertTfliteSettings(*event.tflite_settings());
  }
  proto_event.set_event_type(ConvertBenchmarkEventType(event.event_type()));
  if (event.result() != nullptr) {
    *proto_event.mutable_result() = ConvertBenchmarkResult(*event.result());
  }
  if (event.error() != nullptr) {
    *proto_event.mutable_error() = ConvertBenchmarkError(*event.error());
  }
  proto_event.set_boottime_us(event.boottime_us());
  proto_event.set_wallclock_us(event.wallclock_us());
  return proto_event;
}

proto::BestAccelerationDecision ConvertBestAccelerationDecision(
    const BestAccelerationDecision& decision) {
  proto::BestAccelerationDecision proto_decision;
  proto_decision.set_number_of_source_events(
      decision.number_of_source_events());
  if (decision.min_latency_event() != nullptr) {
    *proto_decision.mutable_min_latency_event() =
        ConvertBenchmarkEvent(*decision.min_latency_event());
  }
  proto_decision.set_min_inference_time_us(decision.min_inference_time_us());
  return proto_decision;
}

proto::BenchmarkInitializationFailure ConvertBenchmarkInitializationFailure(
    const BenchmarkInitializationFailure& init_failure) {
  proto::BenchmarkInitializationFailure proto_init_failure;
  proto_init_failure.set_initialization_status(
      init_failure.initialization_status());
  return proto_init_failure;
}

}  // namespace

proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettings& settings, bool skip_mini_benchmark_settings) {
  proto::ComputeSettings proto_settings;

  proto_settings.set_preference(
      ConvertExecutionPreference(settings.preference()));
  if (settings.tflite_settings() != nullptr) {
    *(proto_settings.mutable_tflite_settings()) =
        ConvertTfliteSettings(*settings.tflite_settings());
  }
  if (settings.model_namespace_for_statistics() != nullptr) {
    proto_settings.set_model_namespace_for_statistics(
        settings.model_namespace_for_statistics()->str());
  }
  if (settings.model_identifier_for_statistics() != nullptr) {
    proto_settings.set_model_identifier_for_statistics(
        settings.model_identifier_for_statistics()->str());
  }
  if (!skip_mini_benchmark_settings &&
      settings.settings_to_test_locally() != nullptr) {
    *(proto_settings.mutable_settings_to_test_locally()) =
        ConvertMinibenchmarkSettings(*settings.settings_to_test_locally());
  }

  return proto_settings;
}

proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettingsT& settings, bool skip_mini_benchmark_settings) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(ComputeSettings::Pack(fbb, &settings));
  auto settings_fbb =
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer());
  return ConvertFromFlatbuffer(*settings_fbb, skip_mini_benchmark_settings);
}

proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEvent& event) {
  proto::MiniBenchmarkEvent proto_event;
  proto_event.set_is_log_flushing_event(event.is_log_flushing_event());
  if (event.best_acceleration_decision() != nullptr) {
    *proto_event.mutable_best_acceleration_decision() =
        ConvertBestAccelerationDecision(*event.best_acceleration_decision());
  }
  if (event.initialization_failure() != nullptr) {
    *proto_event.mutable_initialization_failure() =
        ConvertBenchmarkInitializationFailure(*event.initialization_failure());
  }

  if (event.benchmark_event() != nullptr) {
    *proto_event.mutable_benchmark_event() =
        ConvertBenchmarkEvent(*event.benchmark_event());
  }

  return proto_event;
}

proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEventT& event) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(MiniBenchmarkEvent::Pack(fbb, &event));
  auto event_fbb =
      flatbuffers::GetRoot<MiniBenchmarkEvent>(fbb.GetBufferPointer());
  return ConvertFromFlatbuffer(*event_fbb);
}
}  // namespace tflite
