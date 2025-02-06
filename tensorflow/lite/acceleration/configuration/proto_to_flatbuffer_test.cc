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
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace {

// Tests converting EdgeTpuSettings from proto to flatbuffer format.
TEST(ConversionTest, EdgeTpuSettings) {
  // Define the fields to be tested.
  const std::vector<int32_t> kHardwareClusterIds{1};
  const std::string kPublicModelId = "public_model_id";
  const tflite::proto::EdgeTpuSettings_UseLayerIrTgcBackend
      kUseLayerIrTgcBackend =
          tflite::proto::EdgeTpuSettings::USE_LAYER_IR_TGC_BACKEND_YES;

  // Create the proto settings.
  proto::ComputeSettings input_settings;
  auto* edgetpu_settings =
      input_settings.mutable_tflite_settings()->mutable_edgetpu_settings();
  edgetpu_settings->set_public_model_id(kPublicModelId);
  edgetpu_settings->set_use_layer_ir_tgc_backend(kUseLayerIrTgcBackend);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;
  *edgetpu_settings->mutable_hardware_cluster_ids() = {
      kHardwareClusterIds.begin(), kHardwareClusterIds.end()};

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder)
                             ->tflite_settings()
                             ->edgetpu_settings();

  // Verify the conversion results.
  EXPECT_EQ(output_settings->hardware_cluster_ids()->size(), 1);
  EXPECT_EQ(output_settings->hardware_cluster_ids()->Get(0),
            kHardwareClusterIds[0]);
  EXPECT_EQ(output_settings->public_model_id()->str(), kPublicModelId);
  EXPECT_EQ(output_settings->use_layer_ir_tgc_backend(),
            tflite::EdgeTpuSettings_::
                UseLayerIrTgcBackend_USE_LAYER_IR_TGC_BACKEND_YES);
}

// Tests converting TFLiteSettings from proto to flatbuffer format.
TEST(ConversionTest, TFLiteSettings) {
  // Define the fields to be tested.
  const std::vector<int32_t> kHardwareClusterIds{1};
  const std::string kPublicModelId = "public_model_id";
  const tflite::proto::EdgeTpuSettings_UseLayerIrTgcBackend
      kUseLayerIrTgcBackend =
          tflite::proto::EdgeTpuSettings::USE_LAYER_IR_TGC_BACKEND_YES;

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  input_settings.set_delegate(::tflite::proto::EDGETPU);
  auto* edgetpu_settings = input_settings.mutable_edgetpu_settings();
  edgetpu_settings->set_public_model_id(kPublicModelId);
  edgetpu_settings->set_use_layer_ir_tgc_backend(kUseLayerIrTgcBackend);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;
  *edgetpu_settings->mutable_hardware_cluster_ids() = {
      kHardwareClusterIds.begin(), kHardwareClusterIds.end()};

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  EXPECT_EQ(output_settings->delegate(), ::tflite::Delegate_EDGETPU);
  const auto* output_edgetpu_settings = output_settings->edgetpu_settings();
  EXPECT_EQ(output_edgetpu_settings->hardware_cluster_ids()->size(), 1);
  EXPECT_EQ(output_edgetpu_settings->hardware_cluster_ids()->Get(0),
            kHardwareClusterIds[0]);
  EXPECT_EQ(output_edgetpu_settings->public_model_id()->str(), kPublicModelId);
  EXPECT_EQ(output_edgetpu_settings->use_layer_ir_tgc_backend(),
            tflite::EdgeTpuSettings_::
                UseLayerIrTgcBackend_USE_LAYER_IR_TGC_BACKEND_YES);
}

TEST(ConversionTest, StableDelegateLoaderSettings) {
  // Define the fields to be tested.
  const std::string kDelegatePath = "TEST_DELEGATE_PATH";
  const std::string kDelegateName = "TEST_DELEGATE_NAME";

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  auto* stable_delegate_loader_settings =
      input_settings.mutable_stable_delegate_loader_settings();
  stable_delegate_loader_settings->set_delegate_path(kDelegatePath);
  stable_delegate_loader_settings->set_delegate_name(kDelegateName);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  const auto* output_stable_delegate_loader_settings =
      output_settings->stable_delegate_loader_settings();
  ASSERT_NE(output_stable_delegate_loader_settings, nullptr);
  EXPECT_EQ(output_stable_delegate_loader_settings->delegate_path()->str(),
            kDelegatePath);
  EXPECT_EQ(output_stable_delegate_loader_settings->delegate_name()->str(),
            kDelegateName);
}

TEST(ConversionTest, CompilationCachingSettings) {
  // Define the fields to be tested.
  const std::string kCacheDir = "TEST_CACHE_DIR";
  const std::string kModelToken = "TEST_MODEL_TOKEN";

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  auto* compilation_caching_settings =
      input_settings.mutable_compilation_caching_settings();
  compilation_caching_settings->set_cache_dir(kCacheDir);
  compilation_caching_settings->set_model_token(kModelToken);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  const auto* output_compilation_caching_settings =
      output_settings->compilation_caching_settings();
  ASSERT_NE(output_compilation_caching_settings, nullptr);
  EXPECT_EQ(output_compilation_caching_settings->cache_dir()->str(), kCacheDir);
  EXPECT_EQ(output_compilation_caching_settings->model_token()->str(),
            kModelToken);
}

TEST(ConversionTest, ArmNNSettings) {
  // Define the fields to be tested.
  const std::string kBackends = "TEST_BACKENDS";
  const bool kFastmath = true;
  const std::string kAdditionalParameters = "TEST_ADDITIONAL_PARAMETERS";

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  auto* armnn_settings = input_settings.mutable_armnn_settings();
  armnn_settings->set_backends(kBackends);
  armnn_settings->set_fastmath(kFastmath);
  armnn_settings->set_additional_parameters(kAdditionalParameters);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  const auto* output_armnn_settings = output_settings->armnn_settings();
  ASSERT_NE(output_armnn_settings, nullptr);
  EXPECT_EQ(output_armnn_settings->backends()->str(), kBackends);
  EXPECT_EQ(output_armnn_settings->fastmath(), kFastmath);
  EXPECT_EQ(output_armnn_settings->additional_parameters()->str(),
            kAdditionalParameters);
}

TEST(ConversionTest, MtkNeuronSettings) {
  // Define the fields to be tested.
  const proto::MtkNeuronSettings_ExecutionPreference kExecutionPreference =
      proto::MtkNeuronSettings::PREFERENCE_FAST_SINGLE_ANSWER;
  const proto::MtkNeuronSettings_ExecutionPriority kExecutionPriority =
      proto::MtkNeuronSettings::PRIORITY_MEDIUM;
  const proto::MtkNeuronSettings_OptimizationHint kOptimizationHint =
      proto::MtkNeuronSettings::OPTIMIZATION_LOW_LATENCY;
  const proto::MtkNeuronSettings_OperationCheckMode kOperationCheckMode =
      proto::MtkNeuronSettings::PER_NODE_OPERATION_CHECK;
  const bool kAllowFp16 = true;
  const bool kUseAhwb = false;
  const bool kUseCacheableBuffer = true;
  const std::string kCompileOptions = "TEST_COMPILE_OPTIONS";
  const std::string kAcceleratorName = "TEST_ACCELERATOR_NAME";
  const std::string kNeuronConfigPath = "TEST_NEURON_CONFIG_PATH";
  const int32_t kInferenceDeadlineMs = 1337;
  const int32_t kInferenceAbortTimeMs = 42;

  // Create the proto settings.
  proto::TFLiteSettings input_settings;
  auto* mtk_neuron_settings = input_settings.mutable_mtk_neuron_settings();
  mtk_neuron_settings->set_execution_preference(kExecutionPreference);
  mtk_neuron_settings->set_execution_priority(kExecutionPriority);
  mtk_neuron_settings->add_optimization_hints(kOptimizationHint);
  mtk_neuron_settings->set_operation_check_mode(kOperationCheckMode);
  mtk_neuron_settings->set_allow_fp16_precision_for_fp32(kAllowFp16);
  mtk_neuron_settings->set_use_ahwb(kUseAhwb);
  mtk_neuron_settings->set_use_cacheable_buffer(kUseCacheableBuffer);
  mtk_neuron_settings->add_compile_options(kCompileOptions);
  mtk_neuron_settings->add_accelerator_names(kAcceleratorName);
  mtk_neuron_settings->set_neuron_config_path(kNeuronConfigPath);
  mtk_neuron_settings->set_inference_deadline_ms(kInferenceDeadlineMs);
  mtk_neuron_settings->set_inference_abort_time_ms(kInferenceAbortTimeMs);
  flatbuffers::FlatBufferBuilder flatbuffers_builder;

  // Convert.
  auto output_settings = ConvertFromProto(input_settings, &flatbuffers_builder);

  // Verify the conversion results.
  const auto* output_mtk_neuron_settings =
      output_settings->mtk_neuron_settings();
  ASSERT_NE(output_mtk_neuron_settings, nullptr);
  EXPECT_EQ(
      output_mtk_neuron_settings->execution_preference(),
      MtkNeuronSettings_::ExecutionPreference_PREFERENCE_FAST_SINGLE_ANSWER);
  EXPECT_EQ(output_mtk_neuron_settings->execution_priority(),
            MtkNeuronSettings_::ExecutionPriority_PRIORITY_MEDIUM);

  EXPECT_EQ(output_mtk_neuron_settings->optimization_hints()->size(), 1);
  EXPECT_EQ(output_mtk_neuron_settings->optimization_hints()->Get(0),
            kOptimizationHint);
  EXPECT_EQ(output_mtk_neuron_settings->operation_check_mode(),
            MtkNeuronSettings_::OperationCheckMode_PER_NODE_OPERATION_CHECK);
  EXPECT_EQ(output_mtk_neuron_settings->allow_fp16_precision_for_fp32(),
            kAllowFp16);
  EXPECT_EQ(output_mtk_neuron_settings->use_ahwb(), kUseAhwb);
  EXPECT_EQ(output_mtk_neuron_settings->use_cacheable_buffer(),
            kUseCacheableBuffer);
  EXPECT_EQ(output_mtk_neuron_settings->compile_options()->size(), 1);
  EXPECT_EQ(output_mtk_neuron_settings->compile_options()->Get(0)->str(),
            kCompileOptions);
  EXPECT_EQ(output_mtk_neuron_settings->accelerator_names()->size(), 1);
  EXPECT_EQ(output_mtk_neuron_settings->accelerator_names()->Get(0)->str(),
            kAcceleratorName);
  EXPECT_EQ(output_mtk_neuron_settings->neuron_config_path()->str(),
            kNeuronConfigPath);
  EXPECT_EQ(output_mtk_neuron_settings->inference_deadline_ms(),
            kInferenceDeadlineMs);
  EXPECT_EQ(output_mtk_neuron_settings->inference_abort_time_ms(),
            kInferenceAbortTimeMs);
}

}  // namespace
}  // namespace tflite
