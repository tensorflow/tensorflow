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
#include "tensorflow/lite/acceleration/configuration/nnapi_plugin.h"

#include <algorithm>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/kernels/test_util.h"

// Tests for checking that the NNAPI Delegate plugin correctly handles all the
// options from the flatbuffer.
//
// Checking done at NNAPI call level, as that is where we have a mockable
// layer.
namespace tflite {
namespace {

using delegate::nnapi::NnApiMock;

class SingleAddOpModel : public tflite::SingleOpModel {
 public:
  // Note the caller owns the memory of the passed-in 'delegate'.
  void Build(TfLiteDelegate* delegate) {
    int input = AddInput({tflite::TensorType_FLOAT32, {1, 2, 2}});
    int constant = AddConstInput({tflite::TensorType_FLOAT32, {1, 2, 2}},
                                 {1.0f, 1.0f, 1.0f, 1.0f});
    AddOutput({tflite::TensorType_FLOAT32, {}});

    SetBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(builder_).Union());

    SetDelegate(delegate);
    // Set 'apply_delegate' to false to manually apply the delegate later and
    // check its return status.
    BuildInterpreter({GetShape(input), GetShape(constant)},
                     /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/true);
  }

  tflite::Interpreter* Interpreter() const { return interpreter_.get(); }
};

class NNAPIPluginTest : public ::testing::Test {
 protected:
  NNAPIPluginTest() : delegate_(nullptr, [](TfLiteDelegate*) {}) {}
  void SetUp() override {
    nnapi_ = const_cast<NnApi*>(NnApiImplementation());
    nnapi_mock_ = std::make_unique<NnApiMock>(nnapi_);
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) -> int {
      supportedOps[0] = true;
      return 0;
    };
  }

  template <NNAPIExecutionPreference input, int output>
  void CheckExecutionPreference() {
    // Note - this uses a template since the NNAPI functions are C function
    // pointers rather than lambdas so can't capture variables.
    nnapi_->ANeuralNetworksCompilation_setPreference =
        [](ANeuralNetworksCompilation* compilation, int32_t preference) {
          return preference - output;
        };
    CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0, input));
    // Since delegation succeeds, the model becomes immutable and hence can't
    // reuse it.
    SingleAddOpModel model;
    model.Build(delegate_.get());
    EXPECT_EQ(model.ApplyDelegate(), kTfLiteOk)
        << " given input: " << input << " expected output: " << output;
  }

  template <NNAPIExecutionPriority input, int output>
  void CheckExecutionPriority() {
    // Note - this uses a template since the NNAPI functions are C function
    // pointers rather than lambdas so can't capture variables.
    nnapi_->ANeuralNetworksCompilation_setPriority =
        [](ANeuralNetworksCompilation* compilation, int32_t priority) {
          return priority - output;
        };
    CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                       NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                       /*allow CPU=*/true, input));
    // Since delegation succeeds, the model becomes immutable and hence can't
    // reuse it.
    SingleAddOpModel model;
    model.Build(delegate_.get());
    EXPECT_EQ(model.ApplyDelegate(), kTfLiteOk)
        << " given input: " << input << " expected output: " << output;
  }

  void CreateDelegate(flatbuffers::Offset<NNAPISettings> nnapi_settings) {
    tflite_settings_ = flatbuffers::GetTemporaryPointer(
        fbb_,
        CreateTFLiteSettings(fbb_, tflite::Delegate_NNAPI, nnapi_settings));

    plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "NnapiPlugin", *tflite_settings_);
    delegate_ = plugin_->Create();
  }

  TfLiteStatus ApplyDelegate() {
    model_.Build(delegate_.get());
    return model_.ApplyDelegate();
  }

  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
  SingleAddOpModel model_;
  flatbuffers::FlatBufferBuilder fbb_;
  const TFLiteSettings* tflite_settings_ = nullptr;
  delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<delegates::DelegatePluginInterface> plugin_;
};

TEST(CompilationCachingFields, SourcedFromNNAPISettingsFields) {
  flatbuffers::FlatBufferBuilder fbb;
  auto nnapi_settings_cache_dir = fbb.CreateString("nnapi_settings_cache_dir");
  auto nnapi_settings_model_token =
      fbb.CreateString("nnapi_settings_model_token");
  NNAPISettingsBuilder nnapi_settings_builder(fbb);
  nnapi_settings_builder.add_cache_directory(nnapi_settings_cache_dir);
  nnapi_settings_builder.add_model_token(nnapi_settings_model_token);
  auto nnapi_settings = nnapi_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(tflite::Delegate_NNAPI);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  auto tflite_settings = tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  ::tflite::delegates::NnapiPlugin plugin(*tflite_settings_root);
  auto options = plugin.Options();
  EXPECT_STREQ(options.cache_dir, "nnapi_settings_cache_dir");
  EXPECT_STREQ(options.model_token, "nnapi_settings_model_token");
}

TEST(CompilationCachingFields,
     TFLiteSettingsFieldsOverrideNNAPISettingsFields) {
  flatbuffers::FlatBufferBuilder fbb;
  auto top_level_cache_dir = fbb.CreateString("top_level_cache_dir");
  auto top_level_model_token = fbb.CreateString("top_level_model_token");

  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_model_token(top_level_model_token);
  compilation_caching_settings_builder.add_cache_dir(top_level_cache_dir);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  auto nnapi_settings_cache_dir = fbb.CreateString("nnapi_settings_cache_dir");
  auto nnapi_settings_model_token =
      fbb.CreateString("nnapi_settings_model_token");
  NNAPISettingsBuilder nnapi_settings_builder(fbb);
  nnapi_settings_builder.add_cache_directory(nnapi_settings_cache_dir);
  nnapi_settings_builder.add_model_token(nnapi_settings_model_token);
  auto nnapi_settings = nnapi_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(tflite::Delegate_NNAPI);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  ::tflite::delegates::NnapiPlugin plugin(*tflite_settings_root);
  auto options = plugin.Options();
  EXPECT_STREQ(options.cache_dir, "top_level_cache_dir");
  EXPECT_STREQ(options.model_token, "top_level_model_token");
}

TEST(CompilationCachingFields,
     NNAPISettingsFieldsUsedIfTFLiteSettingsFieldsArePresentButEmpty) {
  flatbuffers::FlatBufferBuilder fbb;
  auto empty_top_level_cache_dir = fbb.CreateString("");
  auto empty_top_level_model_token = fbb.CreateString("");

  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_model_token(
      empty_top_level_model_token);
  compilation_caching_settings_builder.add_cache_dir(empty_top_level_cache_dir);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  auto nnapi_settings_cache_dir = fbb.CreateString("nnapi_settings_cache_dir");
  auto nnapi_settings_model_token =
      fbb.CreateString("nnapi_settings_model_token");
  NNAPISettingsBuilder nnapi_settings_builder(fbb);
  nnapi_settings_builder.add_cache_directory(nnapi_settings_cache_dir);
  nnapi_settings_builder.add_model_token(nnapi_settings_model_token);
  auto nnapi_settings = nnapi_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(tflite::Delegate_NNAPI);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  ::tflite::delegates::NnapiPlugin plugin(*tflite_settings_root);
  auto options = plugin.Options();
  EXPECT_STREQ(options.cache_dir, "nnapi_settings_cache_dir");
  EXPECT_STREQ(options.model_token, "nnapi_settings_model_token");
}

TEST(CompilationCachingFields,
     FallbackToNNAPISettingsCacheDirFieldIfTFLiteSettingsCacheDirIsMissing) {
  flatbuffers::FlatBufferBuilder fbb;
  auto top_level_model_token = fbb.CreateString("top_level_model_token");

  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_model_token(top_level_model_token);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  auto nnapi_settings_cache_dir = fbb.CreateString("nnapi_settings_cache_dir");
  auto nnapi_settings_model_token =
      fbb.CreateString("nnapi_settings_model_token");
  NNAPISettingsBuilder nnapi_settings_builder(fbb);
  nnapi_settings_builder.add_cache_directory(nnapi_settings_cache_dir);
  nnapi_settings_builder.add_model_token(nnapi_settings_model_token);
  auto nnapi_settings = nnapi_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(tflite::Delegate_NNAPI);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  ::tflite::delegates::NnapiPlugin plugin(*tflite_settings_root);

  auto options = plugin.Options();
  EXPECT_STREQ(options.cache_dir, "nnapi_settings_cache_dir");
  EXPECT_STREQ(options.model_token, "top_level_model_token");
}

TEST(
    CompilationCachingFields,
    FallbackToNNAPISettingsModelTokenFieldIfTFLiteSettingsModelTokenIsMissing) {
  flatbuffers::FlatBufferBuilder fbb;
  auto top_level_cache_dir = fbb.CreateString("top_level_cache_dir");

  CompilationCachingSettingsBuilder compilation_caching_settings_builder(fbb);
  compilation_caching_settings_builder.add_cache_dir(top_level_cache_dir);
  auto compilation_caching_settings =
      compilation_caching_settings_builder.Finish();

  auto nnapi_settings_cache_dir = fbb.CreateString("nnapi_settings_cache_dir");
  auto nnapi_settings_model_token =
      fbb.CreateString("nnapi_settings_model_token");
  NNAPISettingsBuilder nnapi_settings_builder(fbb);
  nnapi_settings_builder.add_cache_directory(nnapi_settings_cache_dir);
  nnapi_settings_builder.add_model_token(nnapi_settings_model_token);
  auto nnapi_settings = nnapi_settings_builder.Finish();

  TFLiteSettingsBuilder tflite_settings_builder(fbb);
  tflite_settings_builder.add_delegate(tflite::Delegate_NNAPI);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  tflite_settings_builder.add_compilation_caching_settings(
      compilation_caching_settings);
  auto tflite_settings = tflite_settings_builder.Finish();
  fbb.Finish(tflite_settings);
  auto tflite_settings_root =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());

  ::tflite::delegates::NnapiPlugin plugin(*tflite_settings_root);

  auto options = plugin.Options();
  EXPECT_STREQ(options.cache_dir, "top_level_cache_dir");
  EXPECT_STREQ(options.model_token, "nnapi_settings_model_token");
}

TEST_F(NNAPIPluginTest, PassesAcceleratorNameFailure) {
  // Fails with non-existent "foo".
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("foo")));
  EXPECT_EQ(kTfLiteDelegateError, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesAcceleratorNameSuccess) {
  // Succeeds with "test-device" supported by the mock.
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("test-device")));
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesExecutionPreference) {
  CheckExecutionPreference<NNAPIExecutionPreference_UNDEFINED,
                           StatefulNnApiDelegate::Options::kUndefined>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_LOW_POWER,
                           StatefulNnApiDelegate::Options::kLowPower>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER,
                           StatefulNnApiDelegate::Options::kFastSingleAnswer>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED,
                           StatefulNnApiDelegate::Options::kSustainedSpeed>();
}

TEST_F(NNAPIPluginTest, PassesExecutionPriority) {
  nnapi_->android_sdk_version =
      tflite::delegate::nnapi::kMinSdkVersionForNNAPI13;
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED,
                         ANEURALNETWORKS_PRIORITY_DEFAULT>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_LOW,
                         ANEURALNETWORKS_PRIORITY_LOW>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM,
                         ANEURALNETWORKS_PRIORITY_MEDIUM>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH,
                         ANEURALNETWORKS_PRIORITY_HIGH>();
}

TEST_F(NNAPIPluginTest, PassesCachingParameters) {
  nnapi_->ANeuralNetworksCompilation_setCaching =
      [](ANeuralNetworksCompilation* compilation, const char* cacheDir,
         const uint8_t* token) -> int {
    if (std::string(cacheDir) != "d") return 1;
    // Token is hashed with other bits, just check that it's not empty.
    if (std::string(reinterpret_cast<const char*>(token)).empty()) return 2;
    return 0;
  };
  CreateDelegate(CreateNNAPISettings(fbb_, 0, fbb_.CreateString("d"),
                                     fbb_.CreateString("t")));
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesFalseNNAPICpuFlag) {
  CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                     NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                     /* allow CPU */ false));
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    // Since no CPU, should only pass one device.
    return numDevices - 1;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesTrueNNAPICpuFlag) {
  CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                     NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                     /* allow CPU */ true));
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    // With CPU allowed, should pass two devices.
    return numDevices - 2;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

/*
 * Building a model with three operations that can be used to create multiple
 * delegated partitions.
 *
 *  input1 ---
 *            | -  ADD -- ROUND --
 *            |                   | - ADD -- output1
 *  input2 ---                    |
 *                                |
 *  input3 -----------------------
 */
class MultiplePartitionsModel : public tflite::MultiOpModel {
 public:
  // Note the caller owns the memory of the passed-in 'delegate'.
  void Build(TfLiteDelegate* delegate) {
    const tflite::TensorData tensors_data = {tflite::TensorType_FLOAT32,
                                             {1, 2, 2}};
    int input1 = AddInput(tensors_data);
    int input2 = AddInput(tensors_data);
    int input3 = AddInput(tensors_data);
    int add_out = AddInnerTensor<float>(tensors_data);
    int round_out = AddInnerTensor<float>(tensors_data);
    int output = AddOutput(tensors_data);

    AddBuiltinOp(
        tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union(),
        {input1, input2}, {add_out});

    AddBuiltinOp(tflite::BuiltinOperator_ROUND, tflite::BuiltinOptions_NONE,
                 /*builtin_options=*/0, {add_out}, {round_out});

    AddBuiltinOp(
        tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union(),
        {round_out, input3}, {output});

    SetDelegate(delegate);
    // Set 'apply_delegate' to false to manually apply the delegate later and
    // check its return status.
    BuildInterpreter({GetShape(input1), GetShape(input2), GetShape(input3)},
                     /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/true);
  }

  tflite::Interpreter* Interpreter() const { return interpreter_.get(); }
};

class NNAPIMultiOpPluginTest : public ::testing::Test {
 protected:
  NNAPIMultiOpPluginTest() : delegate_(nullptr, [](TfLiteDelegate*) {}) {}
  void SetUp() override {
    nnapi_ = const_cast<NnApi*>(NnApiImplementation());
    nnapi_mock_ = std::make_unique<NnApiMock>(nnapi_);
  }

  void CreateDelegate(flatbuffers::Offset<NNAPISettings> nnapi_settings,
                      int max_delegated_partitions) {
    tflite_settings_ = flatbuffers::GetTemporaryPointer(
        fbb_,
        CreateTFLiteSettings(fbb_, tflite::Delegate_NNAPI, nnapi_settings,
                             /* gpu_settings */ 0,
                             /* hexagon_settings */ 0,
                             /* xnnpack_settings */ 0,
                             /* coreml_settings */ 0,
                             /* cpu_settings */ 0, max_delegated_partitions,
                             /* disable_default_delegates */ false,
                             /* stable_delegate_loader_settings */ 0));

    plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "NnapiPlugin", *tflite_settings_);
    delegate_ = plugin_->Create();
  }

  int CountNnApiPartitions() {
    return std::count_if(std::begin(model_.Interpreter()->execution_plan()),
                         std::end(model_.Interpreter()->execution_plan()),
                         [this](const int node_index) {
                           return model_.Interpreter()
                                      ->node_and_registration(node_index)
                                      ->first.delegate != nullptr;
                         });
  }

  TfLiteStatus ApplyDelegate() {
    model_.Build(delegate_.get());
    return model_.ApplyDelegate();
  }

  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
  MultiplePartitionsModel model_;
  flatbuffers::FlatBufferBuilder fbb_;
  const TFLiteSettings* tflite_settings_ = nullptr;
  delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<delegates::DelegatePluginInterface> plugin_;
};

TEST_F(NNAPIMultiOpPluginTest, PassesMaxDelegatedPartitionsFlag) {
  CreateDelegate(CreateNNAPISettings(
                     fbb_, 0, 0, 0, NNAPIExecutionPreference_UNDEFINED, 0, 0,
                     /* allow CPU */ true,
                     /* execution_priority */
                     tflite::NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED,
                     /* allow_dynamic_dimensions */ false,
                     /* allow_fp16_precision_for_fp32 */ false),
                 /* max_delegated_partitions */ 1);
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    supportedOps[1] = false;
    supportedOps[2] = true;
    return 0;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
  EXPECT_EQ(CountNnApiPartitions(), 1);
}

}  // namespace
}  // namespace tflite
