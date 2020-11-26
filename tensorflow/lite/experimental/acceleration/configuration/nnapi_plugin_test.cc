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
#include <memory>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"

// Tests for checking that the NNAPI Delegate plugin correctly handles all the
// options from the flatbuffer.
//
// Checking done at NNAPI call level, as that is where we have a mockable
// layer.
namespace tflite {
namespace {

using delegate::nnapi::NnApiMock;

class SingleAddOpModel : tflite::SingleOpModel {
 public:
  void Build() {
    int input = AddInput({tflite::TensorType_FLOAT32, {1, 2, 2}});
    int constant = AddConstInput({tflite::TensorType_FLOAT32, {1, 2, 2}},
                                 {1.0f, 1.0f, 1.0f, 1.0f});
    AddOutput({tflite::TensorType_FLOAT32, {}});

    SetBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(builder_).Union());
    // Set apply_delegate to false to skip applying TfLite default delegates.
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
    nnapi_mock_ = absl::make_unique<NnApiMock>(nnapi_);
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) -> int {
      supportedOps[0] = true;
      return 0;
    };
    model_.Build();
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
    model.Build();
    EXPECT_EQ(model.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
              kTfLiteOk)
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
    model.Build();
    EXPECT_EQ(model.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
              kTfLiteOk)
        << " given input: " << input << " expected output: " << output;
  }

  void CreateDelegate(flatbuffers::Offset<NNAPISettings> settings) {
    settings_ = flatbuffers::GetTemporaryPointer(
        fbb_, CreateTFLiteSettings(fbb_, tflite::Delegate_NNAPI, settings));

    plugin_ = delegates::DelegatePluginRegistry::CreateByName("NnapiPlugin",
                                                              *settings_);
    delegate_ = plugin_->Create();
  }

  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
  SingleAddOpModel model_;
  flatbuffers::FlatBufferBuilder fbb_;
  const TFLiteSettings* settings_ = nullptr;
  delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<delegates::DelegatePluginInterface> plugin_;
};

TEST_F(NNAPIPluginTest, PassesAcceleratorName) {
  // Fails with non-existent "foo".
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("foo")));
  EXPECT_EQ(model_.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
            kTfLiteDelegateError);

  // Succeeds with "test-device" supported by the mock.
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("test-device")));
  EXPECT_EQ(model_.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
            kTfLiteOk);
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
  EXPECT_EQ(model_.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
            kTfLiteOk);
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
  EXPECT_EQ(model_.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
            kTfLiteOk);
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
  EXPECT_EQ(model_.Interpreter()->ModifyGraphWithDelegate(delegate_.get()),
            kTfLiteOk);
}

}  // namespace
}  // namespace tflite
