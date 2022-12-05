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
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace {

class FloatAddOpModel : public SingleOpModel {
 public:
  FloatAddOpModel() = default;
  void Init(const NnApi* nnapi, tflite::StatefulNnApiDelegate::Options options,
            const TensorData& input1, const TensorData& input2,
            const TensorData& output, ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    stateful_delegate_ =
        std::make_unique<StatefulNnApiDelegate>(nnapi, options);
    SetDelegate(stateful_delegate_.get());

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)}, /*num_threads=*/-1,
                     allow_fp32_relax_to_fp16, /*apply_delegate=*/false);
    compilation_status_ = ApplyDelegate();
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  TfLiteStatus GetCompilationStatus() { return compilation_status_; }

 protected:
  int input1_;
  int input2_;
  int output_;

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
  TfLiteStatus compilation_status_;
};

struct NnApiDeviceSelectionTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {
  void SetUp() override {
    ::tflite::delegate::nnapi::NnApiDelegateMockTest::SetUp();
    nnapi_mock_->GetDeviceCountReturnsCount<3>();
    nnapi_mock_->StubGetDeviceWith(
        [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
          *device = reinterpret_cast<ANeuralNetworksDevice*>(devIndex + 1);
          return 0;
        });
    nnapi_mock_->StubGetDeviceNameWith(
        [](const ANeuralNetworksDevice* device, const char** name) -> int {
          if (device == reinterpret_cast<ANeuralNetworksDevice*>(1)) {
            *name = "dsp";
          } else if (device == reinterpret_cast<ANeuralNetworksDevice*>(2)) {
            *name = "gpu";
          } else {
            *name = "nnapi-reference";
          }
          return ANEURALNETWORKS_NO_ERROR;
        });
    nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) -> int {
          supportedOps[0] = true;
          return ANEURALNETWORKS_NO_ERROR;
        });
  }
  void InitWithOptions(tflite::StatefulNnApiDelegate::Options options) {
    m.Init(nnapi_mock_->GetNnApi(), options, {TensorType_FLOAT32, {1, 2, 2, 1}},
           {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
           ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  }
  FloatAddOpModel m;
};

TEST_F(NnApiDeviceSelectionTest, DoesntSetDevicesWhenCpuAllowed) {
  nnapi_mock_->StubCompilationCreateForDevicesWith(
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         ANeuralNetworksCompilation** compilation) -> int {
        EXPECT_TRUE(false) << "Should not call createForDevices";
        return 1;
      });

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = false;
  InitWithOptions(options);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest, SetsDeviceBasedOnOptions) {
  nnapi_mock_->CompilationCreateReturns<1>();
  nnapi_mock_->StubCompilationCreateForDevicesWith(
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         ANeuralNetworksCompilation** compilation) -> int {
        EXPECT_EQ(numDevices, 1);
        EXPECT_EQ(devices[0], reinterpret_cast<ANeuralNetworksDevice*>(1));
        if (numDevices != 1 ||
            devices[0] != reinterpret_cast<ANeuralNetworksDevice*>(1)) {
          return 1;
        } else {
          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return ANEURALNETWORKS_NO_ERROR;
        }
      });

  tflite::StatefulNnApiDelegate::Options options;
  options.accelerator_name = "dsp";
  InitWithOptions(options);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest, DisallowsCPUBasedOnOptions) {
  nnapi_mock_->CompilationCreateReturns<1>();
  nnapi_mock_->StubCompilationCreateForDevicesWith(
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         ANeuralNetworksCompilation** compilation) -> int {
        EXPECT_EQ(numDevices, 2);
        EXPECT_EQ(devices[0], reinterpret_cast<ANeuralNetworksDevice*>(1));
        EXPECT_EQ(devices[1], reinterpret_cast<ANeuralNetworksDevice*>(2));
        if (numDevices != 2 ||
            devices[0] != reinterpret_cast<ANeuralNetworksDevice*>(1) ||
            devices[1] != reinterpret_cast<ANeuralNetworksDevice*>(2)) {
          return 1;
        } else {
          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return ANEURALNETWORKS_NO_ERROR;
        }
      });

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = true;
  InitWithOptions(options);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest,
       DoesNotDelegateIfOnlyReferenceDeviceIsAvailable_CpuEnabled) {
  // Only nnapi-reference is available on device
  nnapi_mock_->GetDeviceCountReturnsCount<1>();
  nnapi_mock_->GetDeviceNameReturnsName("nnapi-reference");

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = false;
  InitWithOptions(options);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
}

TEST_F(NnApiDeviceSelectionTest,
       DoesNotDelegateIfOnlyReferenceDeviceIsAvailable_CpuDisabled) {
  // Only nnapi-reference is available on device
  nnapi_mock_->GetDeviceCountReturnsCount<1>();
  nnapi_mock_->GetDeviceNameReturnsName("nnapi-reference");

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = true;
  InitWithOptions(options);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
}

struct UnsupportedOperationOnDeviceTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

class AcceleratedModel {
 public:
  StatefulNnApiDelegate* GetDelegate() { return stateful_delegate_.get(); }

 protected:
  // build a delegate with a target accelerator name.
  AcceleratedModel(const NnApi* nnapi, const std::string& accelerator_name,
                   int max_nnapi_partitions = 0) {
    StatefulNnApiDelegate::Options options;
    options.accelerator_name = accelerator_name.c_str();
    options.max_number_delegated_partitions = max_nnapi_partitions;
    stateful_delegate_ =
        std::make_unique<StatefulNnApiDelegate>(nnapi, options);
  }

  // build a delegate with no target accelerator name, can disable the NNAPI CPU
  // fallback implementation using the disallow_nnapi_cpu flag.
  AcceleratedModel(const NnApi* nnapi, bool disallow_nnapi_cpu,
                   int max_nnapi_partitions = 0) {
    StatefulNnApiDelegate::Options options;
    options.disallow_nnapi_cpu = disallow_nnapi_cpu;
    options.max_number_delegated_partitions = max_nnapi_partitions;
    stateful_delegate_ =
        std::make_unique<StatefulNnApiDelegate>(nnapi, options);
  }

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
};

class ArgMaxOpModel : public SingleOpModel, public AcceleratedModel {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType output_type, const NnApi* nnapi,
                const char* device_name)
      : SingleOpModel(), AcceleratedModel(nnapi, device_name) {
    Init(input_shape, input_type, axis_value, output_type);
  }

  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType output_type, const NnApi* nnapi,
                bool disallow_nnapi_cpu)
      : SingleOpModel(), AcceleratedModel(nnapi, disallow_nnapi_cpu) {
    Init(input_shape, input_type, axis_value, output_type);
  }

  int input() const { return input_; }

 protected:
  int input_;
  int axis_;
  int output_;

  void Init(std::initializer_list<int> input_shape, TensorType input_type,
            int axis_value, TensorType output_type) {
    SetDelegate(GetDelegate());
    input_ = AddInput(input_type);
    axis_ = AddConstInput(TensorType_INT32, {axis_value}, {1});
    output_ = AddOutput(output_type);

    SetBuiltinOp(BuiltinOperator_ARG_MAX, BuiltinOptions_ArgMaxOptions,
                 CreateArgMaxOptions(builder_, output_type).Union());
    BuildInterpreter({input_shape, {1}});
  }
};

TEST_F(UnsupportedOperationOnDeviceTest,
       ShouldUseDeviceFeatureLevelWhenSpecifyingTargetDevice) {
  nnapi_mock_->SetAndroidSdkVersion(29);
  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/28);
  // Setting this here because I want the delegate not to be applied in the
  // first case because the feature level is not high enough and not because the
  // operations are not supported by the device.
  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        std::fill(supportedOps, supportedOps + 1, true);
        return ANEURALNETWORKS_NO_ERROR;
      });

  ArgMaxOpModel m({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                  TensorType_INT32, nnapi_mock_->GetNnApi(), "test-device");
  m.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1)
      << "Expected Max not to be delegates since it not supported before NNAPI "
         "1.2 and device declares to support only NNAPI 1.1.";

  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/29);

  ArgMaxOpModel m1({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, nnapi_mock_->GetNnApi(), "test-device");
  m1.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(m1.Invoke(), kTfLiteOk);

  EXPECT_EQ(m1.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since it is supported in NNAPI 1.2.";
}

TEST_F(UnsupportedOperationOnDeviceTest,
       ShouldUseDeviceFeatureLevelWhenDisablingCPU) {
  nnapi_mock_->SetAndroidSdkVersion(29);
  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/28);
  // Setting this here because I want the delegate not to be applied in the
  // first case because the feature level is not high enough and not because the
  // operations are not supported by the device.
  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        std::fill(supportedOps, supportedOps + 1, true);
        return ANEURALNETWORKS_NO_ERROR;
      });

  ArgMaxOpModel m({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                  TensorType_INT32, nnapi_mock_->GetNnApi(),
                  /*disallow_nnapi_cpu=*/true);
  m.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1)
      << "Expected Max not to be delegates since it not supported before NNAPI "
         "1.2 and device declares to support only NNAPI 1.1.";

  ArgMaxOpModel m1({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, nnapi_mock_->GetNnApi(),
                   /*disallow_nnapi_cpu=*/false);
  m1.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(m1.Invoke(), kTfLiteOk);

  EXPECT_EQ(m1.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since we enabled NNAPI CPU "
         "implementation.";

  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/29);

  ArgMaxOpModel m2({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, nnapi_mock_->GetNnApi(),
                   /*disallow_nnapi_cpu=*/true);
  m2.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);

  EXPECT_EQ(m2.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since it is supported in NNAPI 1.2.";
}

// This is a model with two ops:
//
//  input1 ---->
//                ADD --
//  input2   -->        |
//                       -->
//                          SUB --> output
//  input3 ---------------->
//
class AddSubOpsAcceleratedModel : public MultiOpModel, public AcceleratedModel {
 public:
  AddSubOpsAcceleratedModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& input3, const TensorData& output,
                            ActivationFunctionType activation_type,
                            const NnApi* nnapi,
                            const std::string& accelerator_name,
                            bool allow_fp32_relax_to_fp16 = false)
      : MultiOpModel(), AcceleratedModel(nnapi, accelerator_name) {
    SetDelegate(GetDelegate());
    Init(input1, input2, input3, output, activation_type,
         allow_fp32_relax_to_fp16);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int input3() { return input3_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int input3_;
  int output_;

 private:
  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& input3, const TensorData& output,
            ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    input3_ = AddInput(input3);
    const int add_output = AddInnerTensor<float>(output);
    output_ = AddOutput(output);
    AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union(),
                 {input1_, input2_}, {add_output});
    AddBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union(),
                 {add_output, input3_}, {output_});
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)},
                     /*num_threads=*/-1, allow_fp32_relax_to_fp16,
                     /*apply_delegate=*/true);
  }
};

int should_build_model_with_sup_ops_compilation_model_create_count = 0;
int should_build_model_with_sup_ops_add_operation_count = 0;
TEST_F(UnsupportedOperationOnDeviceTest,
       ShouldBuildModelWithOnlyDeviceSupportedOps) {
  nnapi_mock_->SetNnapiSupportedDevice("test-device");

  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        // Returning the first as supported since this will leverage
        // the assertion on caching.
        supportedOps[0] = true;
        supportedOps[1] = false;
        return ANEURALNETWORKS_NO_ERROR;
      });

  nnapi_mock_->StubModelCreateWith([](ANeuralNetworksModel** model) -> int {
    ++should_build_model_with_sup_ops_compilation_model_create_count;
    *model = reinterpret_cast<ANeuralNetworksModel*>(1);
    return ANEURALNETWORKS_NO_ERROR;
  });

  nnapi_mock_->StubAddOperationWith(
      [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
         uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
         const uint32_t* outputs) -> int {
        ++should_build_model_with_sup_ops_add_operation_count;
        return ANEURALNETWORKS_NO_ERROR;
      });

  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
  ASSERT_EQ(should_build_model_with_sup_ops_compilation_model_create_count, 2)
      << "Model with unsupported operations has been cached";
  EXPECT_EQ(should_build_model_with_sup_ops_add_operation_count, 3)
      << "The second model should contain only one operation";
}

TEST_F(UnsupportedOperationOnDeviceTest, ShouldRunOnCpuIfDeviceSupportsNoOps) {
  nnapi_mock_->SetNnapiSupportedDevice("test-device");

  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        std::fill(supportedOps, supportedOps + 2, false);
        return ANEURALNETWORKS_NO_ERROR;
      });

  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 2);
}

int should_cache_model_compilation_model_create_count = 0;
TEST_F(UnsupportedOperationOnDeviceTest, ShouldCacheModelCompilation) {
  nnapi_mock_->SetNnapiSupportedDevice("test-device");

  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        std::fill(supportedOps, supportedOps + 2, true);
        return ANEURALNETWORKS_NO_ERROR;
      });

  nnapi_mock_->StubModelCreateWith([](ANeuralNetworksModel** model) -> int {
    ++should_cache_model_compilation_model_create_count;
    *model = reinterpret_cast<ANeuralNetworksModel*>(1);
    return ANEURALNETWORKS_NO_ERROR;
  });

  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  ASSERT_EQ(m.CountOpsExecutedByCpuKernel(), 0);
  EXPECT_EQ(should_cache_model_compilation_model_create_count, 1);
}

TEST_F(UnsupportedOperationOnDeviceTest,
       ShouldNotApplySupportedOperationsFilterBeforeAndroidSdk29) {
  nnapi_mock_->SetAndroidSdkVersion(28, /*set_unsupported_ops_to_null=*/true);
  nnapi_mock_->ModelCreateReturns<0>();
  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Delegation succeded without failures and all nodes have been delegated.
  ASSERT_EQ(m.CountOpsExecutedByCpuKernel(), 0);
}

// This is a model with two ops:
//
//  input1 ----> HARD_SWISH ---->
//                                ADD --> output
//  input2 ---------------------->
//
class HardSwishAddOpsAcceleratedModel : public MultiOpModel,
                                        public AcceleratedModel {
 public:
  HardSwishAddOpsAcceleratedModel(const TensorData& input1,
                                  const TensorData& input2,
                                  const TensorData& output,
                                  ActivationFunctionType activation_type,
                                  const NnApi* nnapi,
                                  const std::string& accelerator_name,
                                  bool allow_fp32_relax_to_fp16 = false)
      : MultiOpModel(), AcceleratedModel(nnapi, accelerator_name) {
    SetDelegate(GetDelegate());
    Init(input1, input2, output, activation_type, allow_fp32_relax_to_fp16);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;

 private:
  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& output, ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    const int hard_swish_output = AddInnerTensor<float>(output);
    output_ = AddOutput(output);
    AddBuiltinOp(BuiltinOperator_HARD_SWISH, BuiltinOptions_HardSwishOptions,
                 CreateHardSwishOptions(builder_).Union(), {input1_},
                 {hard_swish_output});
    AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union(),
                 {input1_, hard_swish_output}, {output_});
    BuildInterpreter({GetShape(input1_), GetShape(input2_)}, /*num_threads=*/-1,
                     allow_fp32_relax_to_fp16, /*apply_delegate=*/true);
  }
};

struct TfLiteOpMappedToMultipleNnApiOps
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

TEST_F(TfLiteOpMappedToMultipleNnApiOps, AllCostituentOpsNotSupported) {
  nnapi_mock_->ModelCreateReturns<0>();

  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        // HardSwish is mapped to 4 NNAPI ops, none of which supported.
        std::fill(supportedOps, supportedOps + 4, false);
        // After that we have the ADD op that is supported.
        supportedOps[4] = true;
        return ANEURALNETWORKS_NO_ERROR;
      });

  HardSwishAddOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
      nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Delegation succeded without failures and HardSwish has not been delegated
  // but Add has been correctly delegated.
  ASSERT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
}

TEST_F(TfLiteOpMappedToMultipleNnApiOps, NotAllConstitutentOpsSupported) {
  nnapi_mock_->ModelCreateReturns<0>();
  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        // HardSwish is mapped to 4 NNAPI ops (the first 4 ones), so we have 5
        // ops in the NNAPI model.
        std::fill(supportedOps, supportedOps + 5, true);
        // One of the NNAPI ops required by HardSwish is not supported.
        supportedOps[2] = false;
        return ANEURALNETWORKS_NO_ERROR;
      });

  HardSwishAddOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
      nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Delegation succeded without failures. HardSwish has not been delegated
  // but Add is delegated.
  ASSERT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
}

TEST_F(TfLiteOpMappedToMultipleNnApiOps, AllConstitutentOpsSupported) {
  nnapi_mock_->ModelCreateReturns<0>();
  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        // HardSwish is mapped to 4 NNAPI ops (the first 4 ones), so we have 5
        // ops in the NNAPI model.
        // All ops are supported by the accelerator.
        std::fill(supportedOps, supportedOps + 5, true);
        return ANEURALNETWORKS_NO_ERROR;
      });

  HardSwishAddOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE,
      nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Delegation succeded without failures and all nodes have been delegated.
  ASSERT_EQ(m.CountOpsExecutedByCpuKernel(), 0);
}

class QuantizedWeightsConvolutionOpModel : public SingleOpModel,
                                           public AcceleratedModel {
 public:
  QuantizedWeightsConvolutionOpModel(
      const NnApi* nnapi, std::string accelerator_name, const TensorData& input,
      const TensorData& filter, const TensorData& output, int stride_width = 2,
      int stride_height = 2, enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      int num_threads = -1, std::initializer_list<uint8_t> filter_data = {})
      : SingleOpModel(), AcceleratedModel(nnapi, accelerator_name) {
    SetDelegate(GetDelegate());

    input_ = AddInput(input);

    if (filter_data.size()) {
      filter_ = AddConstInput(filter, filter_data);
    } else {
      filter_ = AddInput(filter);
    }

    int bias_size = GetShape(filter_)[0];

    bias_ = AddInput({TensorType_FLOAT32, {bias_size}});

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)},
                     num_threads, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

int quantized_conv2d_model_added_nnapi_ops_count = 0;
TEST_F(TfLiteOpMappedToMultipleNnApiOps,
       AddedDequantizationsAreAccountedInModelOps) {
  nnapi_mock_->ModelCreateReturns<0>();
  nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
        std::fill(supportedOps,
                  supportedOps + quantized_conv2d_model_added_nnapi_ops_count,
                  true);
        return ANEURALNETWORKS_NO_ERROR;
      });
  nnapi_mock_->StubAddOperationWith(
      [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
         uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
         const uint32_t* outputs) -> int {
        ++quantized_conv2d_model_added_nnapi_ops_count;
        return ANEURALNETWORKS_NO_ERROR;
      });

  QuantizedWeightsConvolutionOpModel m(
      nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device", {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64}, {TensorType_FLOAT32, {}});
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 0);
  // When delegating quantized Conv2D, for each quantized inputs a
  // dequantize operation is added to the model.
  // In our case 1 Dequantize op for the weights is expected generating
  // a 2 ops model.
  EXPECT_EQ(quantized_conv2d_model_added_nnapi_ops_count, 2);
}

// Model with a chain of no-op (add with zero operations)
// interleaved with no-op custom nodes.
class LongIdentityModel : public MultiOpModel, public AcceleratedModel {
 public:
  LongIdentityModel(const std::vector<int>& input_shape, int graph_size,
                    const std::unordered_set<int>& custom_nodes_indexes,
                    const NnApi* nnapi, const std::string& accelerator_name,
                    int max_nnapi_partitions)
      : MultiOpModel(),
        AcceleratedModel(nnapi, accelerator_name, max_nnapi_partitions) {
    Init(input_shape, graph_size, custom_nodes_indexes);
  }

  LongIdentityModel(const std::vector<int>& input_shape, int graph_size,
                    const std::unordered_set<int>& custom_nodes_indexes,
                    const NnApi* nnapi, int max_nnapi_partitions)
      : MultiOpModel(), AcceleratedModel(nnapi, false, max_nnapi_partitions) {
    Init(input_shape, graph_size, custom_nodes_indexes);
  }

  void SetInput(std::vector<float> value) { PopulateTensor(input_, value); }

  int CountNnApiPartitions() {
    return std::count_if(
        std::begin(interpreter_->execution_plan()),
        std::end(interpreter_->execution_plan()), [this](const int node_index) {
          return interpreter_->node_and_registration(node_index)
                     ->first.delegate != nullptr;
        });
  }

 private:
  void Init(const std::vector<int>& input_shape, int graph_size,
            const std::unordered_set<int>& custom_nodes_indexes) {
    SetDelegate(GetDelegate());

    const TensorData tensor_data{TensorType_FLOAT32, input_shape};

    input_ = AddInput(tensor_data);
    zero_input_ = AddInput(tensor_data);

    std::vector<int> intermediate_outputs(graph_size - 1);
    std::generate(
        std::begin(intermediate_outputs), std::end(intermediate_outputs),
        [this, &tensor_data]() { return AddInnerTensor<float>(tensor_data); });

    output_ = AddOutput(tensor_data);

    AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_).Union(), {input_, zero_input_},
                 {intermediate_outputs[0]});

    for (int i = 0; i < intermediate_outputs.size() - 1; i++) {
      if (custom_nodes_indexes.count(i + 1) == 1) {
        AddCustomOp("custom_no_op", {}, [this]() { return CustomNoOpNode(); },
                    {intermediate_outputs[i]}, {intermediate_outputs[i + 1]});
      } else {
        AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                     CreateAddOptions(builder_).Union(),
                     {intermediate_outputs[i], zero_input_},
                     {intermediate_outputs[i + 1]});
      }
    }

    AddBuiltinOp(
        BuiltinOperator_ADD, BuiltinOptions_AddOptions,
        CreateAddOptions(builder_).Union(),
        {intermediate_outputs[intermediate_outputs.size() - 1], zero_input_},
        {output_});

    BuildInterpreter({GetShape(input_), GetShape(zero_input_)});

    std::vector<float> zero(GetTensorSize(input_), 0.0);
    PopulateTensor(zero_input_, zero);
  }

  // Return the registration of a custom node simply copying input to output.
  TfLiteRegistration* CustomNoOpNode() {
    static TfLiteRegistration no_op = {
        .init = [](TfLiteContext* context, const char* buffer,
                   size_t length) -> void* { return nullptr; },

        .free = [](TfLiteContext* context, void* buffer) -> void {},

        .prepare = [](TfLiteContext* context,
                      TfLiteNode* node) -> TfLiteStatus {
          if (node->inputs->size != 1 || node->outputs->size != 1) {
            return kTfLiteError;
          }

          return kTfLiteOk;
        },

        .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
          auto input_tensor = context->tensors[node->inputs->data[0]];
          auto output_tensor = context->tensors[node->outputs->data[0]];

          std::copy(input_tensor.data.raw,
                    input_tensor.data.raw + input_tensor.bytes,
                    output_tensor.data.raw);

          return kTfLiteOk;
        },

        .profiling_string = nullptr,
        .builtin_code = kTfLiteBuiltinDelegate,
        .custom_name = "NoOpTestDelegate",
        .version = 1,
    };

    return &no_op;
  }
  int input_;
  int zero_input_;
  int output_;
};

class NodeFilter {
 public:
  void ConfigureSupportedNodes(
      int graph_size, const std::unordered_set<int>& unsupported_indexes) {
    graph_size_ = graph_size;
    unsupported_indexes_ = unsupported_indexes;
  }

  void SetNodeSupport(bool* supported_ops) {
    for (int i = 0; i < graph_size_; i++) {
      supported_ops[i] = (unsupported_indexes_.count(i) == 0);
    }
  }

 private:
  int graph_size_;
  std::unordered_set<int> unsupported_indexes_;
};

// Using the same node filter for all DelegatePartitionLimitTests
// because StubGetSupportedOperationsForDevicesWith wants a C function.
NodeFilter* DelegatePartitionLimitTestNodeFilter() {
  static NodeFilter* node_filter = new NodeFilter();
  return node_filter;
}

class DelegatePartitionLimitTest
    : public ::tflite::delegate::nnapi::NnApiDelegateMockTest {
 protected:
  // Configure the underlying graph to generate a set of nnapi partition
  // with the sizes specified in nnapi_partition_sizes and the given
  // input_shape.
  void Init(int max_nnapi_partitions,
            const std::vector<int>& nnapi_partition_sizes,
            const std::vector<int>& input_shape,
            bool specify_accelerator = true) {
    // The graph will have as number of nodes the sum of nodes in the NNAPI
    // partitions plus nnapi_partition_sizes.size() - 1 nodes that will be
    // not supported by NNAPI and will cause the
    graph_size_ = std::accumulate(std::begin(nnapi_partition_sizes),
                                  std::end(nnapi_partition_sizes),
                                  nnapi_partition_sizes.size() - 1);

    std::unordered_set<int> unsupported_ops_idxs;
    int partition_node_idx = -1;
    for (int i = 0; i < nnapi_partition_sizes.size() - 1; i++) {
      partition_node_idx += nnapi_partition_sizes[i] + 1;
      unsupported_ops_idxs.insert(partition_node_idx);
    }

    if (specify_accelerator) {
      // Building a model that will contain initially a single partition
      // and will get then partitioned by checking the operations supported
      // by the target accelerator.
      // This because I am not able to know the size of each partition in my
      // stubbed GetSupportedOperationsForDevices API.
      DelegatePartitionLimitTestNodeFilter()->ConfigureSupportedNodes(
          graph_size_, unsupported_ops_idxs);

      nnapi_mock_->StubGetSupportedOperationsForDevicesWith(
          [](const ANeuralNetworksModel* model,
             const ANeuralNetworksDevice* const* devices, uint32_t num_devices,
             bool* supported_ops) -> int {
            DelegatePartitionLimitTestNodeFilter()->SetNodeSupport(
                supported_ops);
            return ANEURALNETWORKS_NO_ERROR;
          });

      model_ = std::make_unique<LongIdentityModel>(
          input_shape, graph_size_,
          /*custom_nodes_indexes=*/std::unordered_set<int>(),
          nnapi_mock_->GetNnApi(),
          /*accelerator_name=*/"test-device", max_nnapi_partitions);
    } else {
      // Building a model containing custom nodes that won't be supported
      // by the delegate and generate the partitions.
      model_ = std::make_unique<LongIdentityModel>(
          input_shape, graph_size_, unsupported_ops_idxs,
          nnapi_mock_->GetNnApi(), max_nnapi_partitions);
    }
  }

  std::unique_ptr<LongIdentityModel> model_;

  int OriginalGraphSize() { return graph_size_; }

 private:
  int graph_size_;
};

TEST_F(DelegatePartitionLimitTest, ShouldDelegateOnePartitionOnly) {
  Init(/*max_nnapi_partitions=*/1,
       /*nnapi_partition_sizes=*/{3, 2},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 1);
}

TEST_F(DelegatePartitionLimitTest,
       ShouldDelegateAllPossiblePartitionsIfLimitIsZero) {
  Init(/*max_nnapi_partitions=*/0,
       /*nnapi_partition_sizes=*/{3, 2},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 2);
}

TEST_F(DelegatePartitionLimitTest,
       ShouldDelegateAllPossiblePartitionsIfLimitIsNegative) {
  Init(/*max_nnapi_partitions=*/0,
       /*nnapi_partition_sizes=*/{3, 2},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 2);
}

TEST_F(DelegatePartitionLimitTest,
       ShouldDelegateAllPossiblePartitionsIfBelowLimit) {
  Init(/*max_nnapi_partitions=*/3,
       /*nnapi_partition_sizes=*/{3, 2},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 2);
}

TEST_F(DelegatePartitionLimitTest, ShouldDelegatePartitionWithHigherNodeCount) {
  int kLargestModelSize = 3;
  Init(/*max_nnapi_partitions=*/1,
       /*nnapi_partition_sizes=*/{3, 2},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 1);
  EXPECT_EQ(model_->CountOpsExecutedByCpuKernel(),
            OriginalGraphSize() - kLargestModelSize);
}

TEST_F(DelegatePartitionLimitTest,
       ShouldDelegatePartitionsWithHigherNodeCount) {
  int kLargestModelSize = 5;
  int kSecondLargestModelSize = 4;
  Init(/*max_nnapi_partitions=*/2,
       /*nnapi_partition_sizes=*/
       {1, kLargestModelSize, 2, kSecondLargestModelSize},
       /*input_shape=*/{1, 2, 2, 1});

  EXPECT_EQ(model_->CountNnApiPartitions(), 2);
  EXPECT_EQ(model_->CountOpsExecutedByCpuKernel(), OriginalGraphSize() - 9);
}

TEST_F(DelegatePartitionLimitTest,
       ShouldLimitPartitionsEvenWithoutAcceleratorNameSpecified) {
  int kLargestModelSize = 5;
  int kSecondLargestModelSize = 4;
  Init(/*max_nnapi_partitions=*/2,
       /*nnapi_partition_sizes=*/
       {1, kLargestModelSize, 2, kSecondLargestModelSize},
       /*input_shape=*/{1, 2, 2, 1}, /*specify_accelerator=*/false);

  EXPECT_EQ(model_->CountNnApiPartitions(), 2);
  EXPECT_EQ(
      model_->CountOpsExecutedByCpuKernel(),
      OriginalGraphSize() - (kLargestModelSize + kSecondLargestModelSize));
}

}  // namespace
}  // namespace tflite
