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

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace {

class SingleOpModelWithNNAPI : public SingleOpModel {
 public:
  SingleOpModelWithNNAPI() = default;
  void Init(tflite::StatefulNnApiDelegate::Options options) {
    stateful_delegate_.reset(new StatefulNnApiDelegate(options));
    auto* delegate = stateful_delegate_.get();
    this->SetApplyDelegate([delegate, this](Interpreter* interpreter) {
      compilation_status_ = interpreter->ModifyGraphWithDelegate(delegate);
    });
  }

  StatefulNnApiDelegate* GetDelegate() { return stateful_delegate_.get(); }

  void SetBufferHandle(int index, TfLiteBufferHandle handle) {
    interpreter_->SetBufferHandle(index, handle, stateful_delegate_.get());
  }
  TfLiteStatus GetCompilationStatus() { return compilation_status_; }

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
  TfLiteStatus compilation_status_;
};

class FloatAddOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatAddOpModel() = default;
  void Init(tflite::StatefulNnApiDelegate::Options options,
            const TensorData& input1, const TensorData& input2,
            const TensorData& output, ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    SingleOpModelWithNNAPI::Init(options);
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)},
                     allow_fp32_relax_to_fp16);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;

 private:
};

struct NnApiDeviceSelectionTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {
  void SetUp() override {
    ::tflite::delegate::nnapi::NnApiDelegateMockTest::SetUp();
    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = 3;
      return 0;
    };
    nnapi_->ANeuralNetworks_getDevice =
        [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
      *device = reinterpret_cast<ANeuralNetworksDevice*>(devIndex + 1);
      return 0;
    };
    nnapi_->ANeuralNetworksDevice_getName =
        [](const ANeuralNetworksDevice* device, const char** name) -> int {
      if (device == reinterpret_cast<ANeuralNetworksDevice*>(1)) {
        *name = "dsp";
      } else if (device == reinterpret_cast<ANeuralNetworksDevice*>(2)) {
        *name = "gpu";
      } else {
        *name = "nnapi-reference";
      }
      return 0;
    };
  }
  void InitWithOptions(tflite::StatefulNnApiDelegate::Options options) {
    m.Init(options, {TensorType_FLOAT32, {1, 2, 2, 1}},
           {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
           ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  }
  FloatAddOpModel m;
};

TEST_F(NnApiDeviceSelectionTest, DoesntSetDevicesWithoutFlags) {
  nnapi_->ANeuralNetworksCompilation_createForDevices =
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         ANeuralNetworksCompilation** compilation) -> int {
    EXPECT_TRUE(false) << "Should not call createForDevices";
    return 1;
  };

  tflite::StatefulNnApiDelegate::Options options;
  InitWithOptions(options);
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest, SetsDeviceBasedOnOptions) {
  nnapi_mock_->CompilationCreateReturns<1>();
  nnapi_->ANeuralNetworksCompilation_createForDevices =
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
      return 0;
    }
  };

  tflite::StatefulNnApiDelegate::Options options;
  options.accelerator_name = "dsp";
  InitWithOptions(options);
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest, DisallowsCPUBasedOnOptions) {
  nnapi_mock_->CompilationCreateReturns<1>();
  nnapi_->ANeuralNetworksCompilation_createForDevices =
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
      return 0;
    }
  };

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = true;
  InitWithOptions(options);
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
}

TEST_F(NnApiDeviceSelectionTest,
       DoesNotDelegateIfOnlyReferenceDeviceIsAvailable_CpuEnabled) {
  // Only nnapi-reference is available on device
  nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
    *numDevices = 1;
    return 0;
  };
  nnapi_->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(1)) {
      *name = "nnapi-reference";
    }
    return 0;
  };

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = false;
  InitWithOptions(options);
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);
  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1);
}

TEST_F(NnApiDeviceSelectionTest,
       DoesNotDelegateIfOnlyReferenceDeviceIsAvailable_CpuDisabled) {
  // Only nnapi-reference is available on device
  nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
    *numDevices = 1;
    return 0;
  };
  nnapi_->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(1)) {
      *name = "nnapi-reference";
    }
    return 0;
  };

  tflite::StatefulNnApiDelegate::Options options;
  options.disallow_nnapi_cpu = true;
  InitWithOptions(options);
  m.Invoke();
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
  explicit AcceleratedModel(const std::string& accelerator_name) {
    StatefulNnApiDelegate::Options options;
    options.accelerator_name = accelerator_name.c_str();
    stateful_delegate_.reset(new StatefulNnApiDelegate(options));
  }

  // build a delegate with no target accelerator name, can disable the NNAPI CPU
  // fallback implementation using the disallow_nnapi_cpu flag.
  explicit AcceleratedModel(bool disallow_nnapi_cpu) {
    StatefulNnApiDelegate::Options options;
    options.disallow_nnapi_cpu = disallow_nnapi_cpu;
    stateful_delegate_.reset(new StatefulNnApiDelegate(options));
  }

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
};

class ArgMaxOpModel : public SingleOpModel, public AcceleratedModel {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType output_type, const char* device_name)
      : SingleOpModel(), AcceleratedModel(device_name) {
    Init(input_shape, input_type, axis_value, output_type);
  }

  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int axis_value, TensorType output_type, bool disallow_nnapi_cpu)
      : SingleOpModel(), AcceleratedModel(disallow_nnapi_cpu) {
    Init(input_shape, input_type, axis_value, output_type);
  }

  int input() const { return input_; }

 protected:
  int input_;
  int axis_;
  int output_;

  void Init(std::initializer_list<int> input_shape, TensorType input_type,
            int axis_value, TensorType output_type) {
    auto* delegate = GetDelegate();
    this->SetApplyDelegate([delegate](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(delegate);
    });
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

  ArgMaxOpModel m({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                  TensorType_INT32, "test-device");
  m.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  m.Invoke();

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1)
      << "Expected Max not to be delegates since it not supported before NNAPI "
         "1.2 and device declares to support only NNAPI 1.1.";

  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/29);

  ArgMaxOpModel m1({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, "test-device");
  m1.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  m1.Invoke();

  EXPECT_EQ(m1.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since it is supported in NNAPI 1.2.";
}

TEST_F(UnsupportedOperationOnDeviceTest,
       ShouldUseDeviceFeatureLevelWhenDisablingCPU) {
  nnapi_mock_->SetAndroidSdkVersion(29);
  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/28);

  ArgMaxOpModel m({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                  TensorType_INT32, /*disallow_nnapi_cpu=*/true);
  m.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  m.Invoke();

  EXPECT_EQ(m.CountOpsExecutedByCpuKernel(), 1)
      << "Expected Max not to be delegates since it not supported before NNAPI "
         "1.2 and device declares to support only NNAPI 1.1.";

  ArgMaxOpModel m1({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, /*disallow_nnapi_cpu=*/false);
  m1.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  m1.Invoke();

  EXPECT_EQ(m1.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since we enabled NNAPI CPU "
         "implementation.";

  nnapi_mock_->SetNnapiSupportedDevice("test-device", /* feature_level=*/29);

  ArgMaxOpModel m2({1, 1, 1, 4}, TensorType_FLOAT32, /*axis_value=*/3,
                   TensorType_INT32, /*disallow_nnapi_cpu=*/true);
  m2.PopulateTensor<float>(m.input(), {0.1, 0.9, 0.7, 0.3});
  m2.Invoke();

  EXPECT_EQ(m2.CountOpsExecutedByCpuKernel(), 0)
      << "Expected Max op to be delegated since it is supported in NNAPI 1.2.";
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
