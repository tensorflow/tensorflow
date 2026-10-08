/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace gpu {
namespace {

// In-memory model wrapping a single RESHAPE operator.
class ReshapeOpModel : public SingleOpModel {
 public:
  ReshapeOpModel(const std::vector<int>& input_shape,
                 const std::vector<int>& output_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(output_shape))
            .Union());
    BuildInterpreter({input_shape}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false);
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor<float>(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

// Executes the Reshape model on GPU delegate and compares against linear input
// data.
void RunInputOutputLayoutTest(const std::vector<int>& input_shape,
                              const std::vector<int>& output_shape) {
  ReshapeOpModel model(input_shape, output_shape);

  auto options = TfLiteGpuDelegateOptionsV2Default();
  auto* delegate = TfLiteGpuDelegateV2Create(&options);
  if (!delegate) {
    GTEST_SKIP() << "GPU delegate not supported in this environment.";
  }

  model.SetDelegate({delegate, TfLiteGpuDelegateV2Delete});
  if (model.ApplyDelegate() != kTfLiteOk) {
    GTEST_SKIP() << "ModifyGraphWithDelegate failed (no compatible GPU).";
  }

  if (model.CountOpsExecutedByCpuKernel() != 0) {
    GTEST_SKIP()
        << "GPU delegate did not accept the RESHAPE node in this environment.";
  }

  int64_t num_elements = 1;
  for (int dim : input_shape) {
    num_elements *= dim;
  }

  // Fill input tensor with distinct linear values (1.0, 2.0, ...).
  std::vector<float> input_data(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }
  model.SetInput(input_data);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> output_data = model.GetOutput();

  // Validate element-wise equality.
  // Because RESHAPE preserves linear memory order, output_data[i] must equal
  // input_data[i].
  // In [32, 4, 2], on buggy OpenCL delegate readback (Adreno/Mali/PowerVR),
  // texels are packed Coord X = x*batch + b, which CpuCopier blindly reads
  // into host memory, transposing batch and width into [4, 32, 2] order.
  for (int64_t i = 0; i < num_elements; ++i) {
    float expected_val = input_data[i];
    float actual_val = output_data[i];
    EXPECT_EQ(actual_val, expected_val)
        << "Mismatch at linear index " << i << " (expected " << expected_val
        << ", got " << actual_val << "). On OpenCL mobile GPUs "
        << "(Adreno/Mali/PowerVR), batched tensors with C < 4 are assigned "
        << "SINGLE_TEXTURE_2D and suffer transposition between batch and "
        << "spatial dimensions during CpuCopier transfer.";
  }
}

TEST(InputOutputLayoutTest, DISABLED_FourChannelOutputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{32, 8},
                           /*output_shape=*/{32, 2, 4});
}

TEST(InputOutputLayoutTest, DISABLED_TwoChannelOutputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{32, 8},
                           /*output_shape=*/{32, 4, 2});
}

TEST(InputOutputLayoutTest, DISABLED_TwoChannelInputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{32, 4, 2},
                           /*output_shape=*/{32, 8});
}

TEST(InputOutputLayoutTest, DISABLED_OneChannelImageBatchInputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{4, 8, 8, 1},
                           /*output_shape=*/{4, 64});
}

TEST(InputOutputLayoutTest,
     DISABLED_OneChannelImageBatchOutputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{4, 64},
                           /*output_shape=*/{4, 8, 8, 1});
}

TEST(InputOutputLayoutTest,
     DISABLED_TwoChannelHeightOnlyOutputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{32, 8},
                           /*output_shape=*/{32, 4, 1, 2});
}

TEST(InputOutputLayoutTest,
     DISABLED_ThreeChannelImageBatchInputPreservesLayout) {
  RunInputOutputLayoutTest(/*input_shape=*/{4, 8, 8, 3},
                           /*output_shape=*/{4, 192});
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
