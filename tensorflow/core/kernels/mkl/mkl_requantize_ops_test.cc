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
#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include <cmath>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class MklRequantizatedOpsTest : public OpsTestBase {};

class MklRequantizatedOpsTestHelper {
 public:
  void Setup(Tensor &input_tensor_qint32, float &range_weights_ch1,
             float &range_weights_ch2);
};

void MklRequantizatedOpsTestHelper::Setup(Tensor &input_tensor_qint32,
                                          float &range_weights_ch1,
                                          float &range_weights_ch2) {
  // Step 1: Input range assumptions
  // -------------------------------
  // Assume input tensor T (NHWC) in FP32 has range [0, 5.0]   size nt*ht*wt*ct
  // Assume input filter W (NHWC) with 2 output channels of    size nw*ht*wt*2
  // logically,   filter W has 2 channels W1 and W2 each of    size nw*ht*wt*1
  // Assume input filter W1(NHWC) in FP32 has range [-2.0, 2.0]size nw*ht*wt*1
  // Assume input filter W2(NHWC) in FP32 has range [-3.0, 3.0]size nw*ht*wt*1

  // Step 2: Quantization details (per channel)
  // ------------------------------------------
  // T and W are quantized using a quantize op.
  // The input tensor T (NHWC) is quantized to unsigned int8.
  // Hence T's max value is mapped to ((2^8-1) = 255).
  // The input filter W (NHWC) is quantized to signed int8.
  // Hence W's max value is mapped to ((2^7)-1 = 127)).

  // Range of quantized T  in uint8[0  , 255] maps to orig T  in FP32[0   , 5.0]
  // Range of quantized W1 in int8[-127, 127] maps to orig W1 in FP32[-2.0, 2.0]
  // Range of quantized W2 in int8[-127, 127] maps to orig W2 in FP32[-3.0, 3.0]

  // Hence the resolution of quantized T will be 5.0/255
  // Hence the resolution of quantized W1 will be 2.0/127
  // Hence the resolution of quantized W2 will be 3.0/127

  // Step 3: Assumption of quantizedconv on quantized input&weights(per channel)
  // ---------------------------------------------------------------------------
  // The input T and weights W1 (or W2) will be convolved.
  // The output tensor T is in int32 whose range is [-2^31, 2^31).
  // For simplicity and symmetry, we truncate the above range to (-2^31, 2^31).
  // The range of convolved T*W1 is ((2^31)-1) * 5.0/255 * 2.0/127 = 663110.59
  // So the range of convolved T*W1 in int32(-2^31, 2^31) that maps to
  // orig T range in FP32[0, 5.0] * [-2.0, 2.0] is [-663110.59, 663110.59].

  // The range of convolved T*W2 is (2^31-1) * 5.0/255 * 3.0/127 = 994665.88
  // So the range of convolved T*W2 in int32(-2^31, 2^31) that maps to
  // orig T range in FP32 [0, 5.0] * [-3.0, 3.0]  is [-994665.88, 994665.88]

  // Step 4: Assumption output above is fed to requantization_range_perchannel
  // --------------------------------------------------------------------------
  // Here we recalculate the new range for convolved T*W so that we
  // make good use in int8 quantization from int32 to int8.

  // We assume the above operations are performed and use these values above
  // as ranges for requantization_range_perchannel_op.
  range_weights_ch1 = 663110.59;  // For W1 channel
  range_weights_ch2 = 994665.88;  // For W2 Channel

  // We Fill the input tensor T qint32 with arbitrary int32 values
  test::FillValues<qint32>(
      &input_tensor_qint32,
      {-1000, -2000,  2000,   4000,   -3000,  -6000,  4000,   8000,
       5000,  10000,  -6000,  -12000, 7000,   14000,  8000,   16000,
       9000,  -18000, -10000, -20000, 11000,  22000,  -12000, -24000,
       13000, 26000,  14000,  28000,  -15000, -30000, 16000,  32000});

  // Step 5: Define and run requantization_range_perchannel
  // -------------------------------------------------------
  // See test RequantizationRangePerChannelTest_Basic and/or
  // test RequantizationRangePerChannelTest_ClipMax
}

// Tests the RequantizationRangePerChannel op wherein the range
// of the weights is calculated per channel.
TEST_F(MklRequantizatedOpsTest, RequantizationRangePerChannelTest_Basic) {
  // Let us set up the tensor and inputs before we run this op.
  float clip_value_max = static_cast<float>((1L << 31) - 1);
  float range_weights_ch1 = 0.0;
  float range_weights_ch2 = 0.0;

  // Create the input tensor
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;

  // Define the shape of T.
  Tensor input_tensor_qint32(DT_QINT32,
                             {1, input_height, input_width, input_channels});

  // Explanation and setup prior to this op. Fill T and populate range values.
  MklRequantizatedOpsTestHelper helper;
  helper.Setup(input_tensor_qint32, range_weights_ch1, range_weights_ch2);

  // Step 5: Define and run requantization_range_perchannel
  // -------------------------------------------------------
  // Define, create and initialize the op in question.
  TF_ASSERT_OK(NodeDefBuilder("requantization_range_per_channel",
                              "RequantizationRangePerChannel")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Attr("clip_value_max", clip_value_max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Add the input nodes to the op.
  AddInputFromArray<qint32>(input_tensor_qint32.shape(),
                            input_tensor_qint32.flat<qint32>());

  // Calculate the min and max from the ranges
  float ch1_min = -range_weights_ch1;
  float ch1_max = range_weights_ch1;
  float ch2_min = -range_weights_ch2;
  float ch2_max = range_weights_ch2;

  // Add the perchannel range Nodes to the op.
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_min, ch2_min});
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_max, ch2_max});

  // Run the kernel
  TF_ASSERT_OK(RunOpKernel());

  // Step 6: Verify output and store values to test requantize_perchannel
  // --------------------------------------------------------------------

  // Verify the Expected Outputs
  const float output_min = GetOutput(0)->scalar<float>()();
  const float output_max = GetOutput(1)->scalar<float>()();
  EXPECT_NEAR(-14.8217, output_min, 0.002);
  EXPECT_NEAR(14.8217, output_max, 0.002);

  // Output range is made use in RequantizePerChannelTest_Basic
}

TEST_F(MklRequantizatedOpsTest, RequantizationRangePerChannelTest_ClipMax) {
  // Let us setup the tensor and inputs before we run this op.
  float clip_value_max = 6;  // Can be used as 6 for Relu 6 activations.
  float range_weights_ch1 = 0.0;
  float range_weights_ch2 = 0.0;

  // Create the input tensor
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;

  // define and input tensor T shape.
  Tensor input_tensor_qint32(DT_QINT32,
                             {1, input_height, input_width, input_channels});

  // Explanation and setup prior to this op. Fill T and populate range values.
  MklRequantizatedOpsTestHelper helper;
  helper.Setup(input_tensor_qint32, range_weights_ch1, range_weights_ch2);

  // Step 5: Define and run requantization_range_perchannel
  // -------------------------------------------------------
  // Define, create and initialize the op in question.
  TF_ASSERT_OK(NodeDefBuilder("requantization_range_per_channel",
                              "RequantizationRangePerChannel")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Attr("clip_value_max", clip_value_max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Add the input nodes to the op.
  AddInputFromArray<qint32>(input_tensor_qint32.shape(),
                            input_tensor_qint32.flat<qint32>());

  // Calculate the min and max from the ranges
  float ch1_min = -range_weights_ch1;
  float ch1_max = range_weights_ch1;
  float ch2_min = -range_weights_ch2;
  float ch2_max = range_weights_ch2;

  // Add the perchannel range nodes to the op.
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_min, ch2_min});
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_max, ch2_max});

  // Run the kernel
  TF_ASSERT_OK(RunOpKernel());

  // Step 6: Verify output and store values to test requantize_perchannel
  // --------------------------------------------------------------------

  // Verify the expected outputs
  const float output_min = GetOutput(0)->scalar<float>()();
  const float output_max = GetOutput(1)->scalar<float>()();
  EXPECT_NEAR(-6.0, output_min, 0.002);  // Values are aligned with clip_value.
  EXPECT_NEAR(6.0, output_max, 0.002);   // Values are aligned with clip_value.
}

TEST_F(MklRequantizatedOpsTest, RequantizePerChannelTest_Basic) {
  // Let us setup the tensor and inputs before we run this op.
  float range_weights_ch1 = 0.0;
  float range_weights_ch2 = 0.0;

  // Create the input tensor
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;

  // define an input tensor T shape.
  Tensor input_tensor_qint32(DT_QINT32,
                             {1, input_height, input_width, input_channels});

  // Explanation and setup prior to this op. Fill T and populate range values.
  MklRequantizatedOpsTestHelper helper;
  helper.Setup(input_tensor_qint32, range_weights_ch1, range_weights_ch2);

  // Step 7: Define and run requantize_perchannel
  // --------------------------------------------
  // The output of requantization_range_op_per_channel which calculated the
  // new ranges of int8 is fed to the requantize per channel op.
  // Here the values of convolved T*W is converted from int32 to int8.

  TF_ASSERT_OK(NodeDefBuilder("requantize_per_channel", "RequantizePerChannel")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Attr("out_type", DataTypeToEnum<qint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Add the input Nodes to the op.
  AddInputFromArray<qint32>(input_tensor_qint32.shape(),
                            input_tensor_qint32.flat<qint32>());

  // Calculate the min and max from the ranges
  float ch1_min = -range_weights_ch1;
  float ch1_max = range_weights_ch1;
  float ch2_min = -range_weights_ch2;
  float ch2_max = range_weights_ch2;

  // Add the perchannel range nodes to the op.
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_min, ch2_min});
  AddInputFromArray<float>(TensorShape({input_channels}), {ch1_max, ch2_max});

  // Calculate the min and max from Step 6 above
  // in RequantizationRangePerChannelTest_Basic
  float range_op_output_min = -14.8217;
  float range_op_output_max = 14.8217;

  // Add the requested_min and requested_max stored from Step 6.
  AddInputFromArray<float>(TensorShape({}), {range_op_output_min});
  AddInputFromArray<float>(TensorShape({}), {range_op_output_max});

  // Run the kernel
  TF_ASSERT_OK(RunOpKernel());

  // Verify the output with the expected output
  Tensor output = *GetOutput(0);
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  EXPECT_NEAR(range_op_output_min, output_min, 0.002);
  EXPECT_NEAR(range_op_output_max, output_max, 0.002);
}

}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
