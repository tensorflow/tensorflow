/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

using tensorflow::AllocatorAttributes;
using tensorflow::DT_FLOAT;
using tensorflow::NodeDefBuilder;
using tensorflow::OpsTestBase;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::test::ExpectClose;
using tensorflow::test::FillValues;

class QuantOpsTest : public OpsTestBase {
 protected:
  void AddRandomInput(const TensorShape& shape) {
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DT_FLOAT, shape);
    input->flat<float>().setRandom();
    tensors_.push_back(input);
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), DT_FLOAT);
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DT_FLOAT);
      inputs_.push_back({nullptr, input});
    }
  }

  void RunTestFakeQuantWithMinMaxArgs(const int num_bits,
                                      const bool narrow_range, const float min,
                                      const float max, const TensorShape& shape,
                                      const gtl::ArraySlice<float> data,
                                      gtl::ArraySlice<float> expected_data) {
    TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxArgs")
                     .Input(FakeInput(DT_FLOAT))  // inputs
                     .Attr("min", min)
                     .Attr("max", max)
                     .Attr("num_bits", num_bits)
                     .Attr("narrow_range", narrow_range)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    // Downstream inputs.
    AddInputFromArray<float>(shape, data);

    // Tested code.
    TF_ASSERT_OK(RunOpKernel());

    Tensor* output = GetOutput(0);
    Tensor expected(allocator(), DT_FLOAT, shape);
    FillValues<float>(&expected, expected_data);
    ExpectClose(expected, *output);
  }

  void RunTestFakeQuantWithMinMaxVars(const int num_bits,
                                      const bool narrow_range, const float min,
                                      const float max, const TensorShape& shape,
                                      const gtl::ArraySlice<float> data,
                                      gtl::ArraySlice<float> expected_data) {
    TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVars")
                     .Input(FakeInput(DT_FLOAT))  // inputs
                     .Input(FakeInput(DT_FLOAT))  // min
                     .Input(FakeInput(DT_FLOAT))  // max
                     .Attr("num_bits", num_bits)
                     .Attr("narrow_range", narrow_range)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    // Downstream inputs.
    AddInputFromArray<float>(shape, data);
    // Min.
    AddInputFromArray<float>(TensorShape({}), {min});
    // Max.
    AddInputFromArray<float>(TensorShape({}), {max});

    // Tested code.
    TF_ASSERT_OK(RunOpKernel());

    Tensor* output = GetOutput(0);
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
    FillValues<float>(&expected, expected_data);
    ExpectClose(expected, *output);
  }

  void RunTestFakeQuantWithMinMaxVarsPerChannel(
      const int num_bits, const bool narrow_range,
      const TensorShape& minmax_shape, const gtl::ArraySlice<float> min,
      const gtl::ArraySlice<float> max, const TensorShape& shape,
      const gtl::ArraySlice<float> data, gtl::ArraySlice<float> expected_data) {
    TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannel")
                     .Input(FakeInput(DT_FLOAT))  // inputs
                     .Input(FakeInput(DT_FLOAT))  // min
                     .Input(FakeInput(DT_FLOAT))  // max
                     .Attr("num_bits", num_bits)
                     .Attr("narrow_range", narrow_range)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    // Downstream inputs.
    AddInputFromArray<float>(shape, data);
    // Min.
    AddInputFromArray<float>(minmax_shape, min);
    // Max.
    AddInputFromArray<float>(minmax_shape, max);

    // Tested code.
    TF_ASSERT_OK(RunOpKernel());

    Tensor* output = GetOutput(0);
    Tensor expected(allocator(), DT_FLOAT, shape);
    FillValues<float>(&expected, expected_data);
    ExpectClose(expected, *output);
  }
};

TEST_F(QuantOpsTest, WithArgsNoNudging_RegularRange) {
  // Original quantization range: [-10 + 0 / 4, -10 + 255 / 4], scale: 1/4.
  // Original zero point: 40, no nudging necessary.
  // Expected quantized values: -10.0, -9.75, ..., 53.75.
  RunTestFakeQuantWithMinMaxArgs(
      8, false, -10.0f, 53.75f, TensorShape({2, 3}),
      {-10.1f, -10.0f, -9.9f, -9.75f, 53.75f, 53.8f},
      {-10.0f, -10.0f, -10.0f, -9.75f, 53.75f, 53.75f});
}

TEST_F(QuantOpsTest, WithArgsNoNudging_NarrowRange) {
  // Original quantization range: [-10 + 0 / 4, -10 + 254 / 4], scale: 1/4.
  // Original zero point: 41, no nudging necessary.
  // Expected quantized values: -10.0, -9.75, ..., 53.5.
  RunTestFakeQuantWithMinMaxArgs(
      8, true, -10.0f, 53.5f, TensorShape({2, 3}),
      {-10.1f, -10.0f, -9.9f, -9.75f, 53.5f, 53.6f},
      {-10.0f, -10.0f, -10.0f, -9.75f, 53.5f, 53.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_RegularRange) {
  // Original quantization range: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged range: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  RunTestFakeQuantWithMinMaxArgs(8, false, -0.1f, 63.65f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.25f, 63.75f, 63.8f},
                                 {0.0f, 0.0f, 0.0f, 0.25f, 63.75f, 63.75f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_NarrowRange) {
  // Original quantization range: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged range: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  RunTestFakeQuantWithMinMaxArgs(8, true, -0.1f, 63.4f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.25f, 63.5f, 63.6f},
                                 {0.0f, 0.0f, 0.0f, 0.25f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_RegularRange) {
  // Original quantization range: [-0.51 / 4 + 0 / 4, -0.51 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.51, nudged to 1.
  // Nudged range: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  RunTestFakeQuantWithMinMaxArgs(8, false, -0.1275f, 63.6225f,
                                 TensorShape({2, 3}),
                                 {-0.26f, -0.25f, -0.24f, 0.0f, 63.5f, 63.6f},
                                 {-0.25f, -0.25f, -0.25f, 0.0f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_NarrowRange) {
  // Original quantization range: [-0.51 / 4 + 0 / 4, -0.51 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.51, nudged to 2.
  // Nudged range: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  RunTestFakeQuantWithMinMaxArgs(
      8, true, -0.1275f, 63.3725f, TensorShape({2, 3}),
      {-0.26f, -0.25f, -0.24f, 0.0f, 63.25f, 63.3f},
      {-0.25f, -0.25f, -0.25f, 0.0f, 63.25f, 63.25f});
}

TEST_F(QuantOpsTest, WithArgsNudgedZeroIs255_RegularRange) {
  // Original quantization range: [0.4 / 4 - 255 / 4, 0.4 / 4 + 0 / 4].
  // Scale: 1/4,  original zero point: 254.6, nudged to 255.
  // Nudged range: [-63.75; 0.0].
  // Expected quantized values: -63.75, -63.5, -63.25, ..., 0.0.
  RunTestFakeQuantWithMinMaxArgs(
      8, false, -63.65f, 0.1f, TensorShape({2, 3}),
      {-63.8f, -63.75f, -63.7f, -63.5f, 0.0f, 0.1f},
      {-63.75f, -63.75f, -63.75f, -63.5f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedZeroIs255_NarrowRange) {
  // Original quantization range: [0.4 / 4 - 254 / 4, 0.4 / 4 + 0 / 4].
  // Scale: 1/4,  original zero point: 254.6, nudged to 255.
  // Nudged range: [-63.5; 0.0].
  // Expected quantized values: -63.5, -63.25, -63.0, ..., 0.0.
  RunTestFakeQuantWithMinMaxArgs(8, true, -63.4f, 0.1f, TensorShape({2, 3}),
                                 {-63.6f, -63.5f, -63.4f, -63.25f, 0.0f, 0.1f},
                                 {-63.5f, -63.5f, -63.5f, -63.25f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithArgsNoNudging_4Bits_RegularRange) {
  // Original quantization range: [-6 + 0 / 2, -6 + 15 / 2], scale: 1/2.
  // Original zero point: 12, no nudging necessary.
  // Expected quantized values: -6, -5.5, ..., 1.5.
  RunTestFakeQuantWithMinMaxArgs(4, false, -6.0f, 1.5f, TensorShape({2, 3}),
                                 {-6.1f, -6.0f, -5.9f, -5.5f, 1.5f, 1.6f},
                                 {-6.0f, -6.0f, -6.0f, -5.5f, 1.5f, 1.5f});
}

TEST_F(QuantOpsTest, WithArgsNoNudging_4Bits_NarrowRange) {
  // Original quantization range: [-6 + 0 / 2, -6 + 14 / 2], scale: 1/2.
  // Original zero point: 13, no nudging necessary.
  // Expected quantized values: -6, -5.5, ..., 1.0.
  RunTestFakeQuantWithMinMaxArgs(4, true, -6.0f, 1.0f, TensorShape({2, 3}),
                                 {-6.1f, -6.0f, -5.9f, -5.5f, 1.0f, 1.1f},
                                 {-6.0f, -6.0f, -6.0f, -5.5f, 1.0f, 1.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  RunTestFakeQuantWithMinMaxArgs(4, false, -0.1f, 7.4f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.5f, 7.5f, 7.6f},
                                 {0.0f, 0.0f, 0.0f, 0.5f, 7.5f, 7.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxArgs(4, true, -0.1f, 6.9f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.5f, 7.0f, 7.1f},
                                 {0.0f, 0.0f, 0.0f, 0.5f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxArgs(4, false, -0.4f, 7.1f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 7.0f, 7.1f},
                                 {-0.5f, -0.5f, -0.00f, 0.0f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  RunTestFakeQuantWithMinMaxArgs(4, true, -0.4f, 6.6f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 6.5f, 6.6f},
                                 {-0.5f, -0.5f, 0.0f, 0.0f, 6.5f, 6.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedZeroIs15_4Bits_RegularRange) {
  // Original quantization range: [0.4 / 2 - 15 / 2, 0.4 / 2 + 0 / 2].
  // Scale: 1/2,  original zero point: 14.6, nudged to 15.
  // Nudged range: [-7.5; 0.0].
  // Expected quantized values: -7.5, -7.0, ..., 0.0.
  RunTestFakeQuantWithMinMaxArgs(4, false, -7.3f, 0.2f, TensorShape({2, 3}),
                                 {-7.6f, -7.5f, -7.4f, -7.2f, 0.0f, 0.1f},
                                 {-7.5f, -7.5f, -7.5f, -7.0f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedZeroIs15_4Bits_NarrowRange) {
  // Original quantization range: [0.4 / 2 - 14 / 2, 0.4 / 2 + 0 / 2].
  // Scale: 1/2,  original zero point: 14.6, nudged to 15.
  // Nudged range: [-7.0; 0.0].
  // Expected quantized values: -7.0, -6.5, ..., 0.0.
  RunTestFakeQuantWithMinMaxArgs(4, true, -6.8f, 0.2f, TensorShape({2, 3}),
                                 {-7.1f, -7.0f, -6.9f, -6.7f, 0.0f, 0.1f},
                                 {-7.0f, -7.0f, -7.0f, -6.5f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithArgsNoNudging_2Bits_RegularRange) {
  // Original quantization range: [-1 + 0 / 2, -1 + 3 / 2], scale: 1/2.
  // Original zero point: 2, no nudging necessary.
  // Expected quantized values: -1.0, -0.5, 0.0, 0.5.
  RunTestFakeQuantWithMinMaxArgs(2, false, -1.0f, 0.5f, TensorShape({2, 3}),
                                 {-1.1f, -1.0f, -0.9f, -0.3f, 0.1f, 0.6f},
                                 {-1.0f, -1.0f, -1.0f, -0.5f, 0.0f, 0.5f});
}

TEST_F(QuantOpsTest, WithArgsNoNudging_2Bits_NarrowRange) {
  // Original quantization range: [-1 + 0 / 2, -1 + 2 / 2], scale: 1/2.
  // Original zero point: 3, no nudging necessary.
  // Expected quantized values: -1.0, -0.5, 0.0.
  RunTestFakeQuantWithMinMaxArgs(2, true, -1.0f, 0.0f, TensorShape({2, 3}),
                                 {-1.1f, -1.0f, -0.9f, -0.3f, 0.0f, 0.1f},
                                 {-1.0f, -1.0f, -1.0f, -0.5f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_2Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 3 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 1.5].
  // Expected quantized values: 0.0, 0.5, 1.0, 1.5.
  RunTestFakeQuantWithMinMaxArgs(2, false, -0.1f, 1.4f, TensorShape({2, 3}),
                                 {-0.2f, 0.1f, 0.7f, 1.0f, 1.3f, 1.6f},
                                 {0.0f, 0.0f, 0.5f, 1.0f, 1.5f, 1.5f});
}

TEST_F(QuantOpsTest, WithArgsNudgedDown_2Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 2 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 1.0].
  // Expected quantized values: 0.0, 0.5, 1.0.
  RunTestFakeQuantWithMinMaxArgs(2, true, -0.1f, 0.9f, TensorShape({2, 3}),
                                 {-0.1f, 0.1f, 0.7f, 0.9f, 1.0f, 1.1f},
                                 {-0.0f, 0.0f, 0.5f, 1.0f, 1.0f, 1.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_2Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 3 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 1.0].
  // Expected quantized values: -0.5, 0.0, 0.5, 1.0.
  RunTestFakeQuantWithMinMaxArgs(2, false, -0.4f, 1.1f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 1.0f, 1.1f},
                                 {-0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 1.0f});
}

TEST_F(QuantOpsTest, WithArgsNudgedUp_2Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 2 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 0.5].
  // Expected quantized values: -0.5, 0.0, 0.5.
  RunTestFakeQuantWithMinMaxArgs(2, true, -0.4f, 0.6f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 0.5f, 0.6f},
                                 {-0.5f, -0.5f, -0.00f, 0.0f, 0.5f, 0.5f});
}

TEST_F(QuantOpsTest, WithArgsGradient_RegularRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged range: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxArgsGradient")
                   .Input(FakeInput(DT_FLOAT))  // gradient
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Attr("min", -0.125f)
                   .Attr("max", 63.625f)
                   .Attr("narrow_range", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.26f, -0.25f, -0.24f, 0.0f, 63.5f, 63.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  auto input_flat = GetInput(0).flat<float>();
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  FillValues<float>(&expected, {0.0f, input_flat(1), input_flat(2),
                                input_flat(3), input_flat(4), 0.0f});
  ExpectClose(expected, *output);
}

TEST_F(QuantOpsTest, WithArgsGradient_NarrowRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged range: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxArgsGradient")
                   .Input(FakeInput(DT_FLOAT))  // gradient
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Attr("min", -0.125f)
                   .Attr("max", 63.375f)
                   .Attr("narrow_range", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.26f, -0.25f, -0.24f, 0.0f, 63.25f, 63.3f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  auto input_flat = GetInput(0).flat<float>();
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  FillValues<float>(&expected, {0.0f, input_flat(1), input_flat(2),
                                input_flat(3), input_flat(4), 0.0f});
  ExpectClose(expected, *output);
}

TEST_F(QuantOpsTest, WithArgsGradient_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxArgsGradient")
                   .Input(FakeInput(DT_FLOAT))  // gradient
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Attr("min", -0.4f)
                   .Attr("max", 7.1f)
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 7.0f, 7.1f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  auto input_flat = GetInput(0).flat<float>();
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  FillValues<float>(&expected, {0.0f, input_flat(1), input_flat(2),
                                input_flat(3), input_flat(4), 0.0f});
  ExpectClose(expected, *output);
}

TEST_F(QuantOpsTest, WithArgsGradient_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxArgsGradient")
                   .Input(FakeInput(DT_FLOAT))  // gradient
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Attr("min", -0.4f)
                   .Attr("max", 6.6f)
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 6.5f, 6.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  auto input_flat = GetInput(0).flat<float>();
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  FillValues<float>(&expected, {0.0f, input_flat(1), input_flat(2),
                                input_flat(3), input_flat(4), 0.0f});
  ExpectClose(expected, *output);
}

TEST_F(QuantOpsTest, WithVars_ZeroMinAndMax) {
  RunTestFakeQuantWithMinMaxVars(8, false, 0.0f, 0.0f, TensorShape({2, 3}),
                                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsNoNudging_RegularRange) {
  // Original quantization range: [-10 + 0 / 4, -10 + 255 / 4], scale: 1/4.
  // Original zero point: 40, no nudging necessary.
  // Expected quantized values: -10.0, -10.25, ..., 53.75.
  RunTestFakeQuantWithMinMaxVars(
      8, false, -10.0f, 53.75f, TensorShape({2, 3}),
      {-10.1f, -10.0f, -9.9f, -9.75f, 53.75f, 53.8f},
      {-10.0f, -10.0f, -10.0f, -9.75f, 53.75f, 53.75f});
}

TEST_F(QuantOpsTest, WithVarsNoNudging_NarrowRange) {
  // Original quantization range: [-10 + 0 / 4, -10 + 254 / 4], scale: 1/4.
  // Original zero point: 41, no nudging necessary.
  // Expected quantized values: -10.0, -10.25, ..., 53.5.
  RunTestFakeQuantWithMinMaxVars(
      8, true, -10.0f, 53.5f, TensorShape({2, 3}),
      {-10.1f, -10.0f, -9.90f, -9.75f, 53.5f, 53.6f},
      {-10.0f, -10.0f, -10.0f, -9.75f, 53.5f, 53.5f});
}

TEST_F(QuantOpsTest, WithVarsNudgedDown_RegularRange) {
  // Original quantization range: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged range: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  RunTestFakeQuantWithMinMaxVars(8, false, -0.1f, 63.65f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.25f, 63.75f, 63.8f},
                                 {-0.0f, 0.0f, 0.0f, 0.25f, 63.75f, 63.75f});
}

TEST_F(QuantOpsTest, WithVarsNudgedDown_NarrowRange) {
  // Original quantization range: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged range: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  RunTestFakeQuantWithMinMaxVars(8, true, -0.1f, 63.4f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.25f, 63.5f, 63.6f},
                                 {-0.0f, 0.0f, 0.0f, 0.25f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsNudgedUp_RegularRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged range: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  RunTestFakeQuantWithMinMaxVars(8, false, -0.125f, 63.625f,
                                 TensorShape({2, 3}),
                                 {-0.26f, -0.25f, -0.24f, 0.0f, 63.5f, 63.6f},
                                 {-0.25f, -0.25f, -0.25f, 0.0f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsNudgedUp_NarrowRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged range: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  RunTestFakeQuantWithMinMaxVars(
      8, true, -0.125f, 63.375f, TensorShape({2, 3}),
      {-0.26f, -0.25f, -0.24f, 0.0f, 63.25f, 63.3f},
      {-0.25f, -0.25f, -0.25f, 0.0f, 63.25f, 63.25f});
}

TEST_F(QuantOpsTest, WithVarsNudgedZeroIs255_RegularRange) {
  // Original quantization range: [0.4 / 4 - 255 / 4, 0.4 / 4 + 0 / 4].
  // Scale: 1/4,  original zero point: 254.6, nudged to 255.
  // Nudged range: [-63.75; 0.0].
  // Expected quantized values: -63.75, -63.5, -63.25, ..., 0.0.
  RunTestFakeQuantWithMinMaxVars(
      8, false, -63.65f, 0.1f, TensorShape({2, 3}),
      {-63.80f, -63.75f, -63.70f, -63.5f, 0.0f, 0.1f},
      {-63.75f, -63.75f, -63.75f, -63.5f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsNudgedZeroIs255_NarrowRange) {
  // Original quantization range: [0.4 / 4 - 254 / 4, 0.4 / 4 + 0 / 4].
  // Scale: 1/4,  original zero point: 254.6, nudged to 255.
  // Nudged range: [-63.5; 0.0].
  // Expected quantized values: -63.5, -63.25, -63.0, ..., 0.0.
  RunTestFakeQuantWithMinMaxVars(8, true, -63.4f, 0.1f, TensorShape({2, 3}),
                                 {-63.6f, -63.5f, -63.4f, -63.25f, 0.0f, 0.1f},
                                 {-63.5f, -63.5f, -63.5f, -63.25f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsNoNudging_4Bits_RegularRange) {
  // Original quantization range: [-6 + 0 / 2, -6 + 15 / 2], scale: 1/2.
  // Original zero point: 12, no nudging necessary.
  // Expected quantized values: -6, -5.5, ..., 1.5.
  RunTestFakeQuantWithMinMaxVars(4, false, -6.0f, 1.5f, TensorShape({2, 3}),
                                 {-6.1f, -6.0f, -5.9f, -5.5f, 1.5f, 1.6f},
                                 {-6.0f, -6.0f, -6.0f, -5.5f, 1.5f, 1.5f});
}

TEST_F(QuantOpsTest, WithVarsNoNudging_4Bits_NarrowRange) {
  // Original quantization range: [-6 + 0 / 2, -6 + 14 / 2], scale: 1/2.
  // Original zero point: 13, no nudging necessary.
  // Expected quantized values: -6, -5.5, ..., 1.0.
  RunTestFakeQuantWithMinMaxVars(4, true, -6.0f, 1.0f, TensorShape({2, 3}),
                                 {-6.1f, -6.0f, -5.9f, -5.5f, 1.0f, 1.1f},
                                 {-6.0f, -6.0f, -6.0f, -5.5f, 1.0f, 1.0f});
}

TEST_F(QuantOpsTest, WithVarsNudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  RunTestFakeQuantWithMinMaxVars(4, false, -0.1f, 7.4f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.5f, 7.5f, 7.6f},
                                 {-0.0f, 0.0f, 0.0f, 0.5f, 7.5f, 7.5f});
}

TEST_F(QuantOpsTest, WithVarsNudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVars(4, true, -0.1f, 6.9f, TensorShape({2, 3}),
                                 {-0.1f, 0.0f, 0.1f, 0.5f, 7.0f, 7.1f},
                                 {-0.0f, 0.0f, 0.0f, 0.5f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsNudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVars(4, false, -0.4f, 7.1f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 7.0f, 7.1f},
                                 {-0.5f, -0.5f, -0.00f, 0.0f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsNudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  RunTestFakeQuantWithMinMaxVars(4, true, -0.4f, 6.6f, TensorShape({2, 3}),
                                 {-0.6f, -0.5f, -0.24f, 0.0f, 6.5f, 6.6f},
                                 {-0.5f, -0.5f, -0.00f, 0.0f, 6.5f, 6.5f});
}

TEST_F(QuantOpsTest, WithVarsNudgedZero15_4Bits_RegularRange) {
  // Original quantization range: [0.4 / 2 - 15 / 2, 0.4 / 2 + 0 / 2].
  // Scale: 1/2,  original zero point: 14.6, nudged to 15.
  // Nudged range: [-7.5; 0.0].
  // Expected quantized values: -7.5, -7.0, ..., 0.0.
  RunTestFakeQuantWithMinMaxVars(4, false, -7.3f, 0.2f, TensorShape({2, 3}),
                                 {-7.6f, -7.5f, -7.4f, -7.2f, 0.0f, 0.1f},
                                 {-7.5f, -7.5f, -7.5f, -7.0f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsNudgedZero15_4Bits_NarrowRange) {
  // Original quantization range: [0.4 / 2 - 14 / 2, 0.4 / 2 + 0 / 2].
  // Scale: 1/2,  original zero point: 14.6, nudged to 15.
  // Nudged range: [-7.0; 0.0].
  // Expected quantized values: -7.0, -6.5, ..., 0.0.
  RunTestFakeQuantWithMinMaxVars(4, true, -6.8f, 0.2f, TensorShape({2, 3}),
                                 {-7.1f, -7.0f, -6.9f, -6.5f, 0.0f, 0.1f},
                                 {-7.0f, -7.0f, -7.0f, -6.5f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsGradient_ZeroMinAndMax) {
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  // Min.
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  // Max.
  AddInputFromArray<float>(TensorShape({}), {0.0f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto in_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {in_flat(0), in_flat(1), in_flat(2), in_flat(3), in_flat(4), in_flat(5)});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_min.flat<float>()(0) = 0.0f;
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_max.flat<float>()(0) = 0.0f;
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsGradient_RegularRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged range: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.26f, -0.25f, -0.24f, 0.0f, 63.5f, 63.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({}), {-0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({}), {63.625f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto in_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input, {0.0f, in_flat(1), in_flat(2),
                                                in_flat(3), in_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_min.flat<float>()(0) = in_flat(0);
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_max.flat<float>()(0) = in_flat(5);
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsGradient_NarrowRange) {
  // Original quantization range: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged range: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.26f, -0.25f, -0.24f, 0.0f, 63.25f, 63.3f});
  // Min.
  AddInputFromArray<float>(TensorShape({}), {-0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({}), {63.375f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto in_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input, {0.0f, in_flat(1), in_flat(2),
                                                in_flat(3), in_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_min.flat<float>()(0) = in_flat(0);
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_max.flat<float>()(0) = in_flat(5);
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsGradient_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 7.0f, 7.1f});
  // Min.
  AddInputFromArray<float>(TensorShape({}), {-0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({}), {7.1f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto in_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input, {0.0f, in_flat(1), in_flat(2),
                                                in_flat(3), in_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_min.flat<float>()(0) = in_flat(0);
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_max.flat<float>()(0) = in_flat(5);
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsGradient_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 6.5f, 6.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({}), {-0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({}), {6.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto in_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input, {0.0f, in_flat(1), in_flat(2),
                                                in_flat(3), in_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_min.flat<float>()(0) = in_flat(0);
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({}));
  expected_bprop_wrt_max.flat<float>()(0) = in_flat(5);
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannel_ZeroMinAndMax) {
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false, TensorShape({4}), {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f}, TensorShape({4}), {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false, TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {63.65f, 63.65f, 63.65f, 63.65f}, TensorShape({4}),
      {-0.1f, 0.0f, 63.75f, 63.8f}, {0.0f, 0.0f, 63.75f, 63.75f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true, TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {63.4f, 63.4f, 63.4f, 63.4f}, TensorShape({4}),
      {-0.1f, 0.0f, 63.5f, 63.6f}, {0.0f, 0.0f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false, TensorShape({4}), {-0.125f, -0.125f, -0.125f, -0.125f},
      {63.625f, 63.625f, 63.625f, 63.625f}, TensorShape({4}),
      {-0.26f, -0.25f, -0.24f, 63.6f}, {-0.25f, -0.25f, -0.25f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true, TensorShape({4}), {-0.125f, -0.125f, -0.125f, -0.125f},
      {63.375f, 63.375f, 63.375f, 63.375f}, TensorShape({4}),
      {-0.26f, -0.25f, -0.24f, 63.3f}, {-0.25f, -0.25f, -0.25f, 63.25f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false, TensorShape({3}), {-0.1f, -0.1f, -0.1f},
      {63.65f, 63.65f, 63.65f}, TensorShape({2, 3}),
      {-0.1f, 0.0f, 0.1f, 0.25f, 63.75f, 63.80f},
      {-0.0f, 0.0f, 0.0f, 0.25f, 63.75f, 63.75f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true, TensorShape({3}), {-0.1f, -0.1f, -0.1f}, {63.4f, 63.4f, 63.4f},
      TensorShape({2, 3}), {-0.1f, 0.0f, 0.1f, 0.25f, 63.5f, 63.6f},
      {0.0f, 0.0f, 0.0f, 0.25f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false, TensorShape({3}), {-0.125f, -0.125f, -0.125f},
      {63.625f, 63.625f, 63.625f}, TensorShape({2, 3}),
      {-0.26f, -0.25f, -0.24f, 0.0f, 63.5f, 63.6f},
      {-0.25f, -0.25f, -0.25f, 0.0f, 63.5f, 63.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true, TensorShape({3}), {-0.125f, -0.125f, -0.125f},
      {63.375f, 63.375f, 63.375f}, TensorShape({2, 3}),
      {-0.26f, -0.25f, -0.24f, 0.0f, 63.25f, 63.3f},
      {-0.25f, -0.25f, -0.25f, 0.0f, 63.25f, 63.25f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false,
      TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {63.65f, 63.65f, 63.65f, 63.65f},
      TensorShape({1, 2, 3, 4}),
      {-0.1f,   0.0f,   0.1f,   0.25f,  0.5f,    0.75f,
        1.0f,   1.25f,  1.5f,   1.75f,  2.0f,    2.25f,
       63.0f,  63.25f, 63.5f,  63.7f,  63.75f,  63.8f,
       63.9f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { 0.0f,   0.0f,   0.0f,   0.25f,  0.5f,    0.75f,
        1.0f,   1.25f,  1.5f,   1.75f,  2.0f,    2.25f,
       63.0f,  63.25f, 63.5f,  63.75f, 63.75f,  63.75f,
       63.75f, 63.75f, 63.75f, 63.75f, 63.75f,  63.75f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true,
      TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {63.4f, 63.4f, 63.4f, 63.4f},
      TensorShape({1, 2, 3, 4}),
      {-0.1f,   0.0f,   0.1f,   0.25f,  0.5f,    0.75f,
        1.0f,   1.25f,  1.5f,   1.75f,  2.0f,    2.25f,
       63.0f,  63.25f, 63.3f,  63.4f,  63.5f,   63.6f,
       63.7f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { 0.0f,   0.0f,   0.0f,   0.25f,  0.5f,    0.75f,
        1.0f,   1.25f,  1.5f,   1.75f,  2.0f,    2.25f,
       63.0f,  63.25f, 63.25f, 63.5f,  63.5f,   63.5f,
       63.5f,  63.5f,  63.5f,  63.5f,  63.5f,   63.5f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, false,
      TensorShape({4}), {-0.125f, -0.125f, -0.125f, -0.125f},
      {63.625f, 63.625f, 63.625f, 63.625f},
      TensorShape({1, 2, 3, 4}),
      { -0.3f,  -0.25f, -0.2f,   0.0f,    0.25f,  0.5f,
         0.75f,  1.0f,   1.25f,  1.5f,    1.75f,  2.0f,
        63.0f,  63.25f, 63.4f,  63.5f,   63.6f,  63.7f,
       100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      {-0.25f,  -0.25f, -0.25f,  0.0f,   0.25f,   0.5f,
        0.75f,   1.0f,   1.25f,  1.5f,   1.75f,   2.0f,
        63.0f,  63.25f, 63.5f,  63.5f,  63.5f,   63.5f,
        63.5f,  63.5f,  63.5f,  63.5f,  63.5f,   63.5f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      8, true,
      TensorShape({4}), {-0.125f, -0.125f, -0.125f, -0.125f},
      {63.375f, 63.375f, 63.375f, 63.375f},
      TensorShape({1, 2, 3, 4}),
      { -0.3f,  -0.25f, -0.2f,   0.0f,   0.25f,   0.5f,
         0.75f,  1.0f,   1.25f,  1.5f,   1.75f,   2.0f,
        63.0f,  63.2f,  63.25f, 63.3f,  63.4f,   63.5f,
       100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { -0.25f, -0.25f, -0.25f,  0.0f,   0.25f,   0.5f,
         0.75f,  1.0f,   1.25f,  1.5f,   1.75f,   2.0f,
        63.0f,  63.25f, 63.25f, 63.25f, 63.25f,  63.25f,
        63.25f, 63.25f, 63.25f, 63.25f, 63.25f,  63.25f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false, TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {7.4f, 7.4f, 7.4f, 7.4f}, TensorShape({4}), {-0.1f, 0.0f, 7.5f, 7.6f},
      {0.0f, 0.0f, 7.5f, 7.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true, TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f},
      {6.9f, 6.9f, 6.9f, 6.9f}, TensorShape({4}), {-0.1f, 0.0f, 7.0f, 7.1f},
      {0.0f, 0.0f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false, TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f},
      {7.1f, 7.1f, 7.1f, 7.1f}, TensorShape({4}), {-0.6f, -0.5f, 7.0f, 7.1f},
      {-0.5f, -0.5f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1NudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true, TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f},
      {6.6f, 6.6f, 6.6f, 6.6f}, TensorShape({4}), {-0.6f, -0.5f, 6.5f, 6.6f},
      {-0.5f, -0.5f, 6.5f, 6.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false, TensorShape({3}), {-0.1f, -0.1f, -0.1f}, {7.4f, 7.4f, 7.4f},
      TensorShape({2, 3}), {-0.1f, 0.0f, 0.1f, 0.5f, 7.5f, 7.6f},
      {0.0f, 0.0f, 0.0f, 0.5f, 7.5f, 7.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true, TensorShape({3}), {-0.1f, -0.1f, -0.1f}, {6.9f, 6.9f, 6.9f},
      TensorShape({2, 3}), {-0.1f, 0.0f, 0.1f, 0.5f, 7.0f, 7.1f},
      {0.0f, 0.0f, 0.0f, 0.5f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false, TensorShape({3}), {-0.4f, -0.4f, -0.4f}, {7.1f, 7.1f, 7.1f},
      TensorShape({2, 3}), {-0.51f, -0.5f, -0.24f, 0.0f, 7.0f, 7.1f},
      {-0.5f, -0.5f, 0.0f, 0.0f, 7.0f, 7.0f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2NudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true, TensorShape({3}), {-0.4f, -0.4f, -0.4f}, {6.6f, 6.6f, 6.6f},
      TensorShape({2, 3}), {-0.6f, -0.5f, -0.24f, 0.0f, 6.5f, 6.6f},
      {-0.5f, -0.5f, 0.0f, 0.0f, 6.5f, 6.5f});
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false,
      TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f}, {7.4f, 7.4f, 7.4f, 7.4f},
      TensorShape({1, 2, 3, 4}),
      {-0.1f,   0.0f,   0.1f,   0.5f,   1.0f,    1.5f,
        1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
        6.0f,   6.5f,   7.0f,   7.4f,   7.5f,    7.7f,
        7.8f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { 0.0f,   0.0f,   0.0f,   0.5f,   1.0f,    1.5f,
        1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
        6.0f,   6.5f,   7.0f,   7.5f,   7.5f,    7.5f,
        7.5f,   7.5f,   7.5f,   7.5f,   7.5f,    7.5f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true,
      TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f}, {6.9f, 6.9f, 6.9f, 6.9f},
      TensorShape({1, 2, 3, 4}),
      {-0.1f,   0.0f,   0.1f,   0.5f,   1.0f,    1.5f,
        1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
        6.0f,   6.5f,   6.8f,   6.9f,   7.0f,    7.1f,
        7.2f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { 0.0f,   0.0f,   0.0f,   0.5f,   1.0f,    1.5f,
        1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
        6.0f,   6.5f,   7.0f,   7.0f,   7.0f,    7.0f,
        7.0f,   7.0f,   7.0f,   7.0f,   7.0f,    7.0f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, false,
      TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f}, {7.1f, 7.1f, 7.1f, 7.1f},
      TensorShape({1, 2, 3, 4}),
      { -0.6f,  -0.5f,  -0.4f,   0.0f,   0.5f,    1.0f,
         1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
         6.0f,   6.5f,   6.9f,   7.0f,   7.1f,    7.7f,
       100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { -0.5f, -0.5f,   -0.5f,   0.0f,   0.5f,    1.0f,
         1.5f,  2.0f,    2.5f,   3.0f,   3.5f,    4.0f,
         6.0f,  6.5f,    7.0f,   7.0f,   7.0f,    7.0f,
         7.0f,  7.0f,    7.0f,   7.0f,   7.0f,    7.0f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4NudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  // clang-format off
  RunTestFakeQuantWithMinMaxVarsPerChannel(
      4, true,
      TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f}, {6.6f, 6.6f, 6.6f, 6.6f},
      TensorShape({1, 2, 3, 4}),
      { -0.6f,  -0.5f,  -0.4f,   0.0f,   0.5f,    1.0f,
         1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
         5.5f,   6.0f,   6.4f,   6.5f,   6.6f,    6.7f,
       100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 1000.0f},
      { -0.5f , -0.5f,  -0.5f,   0.0f,   0.5f,    1.0f,
         1.5f,   2.0f,   2.5f,   3.0f,   3.5f,    4.0f,
         5.5f,   6.0f,   6.5f,   6.5f,   6.5f,    6.5f,
         6.5f,   6.5f,   6.5f,   6.5f,   6.5f,    6.5f});
  // clang-format on
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedDown_ZeroMinAndMax) {
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {0.0, 0.0, 0.0, 0.0f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {0.0, 0.0, 0.0, 0.0f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {0.0, 0.0, 0.0, 0.0f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {grad_flat(0), grad_flat(1), grad_flat(2), grad_flat(3)});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {0.0f, 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, 0.0f, 63.75f, 63.8f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {63.65f, 63.65f, 63.65f, 63.65f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, 0.0f, 63.5f, 63.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {63.4f, 63.4f, 63.4f, 63.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.3f, -0.25f, 63.5f, 63.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}),
                           {-0.125f, -0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}),
                           {63.625f, 63.625f, 63.625f, 63.625f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.3f, -0.25f, 63.25f, 63.3f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}),
                           {-0.125f, -0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}),
                           {63.375f, 63.375f, 63.375f, 63.375f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2GradientNudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.1f, 0.0f, 0.1f, 0.25f, 63.75f, 63.8f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {63.65f, 63.65f, 63.65f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2GradientNudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.1f, 0.0f, 0.1f, 0.25f, 63.5f, 63.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {63.4f, 63.4f, 63.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2GradientNudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.3f, -0.25f, -0.2f, 0.0f, 63.5f, 63.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {63.625f, 63.625f, 63.625f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2GradientNudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.3f, -0.25f, -0.2f, 0.0f, 63.25f, 63.3f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {63.375f, 63.375f, 63.375f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4GradientNudgedDown_RegularRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.4, nudged to 0.
  // Nudged ranges: [0.0; 63.75].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.75.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.1f,   0.0f, 63.75f, 63.8f, -0.1f,   0.0f,
                            63.75f, 63.8f, -0.1f,   0.0f, 63.75f, 63.8f,
                            -0.1f,   0.0f, 63.75f, 63.8f, -0.1f,   0.0f,
                            63.75f, 63.8f, -0.1f,   0.0f, 63.75f, 63.8f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {63.65f, 63.65f, 63.65f, 63.65f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4GradientNudgedDown_NarrowRange) {
  // Original quantization ranges: [-0.4 / 4 + 0 / 4, -0.4 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.4, nudged to 1.
  // Nudged ranges: [0.0; 63.5].
  // Expected quantized values: 0.0, 0.25, 0.5, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.1f,  0.0f, 63.5f, 63.6f, -0.1f,  0.0f,
                            63.5f, 63.6f, -0.1f,  0.0f, 63.5f, 63.6f,
                            -0.1f,  0.0f, 63.5f, 63.6f, -0.1f,  0.0f,
                            63.5f, 63.6f, -0.1f,  0.0f, 63.5f, 63.6f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {63.4f, 63.4f, 63.4f, 63.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4GradientNudgedUp_RegularRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 255 / 4].
  // Scale: 1/4,  original zero point: 0.5, nudged to 1.
  // Nudged ranges: [-0.25; 63.5].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.3f, -0.25f, 63.5f, 63.6f,  -0.3f, -0.25f,
                            63.5f, 63.6f,  -0.3f, -0.25f, 63.5f, 63.6f,
                            -0.3f, -0.25f, 63.5f, 63.6f,  -0.3f, -0.25f,
                            63.5f, 63.6f,  -0.3f, -0.25f, 63.5f, 63.6f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}),
                           {-0.125f, -0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}),
                           {63.625f, 63.625f, 63.625f, 63.625f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4GradientNudgedUp_NarrowRange) {
  // Original quantization ranges: [-0.5 / 4 + 0 / 4, -0.5 / 4 + 254 / 4].
  // Scale: 1/4,  original zero point: 1.5, nudged to 2.
  // Nudged ranges: [-0.25; 63.25].
  // Expected quantized values: -0.25, 0.0, 0.25, ..., 63.25.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           { -0.3f,  -0.25f, 63.25f, 63.3f,  -0.3f,  -0.25f,
                             63.25f, 63.3f,  -0.3f,  -0.25f, 63.25f, 63.3f,
                             -0.3f,  -0.25f, 63.25f, 63.3f,  -0.3f,  -0.25f,
                             63.25f, 63.3f,  -0.3f,  -0.25f, 63.25f, 63.3f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}),
                           {-0.125f, -0.125f, -0.125f, -0.125f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}),
                           {63.375f, 63.375f, 63.375f, 63.375f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim1GradientNudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, 0.0f, 7.5f, 7.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {7.4f, 7.4f, 7.4f, 7.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim1GradientNudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, 0.0f, 7.0f, 7.1f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {6.9f, 6.9f, 6.9f, 6.9f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim1GradientNudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.6f, -0.5f, 7.0f, 7.1f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {7.1f, 7.1f, 7.1f, 7.1f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim1GradientNudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({4}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({4}), {-0.6f, -0.5f, 6.5f, 6.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {6.6f, 6.6f, 6.6f, 6.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1), grad_flat(2), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, 0.0f, grad_flat(3)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim2GradientNudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.1f, 0.0f, 0.1f, 0.5f, 7.5f, 7.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {7.4f, 7.4f, 7.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim2GradientNudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.1f, 0.0f, 0.1f, 0.5f, 7.0f, 7.1f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {6.9f, 6.9f, 6.9f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim2GradientNudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 7.0f, 7.1f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {7.1f, 7.1f, 7.1f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim2GradientNudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 6.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({2, 3}));
  // Downstream inputs.
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {-0.6f, -0.5f, -0.4f, 0.0f, 6.5f, 6.6f});
  // Min.
  AddInputFromArray<float>(TensorShape({3}), {-0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({3}), {6.6f, 6.6f, 6.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT, TensorShape({2, 3}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(
      &expected_bprop_wrt_input,
      {0.0f, grad_flat(1), grad_flat(2), grad_flat(3), grad_flat(4), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_min, {grad_flat(0), 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({3}));
  FillValues<float>(&expected_bprop_wrt_max, {0.0f, 0.0f, grad_flat(5)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim4GradientNudgedDown_4Bits_RegularRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.2, nudged to 0.
  // Nudged range: [0.0; 7.5].
  // Expected quantized values: 0.0, 0.5, ..., 7.5.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.1f, 0.0f,  7.5f, 7.6f, -0.1f, 0.0f,
                             7.5f, 7.6f, -0.1f, 0.0f,  7.5f, 7.6f,
                            -0.1f, 0.0f,  7.5f, 7.6f, -0.1f, 0.0f,
                             7.5f, 7.6f, -0.1f, 0.0f,  7.5f, 7.6f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {7.4f, 7.4f, 7.4f, 7.4f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim4GradientNudgedDown_4Bits_NarrowRange) {
  // Original quantization range: [-0.2 / 2 + 0 / 2, -0.2 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.2, nudged to 1.
  // Nudged range: [0.0; 7.0].
  // Expected quantized values: 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.1f, 0.0f,  7.0f, 7.1f, -0.1f, 0.0f,
                             7.0f, 7.1f, -0.1f, 0.0f,  7.0f, 7.1f,
                            -0.1f, 0.0f,  7.0f, 7.1f, -0.1f, 0.0f,
                             7.0f, 7.1f, -0.1f, 0.0f,  7.0f, 7.1f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.1f, -0.1f, -0.1f, -0.1f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {6.9f, 6.9f, 6.9f, 6.9f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest,
       WithVarsPerChannelDim4GradientNudgedUp_4Bits_RegularRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 15 / 2].
  // Scale: 1/2,  original zero point: 0.8, nudged to 1.
  // Nudged range: [-0.5; 7.0].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", false)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.6f, -0.5f,  7.0f,  7.1f, -0.6f, -0.5f,
                             7.0f,  7.1f, -0.6f, -0.5f,  7.0f,  7.1f,
                            -0.6f, -0.5f,  7.0f,  7.1f, -0.6f, -0.5f,
                             7.0f,  7.1f, -0.6f, -0.5f,  7.0f,  7.1f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {7.1f, 7.1f, 7.1f, 7.1f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

TEST_F(QuantOpsTest, WithVarsPerChannelDim4GradientNudgedUp_4Bits_NarrowRange) {
  // Original quantization range: [-0.8 / 2 + 0 / 2, -0.8 / 2 + 14 / 2].
  // Scale: 1/2,  original zero point: 1.8, nudged to 2.
  // Nudged range: [-0.5; 6.5].
  // Expected quantized values: -0.5, 0.0, 0.5, ..., 7.0.
  TF_EXPECT_OK(NodeDefBuilder("op", "FakeQuantWithMinMaxVarsPerChannelGradient")
                   .Attr("num_bits", 4)
                   .Attr("narrow_range", true)
                   .Input(FakeInput(DT_FLOAT))  // gradients
                   .Input(FakeInput(DT_FLOAT))  // inputs
                   .Input(FakeInput(DT_FLOAT))  // min
                   .Input(FakeInput(DT_FLOAT))  // max
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  // Upstream gradients.
  AddRandomInput(TensorShape({1, 2, 3, 4}));
  // Downstream inputs.
  // clang-format off
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {-0.6f, -0.5f,  6.5f,  6.6f, -0.6f, -0.5f,
                             6.5f,  6.6f, -0.6f, -0.5f,  6.5f,  6.6f,
                            -0.6f, -0.5f,  6.5f,  6.6f, -0.6f, -0.5f,
                             6.5f,  6.6f, -0.6f, -0.5f,  6.5f,  6.6f});
  // clang-format on
  // Min.
  AddInputFromArray<float>(TensorShape({4}), {-0.4f, -0.4f, -0.4f, -0.4f});
  // Max.
  AddInputFromArray<float>(TensorShape({4}), {6.6f, 6.6f, 6.6f, 6.6f});

  // Tested code.
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output_bprop_wrt_input = GetOutput(0);
  Tensor expected_bprop_wrt_input(allocator(), DT_FLOAT,
                                  TensorShape({1, 2, 3, 4}));
  auto grad_flat = GetInput(0).flat<float>();
  FillValues<float>(&expected_bprop_wrt_input,
                    {0.0f, grad_flat(1),  grad_flat(2),  0.0f,
                     0.0f, grad_flat(5),  grad_flat(6),  0.0f,
                     0.0f, grad_flat(9),  grad_flat(10), 0.0f,
                     0.0f, grad_flat(13), grad_flat(14), 0.0f,
                     0.0f, grad_flat(17), grad_flat(18), 0.0f,
                     0.0f, grad_flat(21), grad_flat(22), 0.0f});
  ExpectClose(expected_bprop_wrt_input, *output_bprop_wrt_input);

  Tensor* output_bprop_wrt_min = GetOutput(1);
  Tensor expected_bprop_wrt_min(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_min,
                    {grad_flat(0) + grad_flat(4) + grad_flat(8) +
                         grad_flat(12) + grad_flat(16) + grad_flat(20),
                     0.0f, 0.0f, 0.0f});
  ExpectClose(expected_bprop_wrt_min, *output_bprop_wrt_min);

  Tensor* output_bprop_wrt_max = GetOutput(2);
  Tensor expected_bprop_wrt_max(allocator(), DT_FLOAT, TensorShape({4}));
  FillValues<float>(&expected_bprop_wrt_max,
                    {0.0f, 0.0f, 0.0f,
                     grad_flat(3) + grad_flat(7) + grad_flat(11) +
                         grad_flat(15) + grad_flat(19) + grad_flat(23)});
  ExpectClose(expected_bprop_wrt_max, *output_bprop_wrt_max);
}

}  // namespace tensorflow
