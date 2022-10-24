/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace tensorflow {

namespace {

using protobuf::TextFormat;

constexpr int32_t kInt8Min = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt8Max = std::numeric_limits<int8_t>::max();

template <typename T>
std::vector<T> Arange(int start, int stop, int step = 1) {
  std::vector<T> array;
  int val = start;
  while (val < stop) {
    array.push_back(val);
    val += step;
  }
  return array;
}

}  // namespace

class UniformQuantizedConvolutionTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizedConvolutionTest, HybridPerTensorQuantizedDefaultAttrs) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Finalize(node_def()));
  // Uses default Attrs (and default conv_params settings).
  //
  // batch_group_count = 1
  // feature_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2, 2}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {12176., 11480., 9392.,  8696.,  2960.,  2840.,  2480.,  2360.,
       -6256., -5800., -4432., -3976., -4528., -5224., -7312., -8008.,
       80.,    -40.,   -400.,  -520.,  4688.,  5144.,  6512.,  6968.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest, HybridPerTensorQuantizedSetStrides) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Attr("window_strides", {2, 3})
                   .Finalize(node_def()));
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 1, 1}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(&expected,
                          {12176., 2960., -6256., -4528., 80., 4688.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest, HybridPerTensorQuantizedSetPadding) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "EXPLICIT")
                   .Attr("explicit_padding", {0, 1, 1, 2})
                   .Finalize(node_def()));
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 5}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {8072.,  12176., 11480., 7640.,  3808.,  6280.,  9392.,  8696.,  5720.,
       2816.,  2896.,  4288.,  3904.,  2536.,  1232.,  1736.,  2960.,  2840.,
       2072.,  1120.,  1480.,  2480.,  2360.,  1688.,  896.,   880.,   1408.,
       1312.,  904.,   464.,   -4600., -6256., -5800., -3496., -1568., -3320.,
       -4432., -3976., -2344., -1024., -1136., -1472., -1280., -728.,  -304.,
       -2680., -4528., -5224., -3880., -2144., -4472., -7312., -8008., -5800.,
       -3136., -3056., -4928., -5312., -3800., -2032., 200.,   80.,    -40.,
       -232.,  -224.,  -56.,   -400.,  -520.,  -616.,  -448.,  -464.,  -896.,
       -992.,  -824.,  -496.,  3080.,  4688.,  5144.,  3416.,  1696.,  4360.,
       6512.,  6968.,  4568.,  2240.,  2128.,  3136.,  3328.,  2152.,  1040.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1.5, /*rtol=*/0.04);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetExplicitPadding) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "EXPLICIT")
                   .Attr("explicit_padding", {0, 1, 1, 2})
                   .Finalize(node_def()));
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 5}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {8072.,  12176., 11480., 7640.,  3808.,  6280.,  9392.,  8696.,  5720.,
       2816.,  2896.,  4288.,  3904.,  2536.,  1232.,  1736.,  2960.,  2840.,
       2072.,  1120.,  1480.,  2480.,  2360.,  1688.,  896.,   880.,   1408.,
       1312.,  904.,   464.,   -4600., -6256., -5800., -3496., -1568., -3320.,
       -4432., -3976., -2344., -1024., -1136., -1472., -1280., -728.,  -304.,
       -2680., -4528., -5224., -3880., -2144., -4472., -7312., -8008., -5800.,
       -3136., -3056., -4928., -5312., -3800., -2032., 200.,   80.,    -40.,
       -232.,  -224.,  -56.,   -400.,  -520.,  -616.,  -448.,  -464.,  -896.,
       -992.,  -824.,  -496.,  3080.,  4688.,  5144.,  3416.,  1696.,  4360.,
       6512.,  6968.,  4568.,  2240.,  2128.,  3136.,  3328.,  2152.,  1040.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1.5, /*rtol=*/0.04);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetDimensionNumbers) {
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 1
                                            input_feature_dimension: 3
                                            input_spatial_dimensions: 2
                                            input_spatial_dimensions: 0
                                            kernel_output_feature_dimension: 2
                                            kernel_input_feature_dimension: 1
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 3
                                            output_batch_dimension: 2
                                            output_feature_dimension: 1
                                            output_spatial_dimensions: 3
                                            output_spatial_dimensions: 0
                                          )pb",
                                          &dimension_numbers));
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tlhs", DT_FLOAT)
          .Attr("Trhs", DT_QINT8)
          .Attr("Tout", DT_FLOAT)
          .Attr("rhs_quantization_axis", -1)
          .Attr("rhs_quantization_min_val", kInt8Min)
          .Attr("rhs_quantization_max_val", kInt8Max)
          .Attr("padding", "VALID")
          .Attr("dimension_numbers", dimension_numbers.SerializeAsString())
          .Finalize(node_def()));
  // strides = [1, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({4, 2, 3, 2}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 2, 3, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2, 2}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected, {3960., 3432., 2376., 1848., 2304., 2064., 1584., 1344.,
                  648.,  696.,  792.,  840.,  792.,  264.,  -792., -1320.,
                  864.,  624.,  144.,  -96.,  936.,  984.,  1080., 1128.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/10, /*rtol=*/0.02);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetFeatureGroupCount) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Attr("feature_group_count", 2)
                   .Finalize(node_def()));
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // batch_group_count = 1
  // strides = [1, 1]
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 4, 3, 4}),
                           Arange<float>(-98, 94, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({4, 2, 2, 3}), Arange<qint8>(-24, 24));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4, 2, 2}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {40400., 39416., 36464.,  35480.,  17360.,  16952.,  15728., 15320.,
       -1648., -1480., -976.,   -808.,   -10864., -10120., -7888., -7144.,
       -6832., -7816., -10768., -11752., -2224.,  -2632.,  -3856., -4264.,
       6416.,  6584.,  7088.,   7256.,   24848.,  25592.,  27824., 28568.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetBatchGroupCount) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Attr("batch_group_count", 2)
                   .Finalize(node_def()));
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs (quantized) tensor.
  AddInputFromArray<float>(TensorShape({4, 2, 3, 4}),
                           Arange<float>(-98, 94, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({4, 2, 2, 3}), Arange<qint8>(-24, 24));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4, 2, 2}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {40400., 39416., 36464., 35480., 17360., 16952., 15728., 15320.,
       2384.,  2552.,  3056.,  3224.,  6992.,  7736.,  9968.,  10712.,
       16784., 15800., 12848., 11864., 7568.,  7160.,  5936.,  5528.,
       6416.,  6584.,  7088.,  7256.,  24848., 25592., 27824., 28568.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetLhsDilation) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Attr("lhs_dilation", {2, 2})
                   .Finalize(node_def()));
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // strides = [1, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 4, 5}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected,
      {5032.,  2448.,  4776.,  2320.,  4520.,  3312.,  1600.,  3104.,  1496.,
       2896.,  4008.,  1936.,  3752.,  1808.,  3496.,  2480.,  1184.,  2272.,
       1080.,  2064.,  1480.,  720.,   1416.,  688.,   1352.,  528.,   256.,
       512.,   248.,   496.,   1224.,  592.,   1160.,  560.,   1096.,  464.,
       224.,   448.,   216.,   432.,   -2072., -1008., -1944., -944.,  -1816.,
       -2256., -1088., -2080., -1000., -1904., -1560., -752.,  -1432., -688.,
       -1304., -1552., -736.,  -1376., -648.,  -1200., -1112., -624.,  -1368.,
       -752.,  -1624., -1680., -896.,  -1888., -1000., -2096., -2136., -1136.,
       -2392., -1264., -2648., -2512., -1312., -2720., -1416., -2928., -56.,
       -48.,   -120.,  -80.,   -184.,  144.,   64.,    128.,   56.,    112.,
       -312.,  -176.,  -376.,  -208.,  -440.,  80.,    32.,    64.,    24.,
       48.,    1000.,  528.,   1128.,  592.,   1256.,  1968.,  1024.,  2144.,
       1112.,  2320.,  1512.,  784.,   1640.,  848.,   1768.,  2672.,  1376.,
       2848.,  1464.,  3024.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerTensorQuantizedSetRhsDilation) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Attr("rhs_dilation", {2, 2})
                   .Finalize(node_def()));
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // strides = [1, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 4, 5}),
                           Arange<float>(-82, 78, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2, 1}));
  // Output should be close to
  // conv(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(
      &expected, {18568., 15088., 4744., 4144., -9080., -6800., -9272., -12752.,
                  -56., -656., 9160., 11440.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest, HybridPerChannelQuantizedDefaultAttrs) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("rhs_quantization_min_val", kInt8Min)
                   .Attr("rhs_quantization_max_val", kInt8Max)
                   .Attr("padding", "VALID")
                   .Finalize(node_def()));
  // Uses default Attrs (and default conv_params settings).
  //
  // batch_group_count = 1
  // feature_group_count = 1
  // strides = [1, 1]
  // dimension_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2, 3, 4}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({3, 2, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {2.0, 4.0, 2.0});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2, 2}));
  // Output should be close to
  // conv(lhs, Per-channel-dequantized-rhs)
  test::FillValues<float>(
      &expected,
      {12176., 11480., 9392.,  8696.,  8992.,  8560.,  7264.,  6832.,
       -6256., -5800., -4432., -3976., -4528., -5224., -7312., -8008.,
       -1376., -1808., -3104., -3536., 4688.,  5144.,  6512.,  6968.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerChannelQuantizedTFConv2DLikeConfig) {
  // Like TF Conv2D Default (data_format=NHWC),
  // dimension_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 0
                                            input_feature_dimension: 3
                                            input_spatial_dimensions: 1
                                            input_spatial_dimensions: 2
                                            kernel_output_feature_dimension: 3
                                            kernel_input_feature_dimension: 2
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 1
                                            output_batch_dimension: 0
                                            output_feature_dimension: 3
                                            output_spatial_dimensions: 1
                                            output_spatial_dimensions: 2
                                          )pb",
                                          &dimension_numbers));
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tlhs", DT_FLOAT)
          .Attr("Trhs", DT_QINT8)
          .Attr("Tout", DT_FLOAT)
          .Attr("rhs_quantization_axis", 3)
          .Attr("rhs_quantization_min_val", kInt8Min)
          .Attr("rhs_quantization_max_val", kInt8Max)
          .Attr("padding", "VALID")
          .Attr("dimension_numbers", dimension_numbers.SerializeAsString())
          .Finalize(node_def()));
  // strides = [1, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 3, 4, 2}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3, 2, 3}), Arange<qint8>(-18, 18));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {2.0, 4.0, 2.0});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 3}));
  // Output should be close to
  // conv(lhs, Per-channel-dequantized-rhs)
  test::FillValues<float>(
      &expected, {5256., 12288., 3480., 4920.,  11424., 3336., 3912.,  8832.,
                  2904., 3576.,  7968., 2760.,  1224.,  1920., 1752.,  888.,
                  1056., 1608.,  -120., -1536., 1176.,  -456., -2400., 1032.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/4, /*rtol=*/0.04);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerChannelQuantizedTFDepthwiseConv2DLikeConfig) {
  // Like TF DepthwiseConv2D Default (data_format=NHWC),
  // dimension_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // Where the input shapes are set to make feature_group_count to input feature
  // dimension size.
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 0
                                            input_feature_dimension: 3
                                            input_spatial_dimensions: 1
                                            input_spatial_dimensions: 2
                                            kernel_output_feature_dimension: 3
                                            kernel_input_feature_dimension: 2
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 1
                                            output_batch_dimension: 0
                                            output_feature_dimension: 3
                                            output_spatial_dimensions: 1
                                            output_spatial_dimensions: 2
                                          )pb",
                                          &dimension_numbers));
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tlhs", DT_FLOAT)
          .Attr("Trhs", DT_QINT8)
          .Attr("Tout", DT_FLOAT)
          .Attr("rhs_quantization_axis", 3)
          .Attr("rhs_quantization_min_val", kInt8Min)
          .Attr("rhs_quantization_max_val", kInt8Max)
          .Attr("padding", "VALID")
          .Attr("feature_group_count", 2)
          .Attr("dimension_numbers", dimension_numbers.SerializeAsString())
          .Finalize(node_def()));
  // strides = [1, 1]
  // batch_group_count = 1
  // lhs_dilation = [1, 1]
  // rhs_dilation = [1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 3, 4, 2}),
                           Arange<float>(-50, 46, 2));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3, 1, 2}), Arange<qint8>(-6, 6));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({2}), {2.0, 4.0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 4});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 2}));
  // Output should be close to
  // conv(lhs, Per-channel-dequantized-rhs)
  test::FillValues<float>(
      &expected, {1720., 4160., 1576., 3776., 1144., 2624., 1000., 2240., -8.,
                  -448., -152., -832., -584., -1984., -728., -2368.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedConvolutionTest,
       HybridPerChannelQuantizedTFConv3DLikeConfig) {
  // Like TF Conv3D Default (data_format=NDHWC),
  // dimension_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            input_batch_dimension: 0
                                            input_feature_dimension: 4
                                            input_spatial_dimensions: 1
                                            input_spatial_dimensions: 2
                                            input_spatial_dimensions: 3
                                            kernel_output_feature_dimension: 4
                                            kernel_input_feature_dimension: 3
                                            kernel_spatial_dimensions: 0
                                            kernel_spatial_dimensions: 1
                                            kernel_spatial_dimensions: 2
                                            output_batch_dimension: 0
                                            output_feature_dimension: 4
                                            output_spatial_dimensions: 1
                                            output_spatial_dimensions: 2
                                            output_spatial_dimensions: 3
                                          )pb",
                                          &dimension_numbers));
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedConvolutionHybrid")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tlhs", DT_FLOAT)
          .Attr("Trhs", DT_QINT8)
          .Attr("Tout", DT_FLOAT)
          .Attr("rhs_quantization_axis", 4)
          .Attr("rhs_quantization_min_val", kInt8Min)
          .Attr("rhs_quantization_max_val", kInt8Max)
          .Attr("padding", "VALID")
          .Attr("dimension_numbers", dimension_numbers.SerializeAsString())
          .Finalize(node_def()));
  // strides = [1, 1, 1]
  // batch_group_count = 1
  // feature_group_count = 1
  // lhs_dilation = [1, 1, 1]
  // rhs_dilation = [1, 1, 1]
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 3, 4, 2, 2}),
                           Arange<float>(-50, 46));
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3, 2, 2, 2}),
                           Arange<qint8>(-24, 24));
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({2}), {2.0, 4.0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 4});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1, 2}));
  // Output should be close to
  // conv(lhs, Per-channel-dequantized-rhs)
  test::FillValues<float>(
      &expected, {11008., 25520., 10432., 23984., 8704., 19376., 8128., 17840.,
                  4096., 7088., 3520., 5552., 1792., 944., 1216., -592.});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/11, /*rtol=*/0.02);
}

}  // namespace tensorflow
