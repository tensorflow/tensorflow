/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <functional>

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
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizedBiasAddTest : public OpsTestBase {
 protected:
};

TEST_F(QuantizedBiasAddTest, Small) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_bias_add_op", "QuantizedBiasAdd")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = 0.0f;
  const float input_max = 60.0f;
  const int input_height = 2;
  const int input_width = 3;
  Tensor input_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&input_float,
                          {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);

  const float bias_min = 0.0f;
  const float bias_max = 3.0f;
  const int bias_width = 3;
  Tensor bias_float(DT_FLOAT, {bias_width});
  test::FillValues<float>(&bias_float, {1.0f, 2.0f, 3.0f});
  Tensor bias_quantized =
      FloatTensorToQuantized<quint8>(bias_float, bias_min, bias_max);

  Tensor expected_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(&expected_float,
                          {11.0f, 22.0f, 33.0f, 41.0f, 52.0f, 63.0f});

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<quint8>(bias_quantized.shape(),
                            bias_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<float>(TensorShape({1}), {bias_min});
  AddInputFromArray<float>(TensorShape({1}), {bias_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedBiasAddTest, RealData) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_bias_add_op", "QuantizedBiasAdd")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = -2164.25f;
  const float input_max = 2006.27f;
  const int input_height = 1;
  const int input_width = 64;
  Tensor input_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(
      &input_float,
      {-1014.12, -157.382, -810.17,  1435.28,  1016.37,  219.684,  -316.054,
       -2164.25, 2006.27,  -547.444, 857.376,  404.376,  9.72115,  332.588,
       194.385,  -286.57,  26.062,   23.1125,  110.436,  247.055,  -127.683,
       -376.275, -124.81,  -846.826, -77.1507, 305.581,  -202.747, 12.9528,
       9.64886,  872.686,  40.9069,  197.816,  44.16,    -306.768, -1457.52,
       -368.939, -1049.42, -486.353, 1745.87,  95.7695,  395.773,  -254.333,
       -404.27,  787.16,   -2.44114, 199.37,   -1024.08, 784.901,  235.055,
       -42.7295, 241.498,  -245.365, 470.763,  186.159,  186.579,  -220.163,
       1304.58,  386.272,  -358.853, -755.996, 360.109,  -866.007, 55.2828,
       -508.801});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);

  const float bias_min = -0.739539f;
  const float bias_max = 0.641057f;
  const int bias_width = 64;
  Tensor bias_float(DT_FLOAT, {bias_width});
  test::FillValues<float>(
      &bias_float,
      {-0.294619, -0.0670519, 0.261507,   -0.126274, 0.127229,   -0.176945,
       -0.251223, 0.231086,   0.453694,   0.415666,  -0.288733,  0.508717,
       0.211551,  0.0435907,  -0.582383,  -0.308779, 0.0696883,  -0.438122,
       0.114,     0.433964,   0.109883,   0.284931,  -0.149661,  0.108657,
       0.458333,  -0.130231,  -0.35805,   -0.123206, -0.437968,  0.0282411,
       0.628818,  -0.0522173, -0.0233403, 0.124863,  0.217165,   0.262294,
       -0.171005, -0.254693,  -0.200433,  -0.287354, 0.488166,   -0.0354688,
       -0.118091, -0.590444,  0.491537,   -0.739539, 0.083117,   0.282482,
       0.275269,  -0.36574,   0.107476,   0.0511428, -0.136887,  -0.0149852,
       -0.259694, 0.641057,   0.264054,   -0.295126, -0.0218791, 0.361211,
       0.012448,  0.0709718,  -0.392394,  -0.434215});
  Tensor bias_quantized =
      FloatTensorToQuantized<quint8>(bias_float, bias_min, bias_max);

  Tensor expected_float(DT_FLOAT, {input_height, input_width});
  test::FillValues<float>(
      &expected_float,
      {-1014.42, -157.449, -809.908, 1435.16,  1016.5,  219.507,  -316.305,
       -2164.02, 2006.73,  -547.028, 857.088,  404.885, 9.9327,   332.632,
       193.803,  -286.878, 26.1317,  22.6744,  110.55,  247.489,  -127.573,
       -375.99,  -124.959, -846.717, -76.6923, 305.451, -203.105, 12.8296,
       9.21089,  872.714,  41.5357,  197.764,  44.1367, -306.643, -1457.3,
       -368.677, -1049.6,  -486.608, 1745.67,  95.4821, 396.261,  -254.368,
       -404.388, 786.57,   -1.94961, 198.63,   -1024.0, 785.183,  235.33,
       -43.0953, 241.605,  -245.314, 470.627,  186.144, 186.319,  -219.522,
       1304.84,  385.977,  -358.874, -755.635, 360.122, -865.936, 54.8904,
       -509.235});

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<quint8>(bias_quantized.shape(),
                            bias_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<float>(TensorShape({1}), {bias_min});
  AddInputFromArray<float>(TensorShape({1}), {bias_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 20.0);
}

}  // namespace tensorflow
