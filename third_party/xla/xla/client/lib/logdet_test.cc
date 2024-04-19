/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/client/lib/logdet.h"

#include <limits>

#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "xla/statusor.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "tsl/lib/core/status_test_util.h"

namespace {

using LogDetTest = xla::ClientLibraryTestBase;

XLA_TEST_F(LogDetTest, Simple) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::SignAndLogDet slogdet = xla::SLogDet(a);
  xla::XlaOp logdet = xla::LogDet(a);
  xla::Tuple(&builder, {slogdet.sign, slogdet.logdet, logdet});
  xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
      xla::LiteralUtil::CreateR0<float>(1.f),
      xla::LiteralUtil::CreateR0<float>(14.1601f),
      xla::LiteralUtil::CreateR0<float>(14.1601f));
  ComputeAndCompareLiteral(&builder, expected, {a_data.get()},
                           xla::ErrorSpec(1e-4));
}

XLA_TEST_F(LogDetTest, SimpleTriangle) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {4, 6, 8, 10},
      {4, -39, 62, 73},
      {0, 0, -146, 166},
      {4, 6, 8, 320},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::SignAndLogDet slogdet = xla::SLogDet(a);
  xla::XlaOp logdet = xla::LogDet(a);
  xla::Tuple(&builder, {slogdet.sign, slogdet.logdet, logdet});
  xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
      xla::LiteralUtil::CreateR0<float>(1.f),
      xla::LiteralUtil::CreateR0<float>(15.9131355f),
      xla::LiteralUtil::CreateR0<float>(15.9131355f));

  ComputeAndCompareLiteral(&builder, expected, {a_data.get()},
                           xla::ErrorSpec(1e-4));
}

XLA_TEST_F(LogDetTest, SimpleBatched) {
  xla::XlaBuilder builder(TestName());

  xla::Array3D<float> a_vals({
      {
          {4, 6, 8, 10},
          {6, 45, 54, 63},
          {8, 54, 146, 166},
          {10, 63, 166, 310},
      },
      {
          {16, 24, 8, 12},
          {24, 61, 82, 48},
          {8, 82, 456, 106},
          {12, 48, 106, 62},
      },
      {{2, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 8}, {10, 11, 12, 13}},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
  });

  xla::XlaOp a;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::SignAndLogDet slogdet = xla::SLogDet(a);
  xla::XlaOp logdet = xla::LogDet(a);
  xla::Tuple(&builder, {slogdet.sign, slogdet.logdet, logdet});
  xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
      xla::LiteralUtil::CreateR1<float>({1.f, 1.f, -1.f, 0.f}),
      xla::LiteralUtil::CreateR1<float>(
          {14.1601f, 14.3092f, 2.4849f,
           -std::numeric_limits<float>::infinity()}),
      xla::LiteralUtil::CreateR1<float>(
          {14.1601f, 14.3092f, std::numeric_limits<float>::quiet_NaN(),
           -std::numeric_limits<float>::infinity()}));

  ComputeAndCompareLiteral(&builder, expected, {a_data.get()},
                           xla::ErrorSpec(1e-4));
}

XLA_TEST_F(LogDetTest, LogdetOfLargerMatricesBatched) {
  xla::XlaBuilder builder(TestName());

  xla::Array<float> a_vals = {
      {{7.2393, 1.1413, 4.1883, -4.8272, 3.2831, -0.0568, -2.4776},
       {0.4347, 3.4095, 1.6259, -4.7100, 1.5942, 1.4217, -2.8009},
       {3.6964, 0.4882, 6.5276, -1.2128, 1.3851, 0.7417, -3.8515},
       {-3.7986, -5.1188, -1.9410, 14.0205, -5.4515, 3.1831, 5.1488},
       {1.5621, 3.0426, 1.4819, -4.5938, 10.1397, 4.9312, -2.8351},
       {-1.5436, -0.0287, -0.1139, 4.4499, 2.5894, 6.1216, 2.7201},
       {-3.7241, -2.7670, -3.8162, 4.5961, -1.7251, -0.4190, 8.6562}},

      {{3.3789, -2.3607, -1.2471, 2.1503, 0.6062, -0.6057, 1.7748},
       {-1.8670, 11.0947, 0.1229, 0.0599, 3.1714, -4.7941, -4.5442},
       {-0.6905, -0.0829, 5.2156, 2.9528, 2.6200, 6.1638, 1.8652},
       {3.0521, 2.2174, 0.7444, 10.7268, 0.6443, -2.7732, 1.6840},
       {1.8479, 3.0821, 4.5671, 2.9254, 6.1338, 5.2066, 2.3662},
       {-0.0360, -5.5341, 5.9687, -0.3297, 2.1174, 13.0016, 4.0118},
       {0.4380, -4.6683, 3.1548, 0.0924, 0.7176, 6.4679, 6.1819}},

      {{10.0487, 4.0350, -0.8471, -1.2887, -0.8172, -3.3698, 1.3191},
       {4.8678, 4.6081, 0.8419, -0.2454, -3.2599, -1.2386, 2.4070},
       {1.4877, 0.8362, 2.6077, 1.1782, -0.1116, 1.7130, -1.1883},
       {-0.9245, -0.7435, -0.9456, 2.5936, 1.9887, -0.1324, -0.1453},
       {0.2918, -0.5301, -0.8775, 1.0478, 8.9262, 2.4731, -0.4393},
       {-3.5759, -1.5619, 2.4410, 1.3046, 4.2678, 7.3587, -4.0935},
       {-1.1187, 0.9150, -1.8253, 0.0390, -2.5684, -4.0778, 4.1447}}};

  xla::XlaOp a;
  auto a_data = CreateParameter<float>(a_vals, 0, "a", &builder, &a);
  xla::SignAndLogDet slogdet = xla::SLogDet(a);
  xla::XlaOp logdet = xla::LogDet(a);
  xla::Tuple(&builder, {slogdet.sign, slogdet.logdet, logdet});
  xla::Literal expected = xla::LiteralUtil::MakeTupleOwned(
      xla::LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f}),
      xla::LiteralUtil::CreateR1<float>({8.93788053, 6.77846303, 7.4852403}),
      xla::LiteralUtil::CreateR1<float>({8.93788053, 6.77846303, 7.4852403}));

  ComputeAndCompareLiteral(&builder, expected, {a_data.get()},
                           xla::ErrorSpec(1e-4));
}

}  // namespace
