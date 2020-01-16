/* Copyright 2019 Google LLC. All Rights Reserved.

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
#include <iostream>

#include "tensorflow/lite/experimental/ruy/ruy.h"

void ExampleMulFloat(ruy::Context *context) {
  const float lhs_data[] = {1, 2, 3, 4};
  const float rhs_data[] = {1, 2, 3, 4};
  float dst_data[4];

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  lhs.data = lhs_data;
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  rhs.data = rhs_data;
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  dst.data = dst_data;

  ruy::BasicSpec<float, float> spec;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);

  std::cout << "Example Mul, float:\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
}

void ExampleMulFloatWithBiasAddAndClamp(ruy::Context *context) {
  const float lhs_data[] = {1, 2, 3, 4};
  const float rhs_data[] = {1, 2, 3, 4};
  const float bias_data[] = {1, 0};
  float dst_data[4];

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  lhs.data = lhs_data;
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  rhs.data = rhs_data;
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  dst.data = dst_data;

  ruy::BasicSpec<float, float> spec;
  spec.bias = bias_data;
  spec.clamp_min = 0;
  spec.clamp_max = 15;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);

  std::cout << "Example Mul, float with bias addition and clamp:\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
}

void ExampleMulUint8AsymmetricQuantized(ruy::Context *context) {
  const std::uint8_t lhs_data[] = {124, 125, 126, 127};
  const std::uint8_t rhs_data[] = {129, 130, 131, 132};
  std::uint8_t dst_data[4];

  ruy::Matrix<std::uint8_t> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  lhs.data = lhs_data;
  lhs.zero_point = 125;
  ruy::Matrix<std::uint8_t> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  rhs.data = rhs_data;
  rhs.zero_point = 132;
  ruy::Matrix<std::uint8_t> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  dst.data = dst_data;
  dst.zero_point = 129;

  ruy::BasicSpec<std::int32_t, std::uint8_t> spec;
  spec.multiplier_fixedpoint = 1 << 30;

  spec.multiplier_exponent = 0;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);

  std::cout << "Example Mul, uint8 quantized with asymmetric zero points:\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
}
void ExampleMulInt8PerChannelQuantized(ruy::Context *context) {
  const std::int8_t lhs_data[] = {1, 2, 3, 4};
  const std::int8_t rhs_data[] = {1, 2, 3, 4};
  const std::int32_t multiplier_data[] = {3 << 28, 5 << 28};
  const int exponent_data[] = {1, -2};
  std::int8_t dst_data[4];

  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  lhs.data = lhs_data;
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  rhs.data = rhs_data;
  ruy::Matrix<std::int8_t> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  dst.data = dst_data;

  ruy::BasicSpec<std::int32_t, std::int8_t> spec;
  spec.multiplier_fixedpoint_perchannel = multiplier_data;
  spec.multiplier_exponent_perchannel = exponent_data;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);

  std::cout << "Example Mul, int8 quantized with per-channel multipliers\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
}

int main() {
  ruy::Context context;
  ExampleMulFloat(&context);
  ExampleMulFloatWithBiasAddAndClamp(&context);
  ExampleMulUint8AsymmetricQuantized(&context);
  ExampleMulInt8PerChannelQuantized(&context);
}
