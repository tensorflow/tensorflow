/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_

#include <cstdint>

namespace xla::codegen::intrinsic::accuracy {

// Accuracy budgets in ULPs (Unit in the Last Place).
// These budgets are determined based on the performance of the intrinsics
// relative to a high-precision reference (e.g., mpmath).

struct UlpBudget {
  int64_t regular;
  int64_t subnormal;
  int64_t special_values = 0;
};

struct AccuracyBudget {
  UlpBudget cpu;
  UlpBudget gpu;
};

// Exp
constexpr AccuracyBudget kExpF32Budget = {
    /*cpu=*/{/*regular=*/6,
             /*subnormal=*/27},
    /*gpu=*/
    {/*regular=*/7,
     /*subnormal=*/0},
};

constexpr AccuracyBudget kExpF64Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/41132809365},
    /*gpu=*/{/*regular=*/1,
             /*subnormal=*/0}};

// Log1p
constexpr AccuracyBudget kLog1pF32Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/0},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/0},
};

constexpr AccuracyBudget kLog1pF64Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/1},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/0},
};

// Rsqrt
constexpr AccuracyBudget kRsqrtF32Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/0,
             /*special_values=*/4},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/0,
     /*special_values=*/2},
};

constexpr AccuracyBudget kRsqrtF64Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/1000000,
             /*special_values=*/4},
    /*gpu=*/
    {/*regular=*/0,
     /*subnormal=*/0,
     /*special_values=*/2},
};

// Tanh
constexpr AccuracyBudget kTanhF32Budget = {
    /*cpu=*/{/*regular=*/3,
             /*subnormal=*/0},
    /*gpu=*/
    {/*regular=*/3,
     /*subnormal=*/0},
};

constexpr AccuracyBudget kTanhF64Budget = {
    /*cpu=*/{/*regular=*/4,
             /*subnormal=*/1},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/0},
};

// Erf
constexpr AccuracyBudget kErfF32Budget = {
    /*cpu=*/{/*regular=*/2,
             /*subnormal=*/0},
    /*gpu=*/
    {/*regular=*/2,
     /*subnormal=*/0},
};

constexpr AccuracyBudget kErfF64Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/1},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/0},
};

// Sqrt
constexpr AccuracyBudget kSqrtF32Budget = {
    /*cpu=*/{/*regular=*/1,
             /*subnormal=*/1000000},
    /*gpu=*/
    {/*regular=*/1,
     /*subnormal=*/1000000},
};

constexpr AccuracyBudget kSqrtF64Budget = {
    /*cpu=*/{/*regular=*/0,
             /*subnormal=*/2188749418902061056,
             /*special_values=*/4},
    /*gpu=*/
    {/*regular=*/0,
     /*subnormal=*/0,
     /*special_values=*/0},
};

}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
