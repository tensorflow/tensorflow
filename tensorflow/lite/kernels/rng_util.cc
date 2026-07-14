/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/rng_util.h"

#include <array>
#include <cstdint>

namespace tflite {
namespace rng {

// 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
static constexpr uint32_t kThreefryParity = 0x1BD11BDA;
// Constants specified by the Philox algorithm.
static constexpr uint64_t kPhiloxM4x32A = 0xD2511F53;
static constexpr uint64_t kPhiloxM4x32B = 0xCD9E8D57;
static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;

// Implements the ThreeFry counter-based PRNG algorithm. Use 20 rounds.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::array<uint32_t, 2> Threefry2x32(uint32_t key_0, uint32_t key_1,
                                     std::array<uint32_t, 2> ctr) {
  // Rotation distances specified by the Threefry2x32 algorithm.
  constexpr std::array<std::array<int, 4>, 2> rotations{
      std::array<int, 4>{13, 15, 26, 6}, std::array<int, 4>{17, 29, 16, 24}};

  uint32_t key_2 = key_0 ^ key_1 ^ kThreefryParity;
  ctr[0] += key_0;
  ctr[1] += key_1;

  // Performs 4 round of the Threefry2x32 algorithm, rotation amount specified
  // by 'rotations'.
  auto apply_round = [&](int r, uint32_t ks0, uint32_t ks1, int b) {
    for (int rot : rotations[r]) {
      ctr[0] += ctr[1];
      // Rotates the 32-bit integer 'ctr[1]' left by 'rot' bits.
      ctr[1] = (ctr[1] << rot) | (ctr[1] >> (32 - rot));
      ctr[1] ^= ctr[0];
    }
    ctr[0] += ks0;
    ctr[1] += ks1 + b;
  };

  // Applies 20 rounds.
  apply_round(/*r=*/0, /*ks0=*/key_1, /*ks1=*/key_2, /*b=*/1);
  apply_round(/*r=*/1, /*ks0=*/key_2, /*ks1=*/key_0, /*b=*/2);
  apply_round(/*r=*/0, /*ks0=*/key_0, /*ks1=*/key_1, /*b=*/3);
  apply_round(/*r=*/1, /*ks0=*/key_1, /*ks1=*/key_2, /*b=*/4);
  apply_round(/*r=*/0, /*ks0=*/key_2, /*ks1=*/key_0, /*b=*/5);
  return ctr;
}

// Implements the Philox4x32 counter-based PRNG algorithm. Use 10 rounds.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
std::array<uint32_t, 4> Philox4x32(uint32_t key_0, uint32_t key_1,
                                   std::array<uint32_t, 4> ctr) {
  // Compute the high and low words from multiplying two 32-bit integers.
  struct u32pair {
    uint32_t low;
    uint32_t high;
  };
  union prod {
    u32pair hilo;
    uint64_t prod;
  };
  for (int i = 0; i < 10; ++i) {
    prod p0, p1;
    p0.prod = kPhiloxM4x32A * static_cast<uint64_t>(ctr[0]);
    p1.prod = kPhiloxM4x32B * static_cast<uint64_t>(ctr[2]);
    ctr = {{p1.hilo.high ^ ctr[1] ^ key_0, p1.hilo.low,
            p0.hilo.high ^ ctr[3] ^ key_1, p0.hilo.low}};
    key_0 += kPhiloxW32A;
    key_1 += kPhiloxW32B;
  }
  return ctr;
}

}  // namespace rng
}  // namespace tflite
