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

// This file implements random_index_shuffle() by using a simple block chiper
// for pseudorandom permutations.
// This idea is described as cycle-walking cipher in
// https://www.cs.ucdavis.edu/~rogaway/papers/subset.pdf
//
// We use the Simon block cipher described in
// https://eprint.iacr.org/2013/404
// and following recommendations in
// https://nsacyber.github.io/simon-speck/implementations/ImplementationGuide1.1.pdf.
// However we use a single fixed key size and support arbtitary block sizes.
// Further we fixed the number of rounds in the Feistel structuro to be always
// 4. This reduces the computational cost and still gives good shuffle behavior.
//
// Warning: Given the modifications descripted above this implementation should
// not be used for application that require cryptograhic secure RNGs.

#include "tensorflow/core/kernels/random_index_shuffle.h"

#include <assert.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {
// Some of the macros below require a minimum word size of 8
// (2 word size = block size).
// Smaller block sizes might give poor results in terms of randomness.
constexpr int kMinBlockSize = 16;

namespace impl {

#define ROTL(x, r, W) (((x) << (r)) | (x >> (W - (r))))
#define ROTR(x, r, W) (((x) >> (r)) | ((x) << (W - (r))))
#define SIMON_F(x, W) ((ROTL(x, 1, W) & ROTL(x, 8, W) ^ ROTL(x, 2, W)))
#define SIMON_Rx2(x, y, k1, k2, W) \
  (y ^= SIMON_F(x, W), y ^= k1, x ^= SIMON_F(y, W), x ^= k2)

// Returns the keys per round for a Simon cipher.
// This variant uses std::bitset and can generate keys with any number of bits.
// This should not be used to encrypt data. It's not secure. We only use it
// to generate pseudorandom permutations.
template <int W>
std::vector<std::bitset<W>> simon_key_schedule(
    const std::array<uint32_t, 3>& key, const int32_t rounds) {
  // Required by ROTR/ROTL
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  const auto c = std::bitset<W>(0xfffffffc);
  auto z = std::bitset<W>(0x7369f885192c0ef5LL);
  std::vector<std::bitset<W>> rk({key[0], key[1], key[2]});
  rk.reserve(rounds);
  for (int i = 3; i < rounds; i++) {
    rk.push_back(c ^ (z & std::bitset<W>(1)) ^ rk[i - 3] ^
                 ROTR(rk[i - 1], 3, W) ^ ROTR(rk[i - 1], 4, W));
    z >>= 1;
  }
  return rk;
}

// Encrypts the given value using the Simon chipher.
// This is should not be used to encrypt data. It's not secure. We only use it
// to generate pseudorandom permutations.
template <int W>
uint64_t simon_encrypt(const uint64_t value,
                       const std::vector<std::bitset<W>>& round_keys) {
  // Required by ROTR/ROTL
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  std::bitset<W> left(value >> W);
  std::bitset<W> right(value);
  for (int i = 0; i < round_keys.size();) {
    SIMON_Rx2(right, left, round_keys[i++], round_keys[i++], W);
  }
  return (left.to_ullong() << W) | right.to_ullong();
}

// In the original implementation the number of rounds depends on the block
// size, key size and key words. For our purposes of random shuffle a 4 to 8
// rounds is enough.
// W = word size
// B = 2 * W = block size
template <int B>
uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index, const int32_t rounds) {
  const auto round_keys = simon_key_schedule<B / 2>(key, rounds);
  uint64_t new_index = index;
  while (true) {
    new_index = simon_encrypt<B / 2>(new_index, round_keys);
    if (new_index <= max_index) {
      return new_index;
    }
  }
}

#undef ROTL
#undef ROTR
#undef SIMON_F
#undef SIMON_RxC

}  // namespace impl

uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index, const int32_t rounds) {
  if (max_index == 0) {
    return 0;
  }
  // Block size must be large enough to represent max_index and even (since
  // word size is half of it). We force at least 16 bits as minimum block size
  // since we observed pattern in the permutations below.
  int block_size = static_cast<int>(std::ceil(std::log2(max_index)));
  block_size = std::max(block_size + block_size % 2, kMinBlockSize);
  assert(block_size > 0 && block_size % 2 == 0 && block_size <= 64);
  // At least 4 rounds and number of rounds must be even.
  assert(rounds >= 4 && rounds % 2 == 0);
#define HANDLE_BLOCK_SIZE(B) \
  case B:                    \
    return impl::index_shuffle<B>(index, key, max_index, rounds);

  switch (block_size) {
    HANDLE_BLOCK_SIZE(16);
    HANDLE_BLOCK_SIZE(18);
    HANDLE_BLOCK_SIZE(20);
    HANDLE_BLOCK_SIZE(22);
    HANDLE_BLOCK_SIZE(24);
    HANDLE_BLOCK_SIZE(26);
    HANDLE_BLOCK_SIZE(28);
    HANDLE_BLOCK_SIZE(30);
    HANDLE_BLOCK_SIZE(32);
    HANDLE_BLOCK_SIZE(34);
    HANDLE_BLOCK_SIZE(36);
    HANDLE_BLOCK_SIZE(38);
    HANDLE_BLOCK_SIZE(40);
    HANDLE_BLOCK_SIZE(42);
    HANDLE_BLOCK_SIZE(44);
    HANDLE_BLOCK_SIZE(46);
    HANDLE_BLOCK_SIZE(48);
    HANDLE_BLOCK_SIZE(50);
    HANDLE_BLOCK_SIZE(52);
    HANDLE_BLOCK_SIZE(54);
    HANDLE_BLOCK_SIZE(56);
    HANDLE_BLOCK_SIZE(58);
    HANDLE_BLOCK_SIZE(60);
    HANDLE_BLOCK_SIZE(62);
    default:
      return impl::index_shuffle<64>(index, key, max_index, rounds);
  }
#undef HANDLE_BLOCK_SIZE
}

}  // namespace random
}  // namespace tensorflow
