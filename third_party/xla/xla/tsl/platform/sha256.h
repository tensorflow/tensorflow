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

#ifndef XLA_TSL_PLATFORM_SHA256_H_
#define XLA_TSL_PLATFORM_SHA256_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "tsl/platform/platform.h"

// In BoringSSL, sha.h includes deprecated SHA-1 symbols and sha2.h
// is recommended for SHA-256. In standard OpenSSL and open-source builds,
// sha2.h does not exist and SHA-256 symbols are declared in sha.h.
#if defined(PLATFORM_GOOGLE)
#include "openssl/sha2.h"
#else
#include "openssl/sha.h"
#endif

namespace tsl {

// Lightweight SHA-256 cryptographic hash wrapper backed by BoringSSL.
//
// Utilizing BoringSSL provides hardware-accelerated SHA-256 performance
// (SHA-NI on x86_64, ARMv8 SHA2 extensions on ARM64) while remaining
// compatible with OpenXLA and LibTPU open-source export requirements.
class SHA256 {
 public:
  static constexpr int kDigestSize = SHA256_DIGEST_LENGTH;

  SHA256();

  // Non-copyable and non-movable to ensure safe context lifecycle.
  SHA256(const SHA256&) = delete;
  SHA256& operator=(const SHA256&) = delete;

  // Reinitialize the internal hash state.
  void Reset();

  // Digest more data.
  void Update(absl::string_view data);
  void Update(const void* data, size_t size);

  // Return the current raw 256-bit SHA-256 digest for all data digested
  // since the last call to Reset() or constructor.
  std::array<uint8_t, kDigestSize> Digest();

  // Return the raw 256-bit SHA-256 digest as a binary string.
  std::string DigestString();

  // One-shot helper to compute the raw 256-bit SHA-256 digest for the
  // given data.
  static std::array<uint8_t, kDigestSize> Hash(absl::string_view data);
  static std::array<uint8_t, kDigestSize> Hash(const void* data, size_t size);

 private:
  SHA256_CTX ctx_;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_SHA256_H_
