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

#include "xla/tsl/platform/sha256.h"

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

SHA256::SHA256() { Reset(); }

void SHA256::Reset() { SHA256_Init(&ctx_); }

void SHA256::Update(absl::string_view data) {
  Update(data.data(), data.size());
}

void SHA256::Update(const void* data, size_t size) {
  SHA256_Update(&ctx_, data, size);
}

std::array<uint8_t, SHA256::kDigestSize> SHA256::Digest() {
  std::array<uint8_t, kDigestSize> digest;
  SHA256_CTX temp_ctx = ctx_;
  SHA256_Final(digest.data(), &temp_ctx);
  return digest;
}

std::string SHA256::DigestString() {
  std::array<uint8_t, kDigestSize> digest = Digest();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

std::array<uint8_t, SHA256::kDigestSize> SHA256::Hash(absl::string_view data) {
  return Hash(data.data(), data.size());
}

std::array<uint8_t, SHA256::kDigestSize> SHA256::Hash(const void* data,
                                                      size_t size) {
  std::array<uint8_t, kDigestSize> digest;
  ::SHA256(reinterpret_cast<const uint8_t*>(data), size, digest.data());
  return digest;
}

}  // namespace tsl
