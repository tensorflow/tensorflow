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

#include "tensorflow/tsl/platform/base64.h"

#include <cstring>
#include <memory>

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/stringpiece.h"

namespace tsl {
namespace {
// This array must have signed type.
// clang-format off
constexpr int8 kBase64Bytes[128] = {
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  0x3E,  -1,   -1,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D,  -1,   -1,
     -1,   -1,   -1,   -1,   -1,  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,  -1,   -1,   -1,   -1,  0x3F,
     -1,  0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
    0x31, 0x32, 0x33,  -1,   -1,   -1,   -1,   -1};
// clang-format on

constexpr char kBase64UrlSafeChars[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

constexpr char kPadChar = '=';

// Converts a char (8 bits) into a 6-bit value for decoding. If the input char
// is invalid for base64 encoding, the return value has at least its upper 25
// bits set.
inline uint32 Convert(char x) {
  // If x < 128, then we look up x in the table. If x is valid, then the table
  // will have a value <= 0x3F, otherwise the table will have -1. If x >= 128,
  // we still do some table lookup, but the value is ignored since we explicitly
  // set the high bit of y to 1. Either way, y is negative (high bit set) in
  // case of error.
  const int8_t y = kBase64Bytes[x & 0x7F] | (x & 0x80);
  // Casting from int8 to int32 preserves sign by sign extension. If y was
  // negative, at least its 25 high bits of the return value are set.
  const int32_t z = static_cast<int32>(y);
  return static_cast<uint32>(z);
}

Status DecodeThreeChars(const char* codes, char* result) {
  const uint32 packed = (Convert(codes[0]) << 18) | (Convert(codes[1]) << 12) |
                        (Convert(codes[2]) << 6) | (Convert(codes[3]));
  // Convert() return value has upper 25 bits set if input is invalid.
  // Therefore `packed` has high bits set iff at least one of code is invalid.
  if (TF_PREDICT_FALSE((packed & 0xFF000000) != 0)) {
    return errors::InvalidArgument("Invalid character found in base64.");
  }
  result[0] = static_cast<char>(packed >> 16);
  result[1] = static_cast<char>(packed >> 8);
  result[2] = static_cast<char>(packed);
  return OkStatus();
}
}  // namespace

template <typename T>
Status Base64Decode(StringPiece data, T* decoded) {
  if (decoded == nullptr) {
    return errors::Internal("'decoded' cannot be nullptr.");
  }

  if (data.empty()) {
    decoded->clear();
    return OkStatus();
  }

  // This decoding procedure will write 3 * ceil(data.size() / 4) bytes to be
  // output buffer, then truncate if necessary. Therefore we must overestimate
  // and allocate sufficient amount. Currently max_decoded_size may overestimate
  // by up to 3 bytes.
  const size_t max_decoded_size = 3 * (data.size() / 4) + 3;
  std::unique_ptr<char[]> buffer(new char[max_decoded_size]);
  char* current = buffer.get();
  if (current == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate buffer for decoded string.");
  }

  const char* b64 = data.data();
  const char* end = data.data() + data.size();

  while (end - b64 > 4) {
    TF_RETURN_IF_ERROR(DecodeThreeChars(b64, current));
    b64 += 4;
    current += 3;
  }

  if (end - b64 == 4) {
    // The data length is a multiple of 4. Check for padding.
    // Base64 cannot have more than 2 paddings.
    if (b64[2] == kPadChar && b64[3] == kPadChar) {
      end -= 2;
    }
    if (b64[2] != kPadChar && b64[3] == kPadChar) {
      end -= 1;
    }
  }

  const int remain = static_cast<int>(end - b64);
  if (TF_PREDICT_FALSE(remain == 1)) {
    // We may check this condition early by checking data.size() % 4 == 1.
    return errors::InvalidArgument(
        "Base64 string length cannot be 1 modulo 4.");
  }

  // A valid base64 character will replace paddings, if any.
  char tail[4] = {kBase64UrlSafeChars[0], kBase64UrlSafeChars[0],
                  kBase64UrlSafeChars[0], kBase64UrlSafeChars[0]};
  // Copy tail of the input into the array, then decode.
  std::memcpy(tail, b64, remain * sizeof(*b64));
  TF_RETURN_IF_ERROR(DecodeThreeChars(tail, current));
  // We know how many parsed characters are valid.
  current += remain - 1;

  decoded->assign(buffer.get(), current - buffer.get());
  return OkStatus();
}

template <typename T>
Status Base64Encode(StringPiece source, T* encoded) {
  return Base64Encode(source, false, encoded);
}

template <typename T>
Status Base64Encode(StringPiece source, bool with_padding, T* encoded) {
  const char* const base64_chars = kBase64UrlSafeChars;
  if (encoded == nullptr) {
    return errors::Internal("'encoded' cannot be nullptr.");
  }

  // max_encoded_size may overestimate by up to 4 bytes.
  const size_t max_encoded_size = 4 * (source.size() / 3) + 4;
  std::unique_ptr<char[]> buffer(new char[max_encoded_size]);
  char* current = buffer.get();
  if (current == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate buffer for encoded string.");
  }

  const char* data = source.data();
  const char* const end = source.data() + source.size();

  // Encode each block.
  while (end - data >= 3) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[0] & 0x03) << 4) | ((data[1] >> 4) & 0x0F)];
    *current++ =
        base64_chars[((data[1] & 0x0F) << 2) | ((data[2] >> 6) & 0x03)];
    *current++ = base64_chars[data[2] & 0x3F];

    data += 3;
  }

  // Take care of the tail.
  if (end - data == 2) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[0] & 0x03) << 4) | ((data[1] >> 4) & 0x0F)];
    *current++ = base64_chars[(data[1] & 0x0F) << 2];
    if (with_padding) {
      *current++ = kPadChar;
    }
  } else if (end - data == 1) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ = base64_chars[(data[0] & 0x03) << 4];
    if (with_padding) {
      *current++ = kPadChar;
      *current++ = kPadChar;
    }
  }

  encoded->assign(buffer.get(), current - buffer.get());
  return OkStatus();
}

template Status Base64Decode<std::string>(StringPiece data,
                                          std::string* decoded);
template Status Base64Encode<std::string>(StringPiece source,
                                          std::string* encoded);
template Status Base64Encode<std::string>(StringPiece source, bool with_padding,
                                          std::string* encoded);

template Status Base64Decode<tstring>(StringPiece data, tstring* decoded);
template Status Base64Encode<tstring>(StringPiece source, tstring* encoded);
template Status Base64Encode<tstring>(StringPiece source, bool with_padding,
                                      tstring* encoded);

}  // namespace tsl
