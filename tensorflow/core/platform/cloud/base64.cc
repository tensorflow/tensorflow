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

#include "tensorflow/core/platform/cloud/base64.h"
#include <memory>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

constexpr signed char kBase64Bytes[] = {
    -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
    -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
    -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
    -1,   -1,   -1,   -1,   -1,   -1,   -1,   0x3E, -1,   -1,   -1,   0x3F,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, -1,   -1,
    -1,   0x7F, -1,   -1,   -1,   0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, -1,   -1,   -1,   -1,   -1,
    -1,   0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
    0x31, 0x32, 0x33, -1,   -1,   -1,   -1,   -1};

constexpr char kBase64UrlSafeChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

constexpr char kPadChar = '=';
constexpr char kPadByte = 0x7F;
constexpr int kMultilineLineLen = 76;
constexpr int kMultilineNumBlocks = kMultilineLineLen / 4;

Status Base64Encode(StringPiece source, bool multiline, bool with_padding,
                    string *encoded) {
  if (!encoded) {
    return errors::FailedPrecondition("'encoded' cannot be nullptr.");
  }
  size_t data_size = source.size();
  const char *data = source.data();
  const char *base64_chars = kBase64UrlSafeChars;
  const size_t result_projected_size =
      4 * ((data_size + 3) / 3) +
      2 * (multiline ? (data_size / (3 * kMultilineNumBlocks)) : 0) + 1;
  size_t num_blocks = 0;
  size_t i = 0;
  std::unique_ptr<char[]> result(new char[result_projected_size]);
  char *current = result.get();

  /* Encode each block. */
  while (data_size >= 3) {
    *current++ = base64_chars[(data[i] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[i] & 0x03) << 4) | ((data[i + 1] >> 4) & 0x0F)];
    *current++ =
        base64_chars[((data[i + 1] & 0x0F) << 2) | ((data[i + 2] >> 6) & 0x03)];
    *current++ = base64_chars[data[i + 2] & 0x3F];

    data_size -= 3;
    i += 3;
    if (multiline && (++num_blocks == kMultilineNumBlocks)) {
      *current++ = '\r';
      *current++ = '\n';
      num_blocks = 0;
    }
  }

  /* Take care of the tail. */
  if (data_size == 2) {
    *current++ = base64_chars[(data[i] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[i] & 0x03) << 4) | ((data[i + 1] >> 4) & 0x0F)];
    *current++ = base64_chars[(data[i + 1] & 0x0F) << 2];
    if (with_padding) {
      *current++ = kPadChar;
    }
  } else if (data_size == 1) {
    *current++ = base64_chars[(data[i] >> 2) & 0x3F];
    *current++ = base64_chars[(data[i] & 0x03) << 4];
    if (with_padding) {
      *current++ = kPadChar;
      *current++ = kPadChar;
    }
  }

  if (current < result.get() ||
      current >= result.get() + result_projected_size) {
    return errors::Internal("Unexpected encoding bug.");
  }
  *current++ = '\0';
  *encoded = result.get();
  return Status::OK();
}

void DecodeOneChar(const unsigned char *codes, unsigned char *result,
                   size_t *result_offset) {
  const uint32_t packed = ((uint32_t)codes[0] << 2) | ((uint32_t)codes[1] >> 4);
  result[(*result_offset)++] = (unsigned char)packed;
}

void DecodeTwoChars(const unsigned char *codes, unsigned char *result,
                    size_t *result_offset) {
  const uint32_t packed = ((uint32_t)codes[0] << 10) |
                          ((uint32_t)codes[1] << 4) | ((uint32_t)codes[2] >> 2);
  result[(*result_offset)++] = (unsigned char)(packed >> 8);
  result[(*result_offset)++] = (unsigned char)(packed);
}

Status DecodeGroup(const unsigned char *codes, size_t num_codes,
                   unsigned char *result, size_t *result_offset) {
  if (num_codes > 4) {
    return errors::FailedPrecondition("Expected 4 or fewer codes.");
  }

  /* Short end groups that may not have padding. */
  if (num_codes == 1) {
    return errors::FailedPrecondition(
        "Invalid group. Must be at least 2 bytes.");
  }
  if (num_codes == 2) {
    DecodeOneChar(codes, result, result_offset);
    return Status::OK();
  }
  if (num_codes == 3) {
    DecodeTwoChars(codes, result, result_offset);
    return Status::OK();
  }

  /* Regular 4 byte groups with padding or not. */
  if (num_codes != 4) {
    return errors::FailedPrecondition("Expected exactly 4 codes.");
  }
  if (codes[0] == kPadByte || codes[1] == kPadByte) {
    return errors::FailedPrecondition("Invalid padding detected.");
  }
  if (codes[2] == kPadByte) {
    if (codes[3] == kPadByte) {
      DecodeOneChar(codes, result, result_offset);
    } else {
      return errors::FailedPrecondition("Invalid padding detected.");
    }
  } else if (codes[3] == kPadByte) {
    DecodeTwoChars(codes, result, result_offset);
  } else {
    /* No padding. */
    const uint32_t packed = ((uint32_t)codes[0] << 18) |
                            ((uint32_t)codes[1] << 12) |
                            ((uint32_t)codes[2] << 6) | codes[3];
    result[(*result_offset)++] = (unsigned char)(packed >> 16);
    result[(*result_offset)++] = (unsigned char)(packed >> 8);
    result[(*result_offset)++] = (unsigned char)(packed);
  }
  return Status::OK();
}

}  // namespace

Status Base64Encode(StringPiece source, string *encoded) {
  return Base64Encode(source, false, false, encoded);
}

Status Base64Decode(StringPiece data, string *decoded) {
  if (!decoded) {
    return errors::FailedPrecondition("'decoded' cannot be nullptr.");
  }
  std::unique_ptr<unsigned char[]> result(new unsigned char[data.size()]);
  unsigned char *current = result.get();
  size_t result_size = 0;
  unsigned char codes[4];
  size_t num_codes = 0;

  const char *b64 = data.data();
  size_t b64_len = data.size();
  while (b64_len--) {
    unsigned char c = (unsigned char)(*b64++);
    signed char code;
    if (c >= sizeof(kBase64Bytes)) continue;
    if (c == '+' || c == '/') {
      return errors::FailedPrecondition(
          strings::StrCat("Invalid character for url safe base64 ", c));
    }
    if (c == '-') {
      c = '+';
    } else if (c == '_') {
      c = '/';
    }
    code = kBase64Bytes[c];
    if (code == -1) {
      if (c != '\r' && c != '\n') {
        return errors::FailedPrecondition(
            strings::StrCat("Invalid character ", c));
      }
    } else {
      codes[num_codes++] = (unsigned char)code;
      if (num_codes == 4) {
        TF_RETURN_IF_ERROR(
            DecodeGroup(codes, num_codes, current, &result_size));
        num_codes = 0;
      }
    }
  }

  if (num_codes != 0) {
    TF_RETURN_IF_ERROR(DecodeGroup(codes, num_codes, current, &result_size));
  }
  *decoded = string(reinterpret_cast<char *>(result.get()), result_size);
  return Status::OK();
}

}  // namespace tensorflow
