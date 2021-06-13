/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/experimental/type_inference.h"

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr int TABLE_PAD_SIZE = 3;

std::string Pad(int n, std::string s) {
  int to_add = std::max(0, n - static_cast<int>(s.size()));
  for (int i = 0; i < to_add; i++) s += " ";
  return s;
}

TEST(TypeInfereceTest, LatticeBuild) {
  std::vector<full_type::Type> types = {
      TFT_BOOL,
      TFT_UINT8,
      TFT_UINT16,
      TFT_UINT32,
      TFT_UINT64,
      TFT_INT8,
      TFT_INT16,
      TFT_INT32,
      TFT_INT64,
      TFT_BFLOAT16,
      TFT_HALF,
      TFT_FLOAT,
      TFT_DOUBLE,
      TFT_COMPLEX64,
      TFT_COMPLEX128,
      full_type::TFT_BOOL_WEAK,
      full_type::TFT_INT_WEAK,
      full_type::TFT_FLOAT_WEAK,
      full_type::TFT_COMPLEX_WEAK,
  };

  // Produce a table of the cartesian product promotion of `types`.
  std::string table = Pad(TABLE_PAD_SIZE, "");
  for (int i = 0; i < types.size(); i++) {
    table += Pad(TABLE_PAD_SIZE, full_type::ShortName(types[i]));
  }
  table += "\n";
  for (int i = 0; i < types.size(); i++) {
    auto ti = types[i];
    table += Pad(TABLE_PAD_SIZE, full_type::ShortName(ti));
    for (int j = 0; j < types.size(); j++) {
      auto tj = types[j];
      auto tr = full_type::ReturnType(ti, tj);
      table += Pad(TABLE_PAD_SIZE, full_type::ShortName(tr));
    }
    table += "\n";
  }
  // Expected table is in text format for easy understanding.
  ASSERT_EQ(table,
            "   b  u1 u2 u4 u8 i1 i2 i4 i8 bf f2 f4 f8 c4 c8 b* i* f* c* \n"
            "b  b  u1 u2 u4 u8 i1 i2 i4 i8 bf f2 f4 f8 c4 c8 b  i4 f4 c4 \n"
            "u1 u1 u1 u2 u4 u8 i2 i2 i4 i8 bf f2 f4 f8 c4 c8 u1 u1 f4 c4 \n"
            "u2 u2 u2 u2 u4 u8 i4 i4 i4 i8 bf f2 f4 f8 c4 c8 u2 u2 f4 c4 \n"
            "u4 u4 u4 u4 u4 u8 i8 i8 i8 i8 bf f2 f4 f8 c4 c8 u4 u4 f4 c4 \n"
            "u8 u8 u8 u8 u8 u8 f4 f4 f4 f4 bf f2 f4 f8 c4 c8 u8 u8 f4 c4 \n"
            "i1 i1 i2 i4 i8 f4 i1 i2 i4 i8 bf f2 f4 f8 c4 c8 i1 i1 f4 c4 \n"
            "i2 i2 i2 i4 i8 f4 i2 i2 i4 i8 bf f2 f4 f8 c4 c8 i2 i2 f4 c4 \n"
            "i4 i4 i4 i4 i8 f4 i4 i4 i4 i8 bf f2 f4 f8 c4 c8 i4 i4 f4 c4 \n"
            "i8 i8 i8 i8 i8 f4 i8 i8 i8 i8 bf f2 f4 f8 c4 c8 i8 i8 f4 c4 \n"
            "bf bf bf bf bf bf bf bf bf bf bf f4 f4 f8 c4 c8 bf bf bf c4 \n"
            "f2 f2 f2 f2 f2 f2 f2 f2 f2 f2 f4 f2 f4 f8 c4 c8 f2 f2 f2 c4 \n"
            "f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f8 c4 c8 f4 f4 f4 c4 \n"
            "f8 f8 f8 f8 f8 f8 f8 f8 f8 f8 f8 f8 f8 f8 c8 c8 f8 f8 f8 c8 \n"
            "c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c8 c4 c8 c4 c4 c4 c4 \n"
            "c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 c8 \n"
            "b* b  u1 u2 u4 u8 i1 i2 i4 i8 bf f2 f4 f8 c4 c8 b  i4 f4 c4 \n"
            "i* i4 u1 u2 u4 u8 i1 i2 i4 i8 bf f2 f4 f8 c4 c8 i4 i4 f4 c4 \n"
            "f* f4 f4 f4 f4 f4 f4 f4 f4 f4 bf f2 f4 f8 c4 c8 f4 f4 f4 c4 \n"
            "c* c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c4 c8 c4 c8 c4 c4 c4 c4 \n");
}

}  // namespace
}  // namespace tensorflow
