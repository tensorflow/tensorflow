/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/ctstring.h"

#include <memory>
#include <string>

#include "tensorflow/core/platform/ctstring_internal.h"
#include "tensorflow/core/platform/test.h"

static const char kLongString[] =
    "abcdefghij"
    "klmnopqrst"
    "uvwxyz0123"
    "456789ABCD"
    "EFGHIKLMNO";
const size_t kLongStringLen = sizeof(kLongString) / sizeof(char) - sizeof(char);

TEST(TF_CTStringTest, InitAssignMoveDealloc) {
  EXPECT_GT(::strlen(kLongString), TF_TString_SmallCapacity);

  {
    // Empty String
    TF_TString s10, s11, s12;
    TF_TString_Init(&s10);
    TF_TString_Init(&s11);
    TF_TString_Init(&s12);

    EXPECT_EQ(0, TF_TString_GetSize(&s10));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s10));
    EXPECT_STREQ("", TF_TString_GetDataPointer(&s10));
    EXPECT_STREQ("", TF_TString_GetMutableDataPointer(&s10));

    TF_TString_Assign(&s11, &s10);

    EXPECT_EQ(0, TF_TString_GetSize(&s11));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s10));
    EXPECT_STREQ("", TF_TString_GetDataPointer(&s11));
    EXPECT_STREQ("", TF_TString_GetMutableDataPointer(&s11));

    TF_TString_Move(&s12, &s11);

    EXPECT_EQ(0, TF_TString_GetSize(&s11));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s10));
    EXPECT_STREQ("", TF_TString_GetDataPointer(&s11));
    EXPECT_STREQ("", TF_TString_GetMutableDataPointer(&s11));

    EXPECT_EQ(0, TF_TString_GetSize(&s12));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s10));
    EXPECT_STREQ("", TF_TString_GetDataPointer(&s12));
    EXPECT_STREQ("", TF_TString_GetMutableDataPointer(&s12));

    TF_TString_Dealloc(&s10);
    TF_TString_Dealloc(&s11);
    TF_TString_Dealloc(&s12);
  }

  {
    // Small String
    TF_TString s20, s21, s22;
    TF_TString_Init(&s20);
    TF_TString_Init(&s21);
    TF_TString_Init(&s22);

    TF_TString_Copy(&s20, "a", 1);

    EXPECT_EQ(1, TF_TString_GetSize(&s20));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s20));
    EXPECT_STREQ("a", TF_TString_GetDataPointer(&s20));
    EXPECT_STREQ("a", TF_TString_GetMutableDataPointer(&s20));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s20));

    TF_TString_Assign(&s21, &s20);

    EXPECT_EQ(1, TF_TString_GetSize(&s21));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s21));
    EXPECT_STREQ("a", TF_TString_GetDataPointer(&s21));
    EXPECT_STREQ("a", TF_TString_GetMutableDataPointer(&s21));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s21));

    TF_TString_Move(&s22, &s21);

    EXPECT_EQ(1, TF_TString_GetSize(&s22));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s22));
    EXPECT_STREQ("a", TF_TString_GetDataPointer(&s22));
    EXPECT_STREQ("a", TF_TString_GetMutableDataPointer(&s22));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s22));

    TF_TString_Dealloc(&s20);
    TF_TString_Dealloc(&s21);  // Nothing to dealloc, since it was moved.
    TF_TString_Dealloc(&s22);
  }

  {
    // Small String -> Large String and View
    TF_TString s30, s31;
    TF_TString_Init(&s30);
    TF_TString_Init(&s31);

    size_t s = TF_TString_SmallCapacity - 1;

    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s30));

    // Small String
    TF_TString_Copy(&s30, kLongString, s);

    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetDataPointer(&s30));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s30));
    EXPECT_GT(TF_TString_SmallCapacity, TF_TString_GetSize(&s30));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s30));

    // Small String at capacity
    TF_TString_AppendN(&s30, &kLongString[s++], 1);

    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetDataPointer(&s30));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s30));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetSize(&s30));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s30));

    // Large String
    TF_TString_AppendN(&s30, &kLongString[s++], 1);

    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetDataPointer(&s30));
    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetMutableDataPointer(&s30));
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s30));
    EXPECT_EQ(s, TF_TString_GetSize(&s30));
    EXPECT_LT(TF_TString_SmallCapacity, TF_TString_GetSize(&s30));
    EXPECT_LT(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s30));

    // Large String Move
    TF_TString_Move(&s31, &s30);

    EXPECT_STREQ("", TF_TString_GetDataPointer(&s30));
    EXPECT_STREQ("", TF_TString_GetMutableDataPointer(&s30));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s30));
    EXPECT_EQ(0, TF_TString_GetSize(&s30));

    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetDataPointer(&s31));
    EXPECT_STREQ(std::string(kLongString, s).data(),
                 TF_TString_GetMutableDataPointer(&s31));
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s31));
    EXPECT_EQ(s, TF_TString_GetSize(&s31));
    EXPECT_LT(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s31));

    TF_TString_Dealloc(&s30);
    TF_TString_Dealloc(&s31);
  }

  {
    // Small String -> Large String -> Larger -> View
    const char kStr[] = "abcdef";
    const char kStrLen = sizeof(kStr) / sizeof(char) - sizeof(char);
    TF_TString s40, s41;

    TF_TString_Init(&s40);
    TF_TString_Init(&s41);

    TF_TString_Copy(&s40, kLongString, kLongStringLen);

    EXPECT_EQ(kLongStringLen, TF_TString_GetSize(&s40));

    TF_TString_Assign(&s41, &s40);

    EXPECT_STREQ(kLongString, TF_TString_GetDataPointer(&s40));
    EXPECT_STREQ(kLongString, TF_TString_GetMutableDataPointer(&s40));
    EXPECT_EQ(kLongStringLen, TF_TString_GetSize(&s41));

    TF_TString_AppendN(&s40, kLongString, kLongStringLen);
    TF_TString_Append(&s40, &s41);

    std::string longerString(kLongString);
    longerString += kLongString;
    longerString += kLongString;
    EXPECT_STREQ(longerString.data(), TF_TString_GetDataPointer(&s40));
    EXPECT_STREQ(longerString.data(), TF_TString_GetMutableDataPointer(&s40));
    EXPECT_EQ(longerString.size(), TF_TString_GetSize(&s40));

    TF_TString_AssignView(&s40, kStr, kStrLen);

    EXPECT_EQ(TF_TSTR_VIEW, TF_TString_GetType(&s40));
    EXPECT_EQ(kStr, TF_TString_GetDataPointer(&s40));
    EXPECT_EQ(6, TF_TString_GetSize(&s40));
    EXPECT_EQ(0, TF_TString_GetCapacity(&s40));

    EXPECT_NE(kStr, TF_TString_GetMutableDataPointer(&s40));
    EXPECT_STREQ(kStr, TF_TString_GetMutableDataPointer(&s40));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s40));
    EXPECT_EQ(6, TF_TString_GetSize(&s40));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s40));

    TF_TString_Dealloc(&s40);
    TF_TString_Dealloc(&s41);
  }

  {
    // Small String -> Large String -> Smaller
    TF_TString s50;

    TF_TString_Init(&s50);

    TF_TString_Copy(&s50, "a", 1);

    EXPECT_STREQ("a", TF_TString_GetDataPointer(&s50));
    EXPECT_STREQ("a", TF_TString_GetMutableDataPointer(&s50));
    EXPECT_EQ(1, TF_TString_GetSize(&s50));

    TF_TString_Copy(&s50, kLongString, kLongStringLen);

    EXPECT_STREQ(kLongString, TF_TString_GetDataPointer(&s50));
    EXPECT_STREQ(kLongString, TF_TString_GetMutableDataPointer(&s50));
    EXPECT_EQ(kLongStringLen, TF_TString_GetSize(&s50));

    // align16(kLongStringLen) - 1 = 63
    size_t cap1 = TF_TString_GetCapacity(&s50);

    // Test reduced allocation with on large type.
    TF_TString_Copy(&s50, kLongString, TF_TString_SmallCapacity + 1);

    // align16(TF_TString_SmallCapacity+1) - 1 = 31
    size_t cap2 = TF_TString_GetCapacity(&s50);

    EXPECT_STREQ(std::string(kLongString, TF_TString_SmallCapacity + 1).data(),
                 TF_TString_GetMutableDataPointer(&s50));
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s50));

    EXPECT_GT(cap1, cap2);

    TF_TString_Copy(&s50, "c", 1);

    EXPECT_STREQ("c", TF_TString_GetDataPointer(&s50));
    EXPECT_STREQ("c", TF_TString_GetMutableDataPointer(&s50));
    EXPECT_EQ(1, TF_TString_GetSize(&s50));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s50));

    TF_TString_Dealloc(&s50);
  }
}

TEST(TF_CTStringTest, ResizeReserve) {
  {
    // Resize
    TF_TString s60;

    TF_TString_Init(&s60);

    TF_TString_Resize(&s60, 2, 'a');

    EXPECT_EQ(0, ::memcmp("aa", TF_TString_GetDataPointer(&s60), 2));

    TF_TString_Resize(&s60, 4, '\0');

    EXPECT_EQ(0, ::memcmp("aa\0\0", TF_TString_GetDataPointer(&s60), 4));

    TF_TString_Resize(&s60, 6, 'b');

    EXPECT_EQ(0, ::memcmp("aa\0\0bb", TF_TString_GetDataPointer(&s60), 6));

    TF_TString_Resize(&s60, 2, 'c');

    EXPECT_EQ(0, ::memcmp("aa", TF_TString_GetDataPointer(&s60), 2));

    TF_TString_Dealloc(&s60);
  }
  {
    // Reserve
    TF_TString s70;

    TF_TString_Init(&s70);

    TF_TString_Reserve(&s70, TF_TString_SmallCapacity - 1);

    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(0, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_Reserve(&s70, TF_TString_SmallCapacity);

    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(0, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_Copy(&s70, "hello", 5);

    EXPECT_EQ(5, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_Reserve(&s70, 100);

    // Test 16 byte alignment (7*16 - 1 = 111)
    EXPECT_EQ(111, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(5, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));

    TF_TString_AssignView(&s70, kLongString, kLongStringLen);
    TF_TString_Reserve(&s70, 10);

    EXPECT_EQ(TF_TSTR_VIEW, TF_TString_GetType(&s70));
    EXPECT_EQ(0, TF_TString_GetCapacity(&s70));

    TF_TString_Reserve(&s70, 100);

    // Converted to LARGE since it can no longer fit in SMALL.
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));
    EXPECT_EQ(111, TF_TString_GetCapacity(&s70));

    TF_TString_Reserve(&s70, 200);

    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));
    EXPECT_EQ(207, TF_TString_GetCapacity(&s70));

    TF_TString_Dealloc(&s70);
  }
  {
    // ReserveAmortized
    TF_TString s70;

    TF_TString_Init(&s70);

    TF_TString_ReserveAmortized(&s70, TF_TString_SmallCapacity - 1);

    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(0, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_ReserveAmortized(&s70, TF_TString_SmallCapacity);

    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(0, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_Copy(&s70, "hello", 5);

    EXPECT_EQ(5, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TString_SmallCapacity, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(TF_TSTR_SMALL, TF_TString_GetType(&s70));

    TF_TString_ReserveAmortized(&s70, 100);

    // Test 16 byte alignment (7*16 - 1 = 111)
    EXPECT_EQ(111, TF_TString_GetCapacity(&s70));
    EXPECT_EQ(5, TF_TString_GetSize(&s70));
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));

    TF_TString_AssignView(&s70, kLongString, kLongStringLen);
    TF_TString_ReserveAmortized(&s70, 10);

    EXPECT_EQ(TF_TSTR_VIEW, TF_TString_GetType(&s70));
    EXPECT_EQ(0, TF_TString_GetCapacity(&s70));

    TF_TString_ReserveAmortized(&s70, 100);

    // Converted to LARGE since it can no longer fit in SMALL.
    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));
    EXPECT_EQ(111, TF_TString_GetCapacity(&s70));

    TF_TString_ReserveAmortized(&s70, 200);

    EXPECT_EQ(TF_TSTR_LARGE, TF_TString_GetType(&s70));
    // 223 = 2*previous_capacity+1
    EXPECT_EQ(223, TF_TString_GetCapacity(&s70));

    TF_TString_Dealloc(&s70);
  }
}

TEST(TF_CTStringTest, OffsetType) {
  {
    uint8_t str[] = "test";
    constexpr size_t str_size = sizeof(str) / sizeof(str[0]);

    uint8_t buf[sizeof(TF_TString) + str_size];

    memcpy(buf + sizeof(TF_TString), str, str_size);

    TF_TString *offsets = (TF_TString *)buf;
    TF_TString_Init(offsets);
    // using existing TF_le32toh to achieve htole32
    offsets[0].u.offset.size = TF_le32toh(str_size << 2 | TF_TSTR_OFFSET);
    offsets[0].u.offset.offset = TF_le32toh(sizeof(TF_TString));
    offsets[0].u.offset.count = TF_le32toh(1);

    EXPECT_EQ(str_size, TF_TString_GetSize(offsets));
    EXPECT_EQ(TF_TSTR_OFFSET, TF_TString_GetType(offsets));
    EXPECT_EQ(0, ::memcmp(str, TF_TString_GetDataPointer(offsets), str_size));
  }
}
