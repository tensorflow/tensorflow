// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

using litert::BufferRef;
using litert::Mallocator;
using litert::MutableBufferRef;
using litert::Newlocator;
using litert::OwningBufferRef;
using litert::internal::FbBufToStr;
using testing::ElementsAreArray;
using testing::Eq;
using testing::Pointwise;
using testing::StartsWith;

namespace {

static constexpr size_t kOffset = 4;

static constexpr absl::string_view kData = "SomeRawBuffer";
static constexpr absl::string_view kOtherData = "SOMERawBuffer";

absl::Span<const uint8_t> MakeConstFbData(absl::string_view data) {
  const uint8_t* fb_data = reinterpret_cast<const uint8_t*>(data.data());
  return absl::MakeConstSpan(fb_data, data.size());
}

absl::Span<uint8_t> MakeFbData(absl::string_view data) {
  const uint8_t* c_fb_data = reinterpret_cast<const uint8_t*>(data.data());
  uint8_t* fb_data = const_cast<uint8_t*>(c_fb_data);
  return absl::MakeSpan(fb_data, data.size());
}

std::vector<uint8_t> MakeFbDataVec(absl::string_view data) {
  const uint8_t* c_fb_data = reinterpret_cast<const uint8_t*>(data.data());
  uint8_t* fb_data = const_cast<uint8_t*>(c_fb_data);
  return std::vector<uint8_t>(fb_data, fb_data + data.size());
}

template <class Allocator = Newlocator<uint8_t>, typename ByteT = uint8_t>
absl::Span<ByteT> MakeInternalTestBuffer(absl::string_view data) {
  ByteT* buffer = Allocator()(data.size());
  std::memcpy(buffer, data.data(), data.size());
  return absl::MakeSpan(reinterpret_cast<ByteT*>(buffer), data.size());
}

//
// flatbuffer_tools.h
//

TEST(FbBufToStringTest, ConstSpan) {
  EXPECT_THAT(FbBufToStr(MakeConstFbData(kData)), Pointwise(Eq(), kData));
}

TEST(FbBufToStringTest, Span) {
  EXPECT_THAT(FbBufToStr(MakeFbData(kData)), Pointwise(Eq(), kData));
}

TEST(FbBufToStringTest, ConstPointer) {
  auto data = MakeConstFbData(kData);
  EXPECT_THAT(FbBufToStr(data.data(), data.size()), Pointwise(Eq(), kData));
}

TEST(FbBufToStringTest, Pointer) {
  auto data = MakeFbData(kData);
  EXPECT_THAT(FbBufToStr(data.data(), data.size()), Pointwise(Eq(), kData));
}

//
// BufferRef (read-only)
//

TEST(BufferRefTest, Dump) {
  BufferRef buf(kData.data(), kData.size());
  std::stringstream out;
  buf.Dump(out);
  EXPECT_THAT(out.str(), StartsWith("BufferRef"));
}

TEST(BufferRefTest, WithData) {
  auto data = MakeConstFbData(kData);
  BufferRef buf(data.data(), data.size());
  EXPECT_EQ(buf.Span(), data);
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(BufferRefTest, WithDataAndOffset) {
  auto data = MakeConstFbData(kData);
  BufferRef buf(data.data(), data.size(), kOffset);
  EXPECT_EQ(buf.Span(), data.subspan(kOffset, buf.Size()));
  EXPECT_EQ(buf.StrView(), kData.substr(kOffset, buf.Size()));
}

TEST(BufferRefTest, ToVec) {
  auto data = MakeConstFbData(kData);
  BufferRef buf(data.data(), data.size());
  EXPECT_THAT(buf.ToVec(), ElementsAreArray(data));
}

TEST(BufferRefTest, WriteStr) {
  auto data = MakeConstFbData(kData);
  BufferRef buf(data.data(), data.size());
  std::stringstream out;
  buf.WriteStr(out);
  EXPECT_EQ(out.str(), kData);
}

TEST(BufferRefTest, WriteStrOffset) {
  auto data = MakeConstFbData(kData);
  BufferRef buf(data.data(), data.size(), kOffset);
  std::stringstream out;
  buf.WriteStr(out);
  EXPECT_EQ(out.str(), kData.substr(kOffset, buf.Size()));
}

TEST(BufferRefTest, TupleGet) {
  auto input = MakeConstFbData(kData);
  BufferRef buf(input);
  auto [data, size, offset] = buf.Get();
  ASSERT_EQ(offset, 0);
  EXPECT_EQ(input, buf.Span());
}

//
// MutableBufferRef (read/write)
//

TEST(MutableBufferRefTest, Dump) {
  MutableBufferRef<char> buf;
  std::stringstream out;
  buf.Dump(out);
  EXPECT_THAT(out.str(), StartsWith("MutableBufferRef"));
}

TEST(MutableBufferRefTest, WriteInto) {
  auto v_data = MakeFbDataVec(kOtherData);
  MutableBufferRef buf(v_data.data(), v_data.size());
  ASSERT_TRUE(buf.WriteInto("Some"));
  EXPECT_THAT(buf.Span(), ElementsAreArray(v_data));
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(MutableBufferRefTest, WriteIntoOffsetBuf) {
  auto v_data = MakeFbDataVec(kOtherData);
  static constexpr absl::string_view kExpData = "RAWBuffer";
  MutableBufferRef buf(v_data.data(), v_data.size(), kOffset);
  ASSERT_TRUE(buf.WriteInto("RAW"));
  EXPECT_THAT(buf.Span(), ElementsAreArray(MakeConstFbData(kExpData)));
  EXPECT_EQ(buf.StrView(), kExpData);
}

TEST(MutableBufferRefTest, WriteIntoOffsetData) {
  auto v_data = MakeFbDataVec(kOtherData);
  static constexpr absl::string_view kExpData = "SOMERAWBuffer";
  MutableBufferRef buf(v_data.data(), v_data.size());
  ASSERT_TRUE(buf.WriteInto("RAW", kOffset));
  EXPECT_THAT(buf.Span(), ElementsAreArray(MakeConstFbData(kExpData)));
  EXPECT_EQ(buf.StrView(), kExpData);
}

TEST(MutableBufferRefTest, TupleGet) {
  auto input = MakeInternalTestBuffer("FOO");
  MutableBufferRef buf(input);
  auto [data, size, offset] = buf.Get();
  *data = 'b';
  EXPECT_EQ(buf.StrView(), "bOO");
  delete[] input.data();
}

//
// OwningBufferRef (read/write with memory management)
//

TEST(OwningBufferRefTest, Dump) {
  OwningBufferRef buf;
  std::stringstream out;
  buf.Dump(out);
  EXPECT_THAT(out.str(), StartsWith("OwningBufferRef"));
}

TEST(OwningBufferRefTest, MoveCstor) {
  auto raw = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(raw.data(), raw.size());
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> other(std::move(buf));
  EXPECT_EQ(other.StrView(), kData);
}

TEST(OwningBufferRefTest, MoveAssign) {
  auto raw = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(raw.data(), raw.size());
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> other = std::move(buf);
  EXPECT_EQ(other.StrView(), kData);
}

TEST(OwningBufferRefTest, CopyCstor) {
  auto raw = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(raw.data(), raw.size());
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> other(buf);
  other.WriteInto("SOME");
  EXPECT_EQ(buf.StrView(), kData);
  EXPECT_EQ(other.StrView(), "SOMERawBuffer");
}

TEST(OwningBufferRefTest, CopyAssign) {
  auto raw = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(raw.data(), raw.size());
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> other = buf;
  other.WriteInto("SOME");
  EXPECT_EQ(buf.StrView(), kData);
  EXPECT_EQ(other.StrView(), "SOMERawBuffer");
}

TEST(OwningBufferRefTest, InternalMalloc) {
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(kData.size());
  ASSERT_EQ(buf.Size(), kData.size());
  ASSERT_NE(buf.Data(), nullptr);

  buf.WriteInto(kData);
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(OwningBufferRefTest, InternalNew) {
  OwningBufferRef buf(kData.size());
  ASSERT_EQ(buf.Size(), kData.size());
  ASSERT_NE(buf.Data(), nullptr);

  buf.WriteInto(kData);
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(OwningBufferRefTest, TakeOwnershipMalloc) {
  auto malloc_buffer = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(malloc_buffer.data(),
                                                    malloc_buffer.size());
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(OwningBufferRefTest, TakeOwnershipNew) {
  auto new_buffer = MakeInternalTestBuffer(kData);
  OwningBufferRef buf(new_buffer.data(), new_buffer.size());
  EXPECT_EQ(buf.StrView(), kData);
}

TEST(OwningBufferRefTest, TakeOwnershipOffset) {
  auto malloc_buffer = MakeInternalTestBuffer<Mallocator<uint8_t>>(kData);
  OwningBufferRef<uint8_t, Mallocator<uint8_t>> buf(malloc_buffer.data(),
                                                    malloc_buffer.size(),
                                                    /*offset=*/4);
  EXPECT_EQ(buf.StrView(), "RawBuffer");
}

TEST(OwningBufferRefTest, CopyBuffer) {
  auto const_buf = MakeConstFbData(kData);
  OwningBufferRef buf(const_buf.data(), const_buf.size());
  buf.WriteInto("SOME");
  EXPECT_EQ(buf.StrView(), "SOMERawBuffer");
  EXPECT_EQ(FbBufToStr(const_buf), "SomeRawBuffer");
}

TEST(OwningBufferRefTest, ImplicitUpCasts) {
  OwningBufferRef buf(kData.size());
  BufferRef c_buf = buf;

  buf.WriteInto(kData);
  EXPECT_EQ(c_buf.StrView(), buf.StrView());
}

TEST(OwningBufferRefTest, TupleGetWeak) {
  auto input = MakeInternalTestBuffer("FOO");

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  data = input.data();
  size = input.size();
  offset = 0;

  ASSERT_EQ(buf.Size(), input.size());
  ASSERT_EQ(buf.Size(), input.size());

  buf.WriteInto("BAR");

  EXPECT_EQ(buf.StrView(), "BAR");
  EXPECT_EQ(buf.Span(), input);
}

TEST(OwningBufferRefTest, TupleRelease) {
  OwningBufferRef<char> buf("BAZ");

  auto [data, size, offset] = buf.Release();

  EXPECT_EQ(buf.Size(), 0);
  EXPECT_EQ(absl::string_view(data, size), "BAZ");

  delete[] data;
}

TEST(OwningBufferRefTest, Assign) {
  auto const_buf = MakeConstFbData(kData);
  OwningBufferRef buf;
  buf.Assign(const_buf.data(), const_buf.size());
  buf.WriteInto("SOME");
  EXPECT_EQ(buf.StrView(), "SOMERawBuffer");
  EXPECT_EQ(FbBufToStr(const_buf), "SomeRawBuffer");
}

}  // namespace
