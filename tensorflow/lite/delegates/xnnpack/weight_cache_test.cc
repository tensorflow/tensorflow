/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/weight_cache.h"

#include <fcntl.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "experimental.h"  // from @XNNPACK
#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_test_helpers.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite::xnnpack {

std::ostream& operator<<(std::ostream& os, const PackIdentifier& p) {
  return os << "PackIdentifier{pack_algo: " << p.pack_algorithm_id
            << ", weights_id: " << p.weights_id << ", bias_id: " << p.bias_id
            << "}";
}

namespace {

using testing::ElementsAreArray;
using testing::Eq;
using testing::Ge;
using testing::IsNull;
using testing::Ne;
using testing::Not;
#if defined(XNNPACK_TEST_HAS_VMA_NAME)
using testing::EndsWith;
using testing::HasSubstr;
#endif

static xnn_fingerprint kDefaultFingerprint{/*id=*/0xf00d, /*value=*/0xb33f};

struct WeightCacheBuilderTest : testing::Test {
  void SetUp() override { xnn_set_fingerprint(kDefaultFingerprint); }
};

TEST_F(WeightCacheBuilderTest, ReserveAppendWriteWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const PackIdentifier dummy_id{1, 2, 3};

  WeightCacheBuilder builder;
  const std::string cache_path = testing::TempDir() + "/cache";
  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));
  ASSERT_TRUE(builder.StartBuildStep());

  const size_t payload_size = size(payload);
  void* buffer = builder.Reserve(payload_size);
  std::memcpy(buffer, payload.c_str(), payload_size);
  auto loc =
      builder.Append(dummy_id, buffer, payload_size, kDefaultFingerprint.id);

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_GE(builder.capacity(), payload_size);

  ASSERT_TRUE(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
  ASSERT_EQ(header.magic_value, XNNPackCacheHeader::kMagicValue);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const cache::schema::BufferList* const packed_weights =
      cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);

  ASSERT_NE(packed_weights, nullptr);
  ASSERT_NE(packed_weights->buffers(), nullptr);
  ASSERT_EQ(packed_weights->buffers()->size(), 1);
  ASSERT_NE(packed_weights->buffers()->Get(0), nullptr);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->size(), size(payload));
  ASSERT_EQ(packed_weights->buffers()->Get(0)->packing_algorithm_id(),
            dummy_id.pack_algorithm_id);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->weights_id(),
            dummy_id.weights_id);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->bias_id(), dummy_id.bias_id);

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(cache::schema::VerifyBufferListBuffer(verifier));

  ASSERT_LE(packed_weights->base_offset() +
                packed_weights->buffers()->Get(0)->offset(),
            size(handle));
  ASSERT_LE(packed_weights->base_offset() +
                packed_weights->buffers()->Get(0)->offset() +
                packed_weights->buffers()->Get(0)->size(),
            size(handle));
  std::tuple<const char*, size_t> cache_data(
      reinterpret_cast<const char*>(
          handle.data() + packed_weights->base_offset() +
          packed_weights->buffers()->Get(0)->offset()),
      packed_weights->buffers()->Get(0)->size());
  EXPECT_THAT(cache_data, ElementsAreArray(payload));
}

TEST_F(WeightCacheBuilderTest, AppendWithoutReserveWriteWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const PackIdentifier dummy_id{1, 2, 3};

  const std::string cache_path = testing::TempDir() + "/cache";
  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));
  ASSERT_TRUE(builder.StartBuildStep());

  const size_t payload_size = size(payload);
  auto loc = builder.Append(dummy_id, payload.c_str(), payload_size,
                            kDefaultFingerprint.id);

  EXPECT_EQ(loc.size, payload_size);

  ASSERT_TRUE(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
  ASSERT_EQ(header.magic_value, XNNPackCacheHeader::kMagicValue);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const cache::schema::BufferList* const packed_weights =
      cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);
  ASSERT_NE(packed_weights, nullptr);
  ASSERT_NE(packed_weights->buffers(), nullptr);
  ASSERT_EQ(packed_weights->buffers()->size(), 1);
  ASSERT_NE(packed_weights->buffers()->Get(0), nullptr);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->size(), size(payload));
  ASSERT_EQ(packed_weights->buffers()->Get(0)->packing_algorithm_id(),
            dummy_id.pack_algorithm_id);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->weights_id(),
            dummy_id.weights_id);
  ASSERT_EQ(packed_weights->buffers()->Get(0)->bias_id(), dummy_id.bias_id);

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(cache::schema::VerifyBufferListBuffer(verifier));

  ASSERT_LE(packed_weights->base_offset() +
                packed_weights->buffers()->Get(0)->offset(),
            size(handle));
  ASSERT_LE(packed_weights->base_offset() +
                packed_weights->buffers()->Get(0)->offset() +
                packed_weights->buffers()->Get(0)->size(),
            size(handle));
  std::tuple<const char*, size_t> cache_data(
      reinterpret_cast<const char*>(
          handle.data() + packed_weights->base_offset() +
          packed_weights->buffers()->Get(0)->offset()),
      packed_weights->buffers()->Get(0)->size());
  EXPECT_THAT(cache_data, ElementsAreArray(payload));
}

TEST_F(WeightCacheBuilderTest, CorruptBufferListFailsGracefully) {
  const std::string cache_path = testing::TempDir() + "/cache";
  const std::string payload = "This is some data in the file.";
  const PackIdentifier dummy_id{1, 2, 3};

  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));
  ASSERT_TRUE(builder.StartBuildStep());

  const size_t payload_size = size(payload);
  auto loc = builder.Append(dummy_id, payload.c_str(), payload_size,
                            kDefaultFingerprint.id);
  EXPECT_EQ(loc.size, payload_size);
  ASSERT_TRUE(builder.StopBuildStep());

  // corrupt the buffer list data.
  {
    FileDescriptor file_descriptor =
        FileDescriptor::Open(cache_path.c_str(), O_RDWR, 0644);
    ASSERT_TRUE(file_descriptor.IsValid());
    XNNPackCacheHeader header;
    file_descriptor.SetPos(0);
    ASSERT_TRUE(file_descriptor.Read(&header, sizeof(header)));
    file_descriptor.SetPos(header.buffer_list_offset + 1);
    std::string data(8, 'a');
    ASSERT_TRUE(file_descriptor.Write(data.data(), data.size()));
  }

  EXPECT_FALSE(builder.StartBuildStep());
  EXPECT_FALSE(builder.IsBuilding());
}

TEST_F(WeightCacheBuilderTest, EmptyBuildStepEndsTheStep) {
  const std::string cache_path = testing::TempDir() + "/cache";
  const std::string payload1 = "First step payload.";
  const std::string payload2 = "Third step payload.";
  const PackIdentifier id1{1, 2, 3};
  const PackIdentifier id2{4, 5, 6};
  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));

  // Write the first buffer in step 1.
  ASSERT_TRUE(builder.StartBuildStep());
  ASSERT_FALSE(
      builder
          .Append(id1, payload1.data(), payload1.size(), kDefaultFingerprint.id)
          .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep());

  // Reserve without Append in step 2: the step is empty and exits early,
  // releasing the staging buffer and resetting `is_build_step_`.
  ASSERT_TRUE(builder.StartBuildStep());
  EXPECT_NE(builder.Reserve(1024), nullptr);
  ASSERT_TRUE(builder.StopBuildStep());
  EXPECT_FALSE(builder.IsBuilding());
  EXPECT_EQ(builder.capacity(), 0);
  EXPECT_EQ(builder.data(), nullptr);
  EXPECT_EQ(builder.LastBuildStepSize(), 0);

  // Starting step 3 after an empty step reuses the in-memory buffer list and
  // skips re-reading the on-disk flatbuffer, so corrupting the old buffer list
  // on disk before step 3 does not cause `StartBuildStep` to fail.
  ASSERT_NE(file_descriptor.SetPos(builder.LastBuildStepStart() + 1), -1);
  const std::string garbage(8, 'a');
  ASSERT_TRUE(file_descriptor.Write(garbage.data(), garbage.size()));

  ASSERT_TRUE(builder.StartBuildStep());
  ASSERT_FALSE(
      builder
          .Append(id2, payload2.data(), payload2.size(), kDefaultFingerprint.id)
          .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep());

  // Verify that both steps' buffers are present in the finalized cache.
  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));
  const auto& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());
  const cache::schema::BufferList* const packed_weights =
      cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);
  ASSERT_NE(packed_weights, nullptr);
  ASSERT_NE(packed_weights->buffers(), nullptr);
  EXPECT_EQ(packed_weights->buffers()->size(), 2);
}

TEST_F(WeightCacheBuilderTest, RetainAndReleaseStagingBufferAcrossSteps) {
  const std::string cache_path = testing::TempDir() + "/retain_staging_cache";
  const std::string payload1 = "First step payload.";
  const std::string payload2 = "Second step payload.";
  const PackIdentifier id1{1, 2, 3};
  const PackIdentifier id2{4, 5, 6};
  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));

  // Step 1 retains the staging buffer on stop.
  ASSERT_TRUE(builder.StartBuildStep());
  void* buf1 = builder.Reserve(4096);
  ASSERT_NE(buf1, nullptr);
  std::memcpy(buf1, payload1.data(), payload1.size());
  ASSERT_FALSE(
      builder.Append(id1, buf1, payload1.size(), kDefaultFingerprint.id)
          .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep(/*release_staging_buffer=*/false));
  EXPECT_FALSE(builder.IsBuilding());
  uint8_t* const retained_data = builder.data();
  const size_t retained_capacity = builder.capacity();
  ASSERT_NE(retained_data, nullptr);
  EXPECT_GE(retained_capacity, 4096);

  // Step 2 reuses the retained staging buffer without reallocating, and
  // ReleaseStagingBuffer is a no-op while a build step is active.
  ASSERT_TRUE(builder.StartBuildStep());
  void* buf2 = builder.Reserve(1024);
  ASSERT_NE(buf2, nullptr);
  EXPECT_EQ(builder.data(), retained_data);
  EXPECT_EQ(builder.capacity(), retained_capacity);
  builder.ReleaseStagingBuffer();
  EXPECT_EQ(builder.data(), retained_data);
  EXPECT_EQ(builder.capacity(), retained_capacity);
  std::memcpy(buf2, payload2.data(), payload2.size());
  ASSERT_FALSE(
      builder.Append(id2, buf2, payload2.size(), kDefaultFingerprint.id)
          .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep(/*release_staging_buffer=*/false));
  EXPECT_FALSE(builder.IsBuilding());
  EXPECT_EQ(builder.data(), retained_data);
  EXPECT_EQ(builder.capacity(), retained_capacity);

  // An empty step with retention also preserves the staging buffer.
  ASSERT_TRUE(builder.StartBuildStep());
  ASSERT_TRUE(builder.StopBuildStep(/*release_staging_buffer=*/false));
  EXPECT_FALSE(builder.IsBuilding());
  EXPECT_EQ(builder.data(), retained_data);
  EXPECT_EQ(builder.capacity(), retained_capacity);

  // Explicit release frees the staging buffer once no step is active.
  builder.ReleaseStagingBuffer();
  EXPECT_EQ(builder.capacity(), 0);
  EXPECT_EQ(builder.data(), nullptr);

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));
  const auto& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());
  const cache::schema::BufferList* const packed_weights =
      cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);
  ASSERT_NE(packed_weights, nullptr);
  ASSERT_NE(packed_weights->buffers(), nullptr);
  EXPECT_EQ(packed_weights->buffers()->size(), 2);
}

TEST_F(WeightCacheBuilderTest, FailedAppendDoesNotCorruptSchema) {
  const std::string cache_path = testing::TempDir() + "/cache";
  const std::string bad_payload = "Unwritten payload.";
  const std::string good_payload = "Valid packed payload.";
  const PackIdentifier bad_id{9, 9, 9};
  const PackIdentifier good_id{1, 2, 3};

  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));

  // A failed Append (fingerprint_id == 0) on step 0 followed by a restarted
  // step and another failed Append must not leave unwritten BufferT entries.
  ASSERT_TRUE(builder.StartBuildStep());
  EXPECT_TRUE(builder
                  .Append(bad_id, bad_payload.data(), bad_payload.size(),
                          /*fingerprint_id=*/0)
                  .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep());

  ASSERT_TRUE(builder.StartBuildStep());
  EXPECT_TRUE(builder
                  .Append(bad_id, bad_payload.data(), bad_payload.size(),
                          /*fingerprint_id=*/0)
                  .IsInvalid());
  const BufferLocation loc =
      builder.Append(good_id, good_payload.data(), good_payload.size(),
                     kDefaultFingerprint.id);
  ASSERT_FALSE(loc.IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep());

  auto verify_single_good_buffer = [&]() {
    MMapHandle handle;
    ASSERT_TRUE(handle.Map(cache_path.c_str()));
    const auto& header =
        *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());
    ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
    ASSERT_EQ(header.magic_value, XNNPackCacheHeader::kMagicValue);
    const cache::schema::BufferList* const packed_weights =
        cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);
    ASSERT_NE(packed_weights, nullptr);
    ASSERT_NE(packed_weights->buffers(), nullptr);
    ASSERT_EQ(packed_weights->buffers()->size(), 1);
    const auto* entry = packed_weights->buffers()->Get(0);
    EXPECT_EQ(entry->weights_id(), good_id.weights_id);
    EXPECT_EQ(entry->size(), good_payload.size());
    std::tuple<const char*, size_t> cache_data(
        reinterpret_cast<const char*>(
            handle.data() + packed_weights->base_offset() + entry->offset()),
        entry->size());
    EXPECT_THAT(cache_data, ElementsAreArray(good_payload));
  };
  verify_single_good_buffer();

  // Restarting when header.buffer_list_size == 0 also clears stale in-memory
  // schema entries while preserving base_offset.
  XNNPackCacheHeader empty_header{XNNPackCacheHeader::kInvalidHeader};
  empty_header.buffer_list_offset = sizeof(empty_header);
  ASSERT_EQ(file_descriptor.SetPos(0), 0);
  ASSERT_TRUE(file_descriptor.Write(&empty_header, sizeof(empty_header)));

  ASSERT_TRUE(builder.StartBuildStep());
  ASSERT_FALSE(builder
                   .Append(good_id, good_payload.data(), good_payload.size(),
                           kDefaultFingerprint.id)
                   .IsInvalid());
  ASSERT_TRUE(builder.StopBuildStep());
  verify_single_good_buffer();
}

TEST_F(WeightCacheBuilderTest, InvalidFileDescriptorFails) {
  WeightCacheBuilder builder;
  EXPECT_FALSE(builder.Start("", FileDescriptor()));
  EXPECT_FALSE(builder.Start("/seldf/sedsft", FileDescriptor()));
}

TEST_F(WeightCacheBuilderTest, MultipleStepBuild) {
  using std::size;

  const std::string payload1 = "This is some data in the file.";
  const PackIdentifier dummy_id1{1, 2, 3};
  const std::string payload2 = "Other data in the file.";
  const PackIdentifier dummy_id2{2, 3, 4};
  const std::string payload3 =
      GenerateRandomString(/*10 MiB*/ 10 * 1024 * 1024);
  const PackIdentifier dummy_id3{3, 4, 5};

  TempFileDesc tmp_file{TempFileDesc::kAutoClose};

  FileDescriptor file_descriptor = FileDescriptor::Open(
      tmp_file.GetCPath(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(tmp_file.GetCPath(), file_descriptor));
  ASSERT_TRUE(builder.StartBuildStep());

  {
    const size_t payload_size = size(payload1);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload1.c_str(), payload_size);
    const auto loc =
        builder.Append(dummy_id1, buffer, payload_size, kDefaultFingerprint.id);
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }
  {
    const size_t payload_size = size(payload3);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload3.c_str(), payload_size);
    const auto loc =
        builder.Append(dummy_id3, buffer, payload_size, kDefaultFingerprint.id);
    (void)loc;
  }

  ASSERT_TRUE(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

  ASSERT_TRUE(builder.StartBuildStep());
  {
    const size_t payload_size = size(payload2);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload2.c_str(), payload_size);
    const auto loc =
        builder.Append(dummy_id2, buffer, payload_size, kDefaultFingerprint.id);
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }

  ASSERT_TRUE(builder.StopBuildStep());

  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
  ASSERT_EQ(header.magic_value, XNNPackCacheHeader::kMagicValue);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());
  EXPECT_FALSE(header.stale);

  const cache::schema::BufferList* const packed_weights =
      cache::schema::GetBufferList(handle.data() + header.buffer_list_offset);

  ASSERT_NE(packed_weights, nullptr);
  ASSERT_NE(packed_weights->buffers(), nullptr);
  ASSERT_EQ(packed_weights->buffers()->size(), 3);
  // Payload 1.
  const auto* buffer1 = packed_weights->buffers()->Get(0);
  ASSERT_NE(buffer1, nullptr);
  ASSERT_EQ(buffer1->size(), size(payload1));
  ASSERT_EQ(buffer1->packing_algorithm_id(), dummy_id1.pack_algorithm_id);
  ASSERT_EQ(buffer1->weights_id(), dummy_id1.weights_id);
  ASSERT_EQ(buffer1->bias_id(), dummy_id1.bias_id);

  // Payload 3.
  const auto* buffer3 = packed_weights->buffers()->Get(1);
  ASSERT_NE(buffer3, nullptr);
  ASSERT_EQ(buffer3->size(), size(payload3));
  ASSERT_EQ(buffer3->packing_algorithm_id(), dummy_id3.pack_algorithm_id);
  ASSERT_EQ(buffer3->weights_id(), dummy_id3.weights_id);
  ASSERT_EQ(buffer3->bias_id(), dummy_id3.bias_id);

  // Payload 2.
  const auto* buffer2 = packed_weights->buffers()->Get(2);
  ASSERT_NE(buffer2, nullptr);
  ASSERT_EQ(buffer2->size(), size(payload2));
  ASSERT_EQ(buffer2->packing_algorithm_id(), dummy_id2.pack_algorithm_id);
  ASSERT_EQ(buffer2->weights_id(), dummy_id2.weights_id);
  ASSERT_EQ(buffer2->bias_id(), dummy_id2.bias_id);

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(cache::schema::VerifyBufferListBuffer(verifier));

  // Payload 1.
  ASSERT_LE(packed_weights->base_offset() + buffer1->offset(), size(handle));
  ASSERT_LE(packed_weights->base_offset() + buffer1->offset() + buffer1->size(),
            size(handle));

  // Payload 2.
  ASSERT_LE(packed_weights->base_offset() + buffer2->offset(), size(handle));
  ASSERT_LE(packed_weights->base_offset() + buffer2->offset() + buffer2->size(),
            size(handle));

  // Payload 3.
  ASSERT_LE(packed_weights->base_offset() + buffer3->offset(), size(handle));
  ASSERT_LE(packed_weights->base_offset() + buffer3->offset() + buffer3->size(),
            size(handle));

  auto GetBufferData = [&handle, &packed_weights](const auto* buffer) {
    return std::tuple<const char*, size_t>(
        reinterpret_cast<const char*>(
            handle.data() + packed_weights->base_offset() + buffer->offset()),
        buffer->size());
  };

  EXPECT_THAT(GetBufferData(buffer1), ElementsAreArray(payload1));
  EXPECT_THAT(GetBufferData(buffer2), ElementsAreArray(payload2));
  EXPECT_THAT(GetBufferData(buffer3), ElementsAreArray(payload3));
}

TEST(MMapArenaTest, ReservesAlignedPointersAndZeroSizedBuffers) {
  MMapArena arena;
  const size_t kSizes[] = {0, 1, 7, 64, 129, 0};
  std::vector<void*> ptrs;
  for (const size_t size : kSizes) {
    void* const ptr = arena.Reserve(size);
    ASSERT_NE(ptr, nullptr) << size;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 128, 0) << size;
    ASSERT_TRUE(arena.Commit(ptr, size)) << size;
    ptrs.push_back(ptr);
  }
  for (size_t i = 0; i < ptrs.size(); ++i) {
    for (size_t j = i + 1; j < ptrs.size(); ++j) {
      EXPECT_NE(ptrs[i], ptrs[j]) << i << ", " << j;
    }
  }
}

TEST(MMapArenaTest, RewindsUncommittedReservations) {
  MMapArena arena;
  // An oversized (> 1 MiB) uncommitted reservation on an empty arena is
  // discarded when followed directly by a small Reserve.
  ASSERT_NE(arena.Reserve(2 * 1024 * 1024), nullptr);
  EXPECT_TRUE(arena.HasPendingReservation());

  void* const uncommitted = arena.Reserve(64);
  ASSERT_NE(uncommitted, nullptr);
  EXPECT_TRUE(arena.HasPendingReservation());

  // Calling Reserve again without Commit reuses the uncommitted slot.
  void* const committed = arena.Reserve(64);
  EXPECT_EQ(committed, uncommitted);
  std::memset(committed, 0xAB, 64);
  ASSERT_TRUE(arena.Commit(committed, 64));
  EXPECT_FALSE(arena.HasPendingReservation());

  // An overflowing Reserve rewinds any pending reservation, drops an oversized
  // uncommitted tail chunk and returns nullptr without corrupting committed
  // allocations.
  ASSERT_NE(arena.Reserve(2 * 1024 * 1024), nullptr);
  EXPECT_TRUE(arena.HasPendingReservation());
  EXPECT_EQ(arena.Reserve(std::numeric_limits<size_t>::max()), nullptr);
  EXPECT_FALSE(arena.HasPendingReservation());

  void* const after = arena.Reserve(64);
  ASSERT_NE(after, nullptr);
  std::memset(after, 0xCD, 64);
  ASSERT_TRUE(arena.Commit(after, 64));

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<const uint8_t*>(committed)[i], 0xAB) << i;
    EXPECT_EQ(static_cast<const uint8_t*>(after)[i], 0xCD) << i;
  }

  // On a fresh arena (`used == 0`), an overflowing Reserve evicts and unmaps
  // the oversized uncommitted tail chunk.
  {
    MMapArena fresh_arena;
    void* const oversized = fresh_arena.Reserve(2 * 1024 * 1024);
    ASSERT_NE(oversized, nullptr);
    EXPECT_TRUE(fresh_arena.HasPendingReservation());
    EXPECT_EQ(fresh_arena.Reserve(std::numeric_limits<size_t>::max()), nullptr);
    EXPECT_FALSE(fresh_arena.HasPendingReservation());
#if defined(XNNPACK_TEST_HAS_VMA_NAME)
    EXPECT_TRUE(
        GetMapsLine(static_cast<const uint8_t*>(oversized) + 1024 * 1024)
            .empty());
#endif
  }
}

TEST(MMapArenaTest, CommitValidatesReservationAndReclaimsUnusedTail) {
  MMapArena arena;
  uint8_t dummy = 0;
  EXPECT_FALSE(arena.Commit(nullptr, 0));
  EXPECT_FALSE(arena.Commit(&dummy, 1));

  auto* const first = static_cast<uint8_t*>(arena.Reserve(1024));
  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(arena.HasPendingReservation());

  // Mismatched pointers or sizes exceeding the reservation fail without
  // consuming the active reservation.
  EXPECT_FALSE(arena.Commit(first + 1, 64));
  EXPECT_FALSE(arena.Commit(&dummy, 64));
  EXPECT_FALSE(arena.Commit(first, 1025));
  EXPECT_TRUE(arena.HasPendingReservation());

  // Committing a smaller size succeeds, clears the pending reservation and
  // reclaims unused tail bytes for the next allocation.
  ASSERT_TRUE(arena.Commit(first, 64));
  EXPECT_FALSE(arena.HasPendingReservation());
  EXPECT_FALSE(arena.Commit(first, 64));

  auto* const second = static_cast<uint8_t*>(arena.Reserve(64));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second, first + 128);
  EXPECT_TRUE(arena.Commit(second, 64));
}

TEST(MMapArenaTest, ReplacesEmptyChunkWhenLargerUncommittedReserveNeedsRemap) {
  const size_t page_size = RoundUpToPageSize(1);
  MMapHandle blocker_before;
  ASSERT_TRUE(blocker_before.Map(MMapHandle::kAnonymous, page_size));
  MMapArena arena;
  void* const initial = arena.Reserve(64);
  ASSERT_NE(initial, nullptr);
  MMapHandle blocker_after;
  ASSERT_TRUE(blocker_after.Map(MMapHandle::kAnonymous, page_size));

  // With adjacent mappings pinned on both sides of the uncommitted 1-page
  // chunk, a multi-page Reserve cannot resize in place and replaces the unused
  // chunk instead.
  const size_t large_size = 16 * page_size;
  MMapHandle evicted;
  auto* const ptr = static_cast<uint8_t*>(arena.Reserve(large_size, &evicted));
  ASSERT_NE(ptr, nullptr);
  EXPECT_NE(ptr, initial);
  ASSERT_TRUE(evicted.IsMapped());
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  EXPECT_GE(static_cast<const uint8_t*>(initial), evicted.data());
  EXPECT_LT(static_cast<const uint8_t*>(initial), evicted.data() + 128);
#else
  EXPECT_EQ(evicted.data(), initial);
#endif
  std::memset(ptr, 0x7E, large_size);
  ASSERT_TRUE(arena.Commit(ptr, large_size));
  EXPECT_EQ(ptr[0], 0x7E);
  EXPECT_EQ(ptr[large_size - 1], 0x7E);
}

#if (defined(__linux__) || defined(__ANDROID__)) && \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
TEST(MMapArenaTest, ExpandsActiveChunkInPlaceWhenAdjacentSpaceIsFree) {
  const size_t page_size = RoundUpToPageSize(1);
  // Plug any small holes left by prior tests until 1-page and multi-page
  // mappings land contiguously at the mmap frontier.
  std::vector<MMapHandle> hole_plugs;
  for (int i = 0; i < 128; ++i) {
    MMapHandle small_probe;
    MMapHandle large_probe;
    ASSERT_TRUE(small_probe.Map(MMapHandle::kAnonymous, page_size)) << i;
    ASSERT_TRUE(large_probe.Map(MMapHandle::kAnonymous, 6 * page_size)) << i;
    if (large_probe.data() + large_probe.size() == small_probe.data() ||
        small_probe.data() + small_probe.size() == large_probe.data()) {
      break;
    }
    hole_plugs.push_back(std::move(small_probe));
  }

  MMapHandle blocker_before;
  ASSERT_TRUE(blocker_before.Map(MMapHandle::kAnonymous, 3 * page_size));

  MMapArena arena;
  auto* const first = static_cast<uint8_t*>(arena.Reserve(64));
  ASSERT_NE(first, nullptr);
  std::memset(first, 0x5A, 64);
  ASSERT_TRUE(arena.Commit(first, 64));

  MMapHandle blocker_after;
  ASSERT_TRUE(blocker_after.Map(MMapHandle::kAnonymous, 3 * page_size));

  // Unmapping both neighbors guarantees free virtual address space immediately
  // above the active chunk regardless of top-down or bottom-up mmap order.
  blocker_before.UnMap();
  blocker_after.UnMap();

  const size_t second_size = 2 * page_size;
  auto* const second = static_cast<uint8_t*>(arena.Reserve(second_size));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second, first + 128);
  std::memset(second, 0xA5, second_size);
  ASSERT_TRUE(arena.Commit(second, second_size));

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(first[i], 0x5A) << i;
  }
  EXPECT_EQ(second[0], 0xA5);
  EXPECT_EQ(second[second_size - 1], 0xA5);
}
#endif

#if defined(XNNPACK_TEST_CAN_PROBE_UNMAPPABLE_SIZE)
TEST(MMapArenaTest, UnmappableReserveFailsCleanly) {
  constexpr size_t kUnmappableSize = std::numeric_limits<size_t>::max() / 4;
  MMapArena arena;
  EXPECT_EQ(arena.Reserve(kUnmappableSize), nullptr);
  EXPECT_FALSE(arena.HasPendingReservation());

  // A failed large Reserve after an uncommitted > 1 MiB reservation drops the
  // oversized chunk so subsequent small reservations do not keep it mapped.
  void* const oversized = arena.Reserve(2 * 1024 * 1024);
  ASSERT_NE(oversized, nullptr);
  EXPECT_TRUE(arena.HasPendingReservation());
  EXPECT_EQ(arena.Reserve(kUnmappableSize), nullptr);
  EXPECT_FALSE(arena.HasPendingReservation());
#if defined(XNNPACK_TEST_HAS_VMA_NAME)
  EXPECT_TRUE(GetMapsLine(static_cast<const uint8_t*>(oversized) + 1024 * 1024)
                  .empty());
#endif

  void* const ptr = arena.Reserve(64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(arena.Commit(ptr, 64));
}
#endif

TEST(MMapArenaTest, GrowsAcrossMultipleMegabytesWithStableAddresses) {
  MMapArena arena;
  constexpr size_t kBufferCount = 16;
  constexpr size_t kBufferSize = 256 * 1024;
  std::vector<uint8_t*> ptrs;
  for (size_t i = 0; i < kBufferCount; ++i) {
    auto* const ptr = static_cast<uint8_t*>(arena.Reserve(kBufferSize));
    ASSERT_NE(ptr, nullptr) << i;
    std::memset(ptr, static_cast<int>(i + 1), kBufferSize);
    ASSERT_TRUE(arena.Commit(ptr, kBufferSize)) << i;
    ptrs.push_back(ptr);
  }
  for (size_t i = 0; i < kBufferCount; ++i) {
    EXPECT_EQ(ptrs[i][0], static_cast<uint8_t>(i + 1)) << i;
    EXPECT_EQ(ptrs[i][kBufferSize - 1], static_cast<uint8_t>(i + 1)) << i;
  }
}

TEST(MMapArenaTest, MoveResetsSourceArena) {
  MMapArena arena1;
  void* const ptr = arena1.Reserve(128);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(arena1.HasPendingReservation());

  MMapArena arena2(std::move(arena1));
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(arena1.HasPendingReservation());
  // NOLINTEND(bugprone-use-after-move)
  EXPECT_TRUE(arena2.HasPendingReservation());

  MMapArena arena3;
  arena3 = std::move(arena2);
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(arena2.HasPendingReservation());
  // NOLINTEND(bugprone-use-after-move)
  EXPECT_TRUE(arena3.HasPendingReservation());
  EXPECT_TRUE(arena3.Commit(ptr, 128));
}

TEST(MMapArenaTest, ExactPageMultipleReserveDoesNotOvermapPage) {
  const size_t page_size = RoundUpToPageSize(1);
  ASSERT_GT(page_size, 0);

  {
    MMapArena arena;
    auto* const ptr = static_cast<uint8_t*>(arena.Reserve(page_size));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 128, 0);
    std::memset(ptr, 0x5A, page_size);
    ASSERT_TRUE(arena.Commit(ptr, page_size));
    EXPECT_EQ(ptr[0], 0x5A);
    EXPECT_EQ(ptr[page_size - 1], 0x5A);

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
    if (AnonVmaNamingSupported()) {
      const std::string maps_line = GetMapsLine(ptr);
      uintptr_t start = 0;
      uintptr_t end = 0;
      ASSERT_EQ(std::sscanf(maps_line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &start,
                            &end),
                2)
          << maps_line;
      EXPECT_EQ(end - start, page_size) << maps_line;
    }
#endif
  }

  // Rewinding an oversized (> 1 MiB) uncommitted chunk to an exact 1 MiB
  // reservation drops the oversized chunk and maps 1 MiB without extra padding.
  {
    constexpr size_t kOneMiB = 1024 * 1024;
    MMapArena arena;
    ASSERT_NE(arena.Reserve(2 * kOneMiB), nullptr);
    auto* const ptr = static_cast<uint8_t*>(arena.Reserve(kOneMiB));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 128, 0);
    std::memset(ptr, 0xA5, kOneMiB);
    ASSERT_TRUE(arena.Commit(ptr, kOneMiB));
    EXPECT_EQ(ptr[0], 0xA5);
    EXPECT_EQ(ptr[kOneMiB - 1], 0xA5);

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
    if (AnonVmaNamingSupported() && page_size <= kOneMiB) {
      const std::string maps_line = GetMapsLine(ptr);
      uintptr_t start = 0;
      uintptr_t end = 0;
      ASSERT_EQ(std::sscanf(maps_line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &start,
                            &end),
                2)
          << maps_line;
      EXPECT_EQ(end - start, kOneMiB) << maps_line;
    }
#endif
  }

  // Rewinding an 8 MiB uncommitted chunk to a 2 MiB reservation (> 1 MiB) also
  // drops the oversized chunk rather than keeping 6 MiB of unused capacity.
  {
    constexpr size_t kTwoMiB = 2 * 1024 * 1024;
    constexpr size_t kEightMiB = 8 * 1024 * 1024;
    MMapArena arena;
    ASSERT_NE(arena.Reserve(kEightMiB), nullptr);
    auto* const two_mib = static_cast<uint8_t*>(arena.Reserve(kTwoMiB));
    ASSERT_NE(two_mib, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(two_mib) % 128, 0);
    std::memset(two_mib, 0x3C, kTwoMiB);
    ASSERT_TRUE(arena.Commit(two_mib, kTwoMiB));
    EXPECT_EQ(two_mib[0], 0x3C);
    EXPECT_EQ(two_mib[kTwoMiB - 1], 0x3C);

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
    if (AnonVmaNamingSupported() && page_size <= kTwoMiB) {
      const std::string maps_line = GetMapsLine(two_mib);
      uintptr_t start = 0;
      uintptr_t end = 0;
      ASSERT_EQ(std::sscanf(maps_line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &start,
                            &end),
                2)
          << maps_line;
      EXPECT_EQ(end - start, kTwoMiB) << maps_line;
    }
#endif
  }
}

TEST(CacheMissHandlerTest, ReserveThenAppendWorks) {
  const int kBufferSize = 5;
  const size_t kMinOffset = 12;
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  cache_miss_handler.SetMinOffset(kMinOffset);
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(0));
  double* ptr = reinterpret_cast<double*>(
      cache_miss_handler.Reserve(kBufferSize * sizeof(double)));
  ASSERT_THAT(ptr, Not(IsNull()));
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(1));
  EXPECT_FALSE(cache_miss_handler.ConsistentReserveAppendCalls());
  std::iota(ptr, ptr + kBufferSize, 1);

  const PackIdentifier dummy_id{2, 3, 4};
  BufferLocation loc =
      cache_miss_handler.Append(dummy_id, ptr, kBufferSize * sizeof(double),
                                kDefaultFingerprint.id, offset_to_addr);
  EXPECT_THAT(loc.size, Eq(kBufferSize * sizeof(double)));
  EXPECT_THAT(loc.offset, Ge(kMinOffset));
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(1));
  EXPECT_TRUE(cache_miss_handler.ConsistentReserveAppendCalls());
  EXPECT_THAT(offset_to_addr.find(loc.offset), Not(offset_to_addr.end()));
}

TEST(CacheMissHandlerTest, AppendZeroBytesAndFallbackCommitWork) {
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  const PackIdentifier id1{1, 2, 3};
  const BufferLocation zero_loc = cache_miss_handler.Append(
      id1, nullptr, /*size=*/0, kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(zero_loc.IsInvalid());
  EXPECT_EQ(zero_loc.size, 0);
  EXPECT_EQ(cache_miss_handler.BufferCount(), 1);
  ASSERT_NE(offset_to_addr.find(zero_loc.offset), offset_to_addr.end());
  EXPECT_NE(offset_to_addr[zero_loc.offset], nullptr);

  // Appending more bytes than the active reservation rewinds and reserves a
  // larger slot without self-overlapping memcpy when the address is reused.
  auto* const buf = static_cast<uint8_t*>(cache_miss_handler.Reserve(64));
  ASSERT_NE(buf, nullptr);
  std::memset(buf, 0x33, 64);
  ASSERT_EQ(cache_miss_handler.Reserve(16), buf);
  EXPECT_EQ(cache_miss_handler.BufferCount(), 2);
  const PackIdentifier id2{4, 5, 6};
  const BufferLocation fallback_loc = cache_miss_handler.Append(
      id2, buf, /*size=*/64, kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(fallback_loc.IsInvalid());
  EXPECT_EQ(fallback_loc.size, 64);
  EXPECT_EQ(cache_miss_handler.BufferCount(), 2);
  const auto* const stored =
      static_cast<const uint8_t*>(offset_to_addr[fallback_loc.offset]);
  ASSERT_NE(stored, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(stored[i], 0x33) << i;
  }
}

#if defined(XNNPACK_TEST_CAN_PROBE_UNMAPPABLE_SIZE)
TEST(CacheMissHandlerTest, UnmappableAppendReturnsInvalidLocation) {
  constexpr size_t kUnmappableSize = std::numeric_limits<size_t>::max() / 4;
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  const PackIdentifier dummy_id{2, 3, 4};
  const BufferLocation loc =
      cache_miss_handler.Append(dummy_id, nullptr, kUnmappableSize,
                                kDefaultFingerprint.id, offset_to_addr);
  EXPECT_TRUE(loc.IsInvalid());
  EXPECT_EQ(cache_miss_handler.BufferCount(), 0);
  EXPECT_FALSE(cache_miss_handler.HasCacheMisses());
  EXPECT_TRUE(offset_to_addr.empty());
}
#endif

TEST(CacheMissHandlerTest, AppendHandlesSubSlicesAndRejectsChunkOverrun) {
  // Non-zero sub-slice of an uncommitted 2 MiB reservation: rewinding pops the
  // 2 MiB chunk, so the evicted mapping must stay alive until `memmove`
  // completes.
  {
    CacheMissHandler handler;
    std::map<size_t, void*> offset_to_addr;
    constexpr size_t kTwoMiB = 2 * 1024 * 1024;
    auto* const buf = static_cast<uint8_t*>(handler.Reserve(kTwoMiB));
    ASSERT_NE(buf, nullptr);
    for (size_t i = 0; i < 64; ++i) {
      buf[128 + i] = static_cast<uint8_t>(0x80 + i);
    }

    const PackIdentifier id{1, 2, 3};
    const BufferLocation loc = handler.Append(
        id, buf + 128, /*size=*/64, kDefaultFingerprint.id, offset_to_addr);
    ASSERT_FALSE(loc.IsInvalid());
    EXPECT_EQ(loc.size, 64);
    EXPECT_EQ(handler.BufferCount(), 1);
    EXPECT_TRUE(handler.HasCacheMisses());
    const auto* const stored =
        static_cast<const uint8_t*>(offset_to_addr[loc.offset]);
    ASSERT_NE(stored, nullptr);
    for (size_t i = 0; i < 64; ++i) {
      EXPECT_EQ(stored[i], static_cast<uint8_t>(0x80 + i)) << i;
    }
  }

  // Overlapping sub-slice of a retained 128-byte reservation: rewinding reuses
  // `buf2` as destination while copying from `buf2 + 32`.
  {
    CacheMissHandler handler;
    std::map<size_t, void*> offset_to_addr;
    auto* const buf2 = static_cast<uint8_t*>(handler.Reserve(128));
    ASSERT_NE(buf2, nullptr);
    for (size_t i = 0; i < 128; ++i) {
      buf2[i] = static_cast<uint8_t>(i + 1);
    }

    const PackIdentifier id{4, 5, 6};
    const BufferLocation loc = handler.Append(
        id, buf2 + 32, /*size=*/64, kDefaultFingerprint.id, offset_to_addr);
    ASSERT_FALSE(loc.IsInvalid());
    EXPECT_EQ(loc.size, 64);
    const auto* const stored =
        static_cast<const uint8_t*>(offset_to_addr[loc.offset]);
    EXPECT_EQ(stored, buf2);
    for (size_t i = 0; i < 64; ++i) {
      EXPECT_EQ(stored[i], static_cast<uint8_t>(33 + i)) << i;
    }
  }

  // Out-of-bounds slice extending past the active 1-page chunk returns an
  // invalid location before reserving or reading past the mapping end.
  {
    const size_t page_size = RoundUpToPageSize(1);
    ASSERT_GT(page_size, 128);
    CacheMissHandler handler;
    std::map<size_t, void*> offset_to_addr;
    auto* const buf3 = static_cast<uint8_t*>(handler.Reserve(64));
    ASSERT_NE(buf3, nullptr);

    const PackIdentifier id{7, 8, 9};
    const BufferLocation loc = handler.Append(
        id, buf3, 2 * page_size, kDefaultFingerprint.id, offset_to_addr);
    EXPECT_TRUE(loc.IsInvalid());
    EXPECT_FALSE(handler.HasCacheMisses());
    EXPECT_TRUE(offset_to_addr.empty());
  }
}

TEST(CacheMissHandlerTest, MovePreservesStateAndResetsSource) {
  constexpr size_t kMinOffset = 100;
  const PackIdentifier id1{1, 2, 3};
  const PackIdentifier id2{4, 5, 6};
  const PackIdentifier id3{7, 8, 9};
  std::map<size_t, void*> offset_to_addr;

  CacheMissHandler source;
  source.SetMinOffset(kMinOffset);
  auto* const first = static_cast<uint8_t*>(source.Reserve(32));
  ASSERT_NE(first, nullptr);
  std::memset(first, 0x11, 32);
  const BufferLocation loc1 = source.Append(
      id1, first, /*size=*/32, kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(loc1.IsInvalid());
  EXPECT_EQ(loc1.offset, kMinOffset);

  // Leave a second reservation uncommitted before move-constructing.
  auto* const second = static_cast<uint8_t*>(source.Reserve(48));
  ASSERT_NE(second, nullptr);
  std::memset(second, 0x22, 48);
  ASSERT_EQ(source.BufferCount(), 2);
  ASSERT_TRUE(source.HasCacheMisses());
  ASSERT_FALSE(source.ConsistentReserveAppendCalls());

  CacheMissHandler move_constructed(std::move(source));
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_EQ(source.BufferCount(), 0);
  EXPECT_FALSE(source.HasCacheMisses());
  EXPECT_TRUE(source.ConsistentReserveAppendCalls());
  const uint8_t reused_byte = 0x77;
  const BufferLocation reset_loc =
      source.Append(id1, &reused_byte, sizeof(reused_byte),
                    kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(reset_loc.IsInvalid());
  EXPECT_EQ(reset_loc.offset, 0);
  // NOLINTEND(bugprone-use-after-move)

  EXPECT_EQ(move_constructed.BufferCount(), 2);
  EXPECT_TRUE(move_constructed.HasCacheMisses());
  EXPECT_FALSE(move_constructed.ConsistentReserveAppendCalls());

  const BufferLocation loc2 = move_constructed.Append(
      id2, second, /*size=*/48, kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(loc2.IsInvalid());
  EXPECT_EQ(loc2.offset, kMinOffset + 1);
  EXPECT_EQ(move_constructed.BufferCount(), 2);
  EXPECT_TRUE(move_constructed.ConsistentReserveAppendCalls());

  // Leave a third reservation uncommitted before move-assigning onto a
  // non-empty destination.
  auto* const third = static_cast<uint8_t*>(move_constructed.Reserve(24));
  ASSERT_NE(third, nullptr);
  std::memset(third, 0x33, 24);
  ASSERT_FALSE(move_constructed.ConsistentReserveAppendCalls());

  CacheMissHandler move_assigned;
  move_assigned.SetMinOffset(500);
  ASSERT_NE(move_assigned.Reserve(16), nullptr);
  move_assigned = std::move(move_constructed);

  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_EQ(move_constructed.BufferCount(), 0);
  EXPECT_FALSE(move_constructed.HasCacheMisses());
  EXPECT_TRUE(move_constructed.ConsistentReserveAppendCalls());
  // NOLINTEND(bugprone-use-after-move)

  EXPECT_EQ(move_assigned.BufferCount(), 3);
  EXPECT_TRUE(move_assigned.HasCacheMisses());
  EXPECT_FALSE(move_assigned.ConsistentReserveAppendCalls());

  const BufferLocation loc3 = move_assigned.Append(
      id3, third, /*size=*/24, kDefaultFingerprint.id, offset_to_addr);
  ASSERT_FALSE(loc3.IsInvalid());
  EXPECT_EQ(loc3.offset, kMinOffset + 2);
  EXPECT_EQ(move_assigned.BufferCount(), 3);
  EXPECT_TRUE(move_assigned.ConsistentReserveAppendCalls());

  EXPECT_EQ(static_cast<const uint8_t*>(offset_to_addr[loc1.offset])[0], 0x11);
  EXPECT_EQ(static_cast<const uint8_t*>(offset_to_addr[loc2.offset])[0], 0x22);
  EXPECT_EQ(static_cast<const uint8_t*>(offset_to_addr[loc3.offset])[0], 0x33);
}

TEST(CacheMissHandlerTest, AppendWithoutReserveWorks) {
  const double kData[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const size_t kMinOffset = 12;
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  cache_miss_handler.SetMinOffset(kMinOffset);
  const PackIdentifier dummy_id{2, 3, 4};
  BufferLocation loc = cache_miss_handler.Append(
      dummy_id, kData, sizeof(kData), kDefaultFingerprint.id, offset_to_addr);
  EXPECT_THAT(loc.size, Eq(sizeof(kData)));
  EXPECT_THAT(loc.offset, Ge(kMinOffset));
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(1));
  EXPECT_THAT(offset_to_addr.find(loc.offset), Not(offset_to_addr.end()));
}

TEST(CacheMissHandlerTest, MultipleAppendsReturnDifferentOffsets) {
  const size_t kMinOffset = 12;
  const double kData[] = {1, 2, 3, 4, 5, 6, 7, 8};
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  cache_miss_handler.SetMinOffset(kMinOffset);
  const PackIdentifier dummy_id1{2, 3, 4};
  BufferLocation loc1 = cache_miss_handler.Append(
      dummy_id1, kData, sizeof(kData), kDefaultFingerprint.id, offset_to_addr);
  const PackIdentifier dummy_id2{1, 3, 4};
  BufferLocation loc2 = cache_miss_handler.Append(
      dummy_id2, kData, sizeof(kData), kDefaultFingerprint.id, offset_to_addr);
  EXPECT_THAT(loc1.offset, Ne(loc2.offset));
  EXPECT_THAT(loc1.offset, Ge(kMinOffset));
  EXPECT_THAT(loc2.offset, Ge(kMinOffset));
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(2));
  const auto buffer1_it = offset_to_addr.find(loc1.offset);
  const auto buffer2_it = offset_to_addr.find(loc2.offset);
  ASSERT_THAT(buffer1_it, Not(offset_to_addr.end()));
  ASSERT_THAT(buffer2_it, Not(offset_to_addr.end()));
  EXPECT_THAT(buffer1_it->second, Not(buffer2_it->second));
}

TEST(CacheMissHandlerTest, UnusedReserveDoesntReuseOffsets) {
  const double kData[] = {1, 2, 3, 4, 5, 6, 7, 8};
  CacheMissHandler cache_miss_handler;
  std::map<size_t, void*> offset_to_addr;
  const PackIdentifier dummy_id{2, 3, 4};
  // The first buffer is reserved but never appended, e.g. because the packed
  // data was already in the cache. The next reservation replaces it.
  ASSERT_THAT(cache_miss_handler.Reserve(sizeof(kData)), Not(IsNull()));
  void* const ptr1 = cache_miss_handler.Reserve(sizeof(kData));
  std::memcpy(ptr1, kData, sizeof(kData));
  const BufferLocation loc1 = cache_miss_handler.Append(
      dummy_id, ptr1, sizeof(kData), kDefaultFingerprint.id, offset_to_addr);
  const BufferLocation loc2 = cache_miss_handler.Append(
      dummy_id, kData, sizeof(kData), kDefaultFingerprint.id, offset_to_addr);
  EXPECT_THAT(cache_miss_handler.BufferCount(), Eq(2));
  EXPECT_THAT(loc1.offset, Ne(loc2.offset));
  EXPECT_THAT(offset_to_addr.size(), Eq(2));
}

struct FakeContext {
  // Adds a new tensor and it's backing buffer to the context.
  //
  // The tensor `data` will not be set until `FinalizeTensors` is called.
  void AddTensor(int buffer_identifier, size_t size) {
    buffers.emplace_back(size, buffer_identifier);
    tensors.push_back({});
    tensors.back().allocation_type = kTfLiteMmapRo;
    tensor_buffer_identifiers[tensors.size() - 1] = buffer_identifier;
  }

  // Updates the tensor data mappings.
  //
  // This needs to be called every time the context `tensors` list is
  // reallocated (mainly because of insertions).
  void FinalizeTensors() {
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i].data.data = buffers[i].data();
      tensors[i].bytes = buffers[i].size();
    }
  }

  // Creates a look up key for the XNNPack weight provider C interface.
  xnn_weights_cache_look_up_key LookUpKey(const uint32_t algorithm_seed,
                                          const int weights_index) const {
    return {.seed = algorithm_seed,
            .kernel = buffers[weights_index].data(),
            .bias = nullptr,
            .fingerprint_id = kDefaultFingerprint.id};
  }

  // Creates a look up key for the XNNPack weight provider C interface.
  xnn_weights_cache_look_up_key LookUpKey(const uint32_t algorithm_seed,
                                          const int weights_index,
                                          const int bias_index) const {
    return {.seed = algorithm_seed,
            .kernel = buffers[weights_index].data(),
            .bias = buffers[bias_index].data(),
            .fingerprint_id = kDefaultFingerprint.id};
  }

  // Helps creating fake packed data.
  void AddTensorToPack(std::vector<uint8_t>& pack_buffer, int index) {
    const std::vector<uint8_t>& buffer = buffers[index];
    pack_buffer.resize(std::max(size(pack_buffer), size(buffer)));
    for (size_t i = 0; i < size(buffer); ++i) {
      pack_buffer[i] ^= buffer[i];
    }
  }

  // Packs the referenced tensors into one buffer.
  //
  // Returns the pack id to retrieve the packed reference data from
  // `packed_buffers`.
  template <class... Ids>
  PackIdentifier PackTensors(xnn_weights_cache_t weight_cache,
                             const uint32_t algorithm_seed,
                             const Ids... tensor_indices) {
    // Create fake packed and save the result for later lookup tests.

    PackIdentifier pack_id{algorithm_seed,
                           tensor_buffer_identifiers[tensor_indices]...};
    PackedBuffer& packed =
        packed_buffers.emplace(pack_id, PackedBuffer{})->second;
    (AddTensorToPack(packed.buffer, tensor_indices), ...);

    // Add the packed buffer to the XNNPack cache. Normally you would pack in
    // place where the reserved space is.
    xnn_weights_cache_look_up_key look_up_key =
        LookUpKey(algorithm_seed, tensor_indices...);
    packed.offset = weight_cache->look_up_or_insert(
        weight_cache->context, &look_up_key, packed.buffer.data(),
        packed.buffer.size());
    return pack_id;
  }

  struct PackedBuffer {
    size_t offset;
    std::vector<uint8_t> buffer;
  };

  std::vector<TfLiteTensor> tensors;
  std::vector<std::vector<uint8_t>> buffers;
  std::unordered_multimap<PackIdentifier, PackedBuffer, PackIdentifier::Hash>
      packed_buffers;
  std::unordered_map<size_t, size_t> tensor_buffer_identifiers;
};

struct TestVariant {
  bool use_explicit_fd;
  const char* explicit_fd_path = nullptr;
  bool use_in_memory_cache = false;

  static std::string Name(const testing::TestParamInfo<TestVariant>& info) {
    if (info.param.use_in_memory_cache) {
      return "WithInMemoryCache";
    }
    if (info.param.use_explicit_fd) {
      if (info.param.explicit_fd_path) {
        return "WithExplicitFileDescriptorAndPath";
      } else {
        return "WithExplicitFileDescriptorAndNoPath";
      }
    }
    return "WithImplicitFileDescriptor";
  }

  friend std::ostream& operator<<(std::ostream& os, const TestVariant& tv) {
    if (tv.use_in_memory_cache) {
      return os << "in-memory";
    }
    if (tv.use_explicit_fd) {
      os << "explicit fd:";
    } else {
      return os << "implicit fd from path";
    }
    if (tv.explicit_fd_path) {
      os << tv.explicit_fd_path;
    } else {
      os << "[no path]";
    }
    return os;
  }
};

auto TestVariants() {
  return testing::Values(
      TestVariant{/*use_explicit_fd=*/false, /*explicit_fd_path=*/nullptr},
      TestVariant{/*use_explicit_fd=*/true, /*explicit_fd_path=*/nullptr},
      TestVariant{/*use_explicit_fd=*/true,
                  /*explicit_fd_path=*/"explicit file descriptor"},
      TestVariant{/*use_explicit_fd=*/false, /*explicit_fd_path=*/nullptr,
                  /*use_in_memory_cache=*/true});
}

struct BuildMMapWeightCacheProviderTest : testing::TestWithParam<TestVariant> {
  enum { kAlgoSeed1, kAlgoSeed2, kAlgoSeed3 };
  enum { kBufferId1, kBufferId2, kBufferId3, kBufferId4 };

  void SetUp() override {
    if (use_in_memory_cache &&
        !TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider()) {
      GTEST_SKIP() << "In-memory weight cache isn't enabled for this build or "
                      "isn't supported by the current system, skipping test.";
    }
    xnn_set_fingerprint(kDefaultFingerprint);
    AddTensors();
    EndSetup();
  }

  void AddTensors() {
    ctx.AddTensor(/*buffer_identifier=*/kBufferId1, /*size=*/12);
    ctx.AddTensor(/*buffer_identifier=*/kBufferId2, /*size=*/43);
    ctx.AddTensor(/*buffer_identifier=*/kBufferId3, /*size=*/64);
    ctx.AddTensor(/*buffer_identifier=*/kBufferId4, /*size=*/8);
  }

  void EndSetup() {
    ctx.FinalizeTensors();
    ASSERT_TRUE(cache_provider.MapTensorIdentifiers(
        ctx.tensors.data(), ctx.tensors.size(), ctx.tensor_buffer_identifiers));
    if (use_explicit_fd) {
      ASSERT_TRUE(
          cache_provider.StartBuild(explicit_fd_path, tmp_file.Duplicate()));
    } else {
      tmp_file.Close();
      if (use_in_memory_cache) {
        ASSERT_TRUE(cache_provider.StartBuild(kInMemoryCachePath));
      } else {
        ASSERT_TRUE(cache_provider.StartBuild(tmp_file.GetCPath()));
      }
    }
  }

  FakeContext ctx;
  MMapWeightCacheProvider cache_provider;
  TempFileDesc tmp_file;
  const bool use_explicit_fd = GetParam().use_explicit_fd;
  const char* explicit_fd_path = GetParam().explicit_fd_path;
  const bool use_in_memory_cache = GetParam().use_in_memory_cache;
};

INSTANTIATE_TEST_SUITE_P(Test, BuildMMapWeightCacheProviderTest, TestVariants(),
                         TestVariant::Name);

TEST_P(BuildMMapWeightCacheProviderTest, LookUpFailsIfKeyDoesntMatch) {
  xnn_weights_cache_look_up_key look_up_key{};
  EXPECT_EQ(cache_provider.LookUp(&look_up_key), SIZE_MAX);
}

TEST_P(BuildMMapWeightCacheProviderTest, LookUpSucceeds) {
  enum { kWeightIndex, kBiasIndex };
  ASSERT_TRUE(cache_provider.StartBuildStep());
  const auto pack_id = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                       kAlgoSeed1, kWeightIndex, kBiasIndex);
  ASSERT_TRUE(cache_provider.StopBuildStep());
  const xnn_weights_cache_look_up_key look_up_key =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex, kBiasIndex);

  EXPECT_EQ(cache_provider.LookUp(&look_up_key),
            ctx.packed_buffers.find(pack_id)->second.offset);
}

TEST_P(BuildMMapWeightCacheProviderTest,
       DifferentAlgoSeedsSameTensorsDontConflict) {
  enum { kWeightIndex, kBiasIndex };
  ASSERT_TRUE(cache_provider.StartBuildStep());
  const auto pack_id_1 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed1, kWeightIndex, kBiasIndex);
  const auto pack_id_2 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed2, kWeightIndex, kBiasIndex);
  EXPECT_TRUE(cache_provider.StopBuildStep());

  const xnn_weights_cache_look_up_key look_up_key_1 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex, kBiasIndex);
  const xnn_weights_cache_look_up_key look_up_key_2 =
      ctx.LookUpKey(kAlgoSeed2, kWeightIndex, kBiasIndex);

  EXPECT_EQ(cache_provider.LookUp(&look_up_key_1),
            ctx.packed_buffers.find(pack_id_1)->second.offset);
  EXPECT_EQ(cache_provider.LookUp(&look_up_key_2),
            ctx.packed_buffers.find(pack_id_2)->second.offset);
  EXPECT_NE(cache_provider.LookUp(&look_up_key_1),
            cache_provider.LookUp(&look_up_key_2));
}

TEST_P(BuildMMapWeightCacheProviderTest,
       SameAlgoSeedDifferentTensorsDontConflict) {
  enum { kWeightIndex1, kWeightIndex2, kBiasIndex1, kBiasIndex2 };
  ASSERT_TRUE(cache_provider.StartBuildStep());
  const auto pack_id_1 =
      ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                      kWeightIndex1, kBiasIndex1);
  const auto pack_id_2 =
      ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                      kWeightIndex2, kBiasIndex1);
  const auto pack_id_3 =
      ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                      kWeightIndex1, kBiasIndex2);
  const auto pack_id_4 =
      ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                      kWeightIndex2, kBiasIndex2);
  EXPECT_TRUE(cache_provider.StopBuildStep());

  const xnn_weights_cache_look_up_key look_up_key_1 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex1);
  const xnn_weights_cache_look_up_key look_up_key_2 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex2, kBiasIndex1);
  const xnn_weights_cache_look_up_key look_up_key_3 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex2);
  const xnn_weights_cache_look_up_key look_up_key_4 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex2, kBiasIndex2);

  EXPECT_EQ(cache_provider.LookUp(&look_up_key_1),
            ctx.packed_buffers.find(pack_id_1)->second.offset);
  EXPECT_EQ(cache_provider.LookUp(&look_up_key_2),
            ctx.packed_buffers.find(pack_id_2)->second.offset);
  EXPECT_EQ(cache_provider.LookUp(&look_up_key_3),
            ctx.packed_buffers.find(pack_id_3)->second.offset);
  EXPECT_EQ(cache_provider.LookUp(&look_up_key_4),
            ctx.packed_buffers.find(pack_id_4)->second.offset);
  EXPECT_NE(cache_provider.LookUp(&look_up_key_1),
            cache_provider.LookUp(&look_up_key_2));
  EXPECT_NE(cache_provider.LookUp(&look_up_key_1),
            cache_provider.LookUp(&look_up_key_3));
  EXPECT_NE(cache_provider.LookUp(&look_up_key_1),
            cache_provider.LookUp(&look_up_key_4))
      << pack_id_1 << " " << pack_id_4;
  EXPECT_NE(cache_provider.LookUp(&look_up_key_2),
            cache_provider.LookUp(&look_up_key_3));
  EXPECT_NE(cache_provider.LookUp(&look_up_key_2),
            cache_provider.LookUp(&look_up_key_4));
  EXPECT_NE(cache_provider.LookUp(&look_up_key_3),
            cache_provider.LookUp(&look_up_key_4));
}

TEST_P(BuildMMapWeightCacheProviderTest, BuildStepSequenceWorks) {
  enum { kWeightIndex1, kBiasIndex, kWeightIndex2 };
  ASSERT_TRUE(cache_provider.StartBuildStep());

  ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1, kWeightIndex1,
                  kBiasIndex);
  ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed2,
                  kWeightIndex2);

  EXPECT_TRUE(cache_provider.IsActive());
  // In-memory caches don't use build steps.
  EXPECT_EQ(cache_provider.IsBuilding(), !use_in_memory_cache);
  ASSERT_TRUE(cache_provider.StopBuildStep());

  ASSERT_TRUE(cache_provider.IsActive());
  EXPECT_FALSE(cache_provider.IsBuilding());
}

TEST_P(BuildMMapWeightCacheProviderTest,
       RetainAndReleaseStagingBufferAcrossSteps) {
  enum { kWeightIndex1, kBiasIndex, kWeightIndex2 };
  ASSERT_TRUE(cache_provider.StartBuildStep());
  void* const buf1 = cache_provider.ReserveSpace(4096);
  ASSERT_NE(buf1, nullptr);
  const auto pack_id_1 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed1, kWeightIndex1, kBiasIndex);
  ASSERT_TRUE(cache_provider.StopBuildStep(/*release_staging_buffer=*/false));
  EXPECT_FALSE(cache_provider.IsBuilding());

  // Step 1's buffer is mapped and readable immediately after step 1 stops.
  const xnn_weights_cache_look_up_key key_1 =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex);
  const size_t offset_1 = cache_provider.LookUp(&key_1);
  ASSERT_EQ(offset_1, ctx.packed_buffers.find(pack_id_1)->second.offset);
  EXPECT_NE(cache_provider.OffsetToAddr(offset_1), nullptr);

  // Step 2 reuses the retained staging buffer and maps step 2's buffer, while
  // ReleaseStagingBuffer is a no-op during the active build step.
  //
  // In-memory caches don't use the builder's staging buffer: every reservation
  // gets its own buffer.
  ASSERT_TRUE(cache_provider.StartBuildStep());
  if (!use_in_memory_cache) {
    EXPECT_EQ(cache_provider.ReserveSpace(1024), buf1);
    cache_provider.ReleaseStagingBuffer();
    EXPECT_EQ(cache_provider.ReserveSpace(1024), buf1);
  }
  const auto pack_id_2 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed2, kWeightIndex2);
  ASSERT_TRUE(cache_provider.StopBuildStep(/*release_staging_buffer=*/false));

  const xnn_weights_cache_look_up_key key_2 =
      ctx.LookUpKey(kAlgoSeed2, kWeightIndex2);
  const size_t offset_2 = cache_provider.LookUp(&key_2);
  ASSERT_EQ(offset_2, ctx.packed_buffers.find(pack_id_2)->second.offset);
  EXPECT_NE(cache_provider.OffsetToAddr(offset_2), nullptr);

  cache_provider.ReleaseStagingBuffer();
  ASSERT_TRUE(cache_provider.StartBuildStep());
  EXPECT_NE(cache_provider.ReserveSpace(1024), nullptr);
  ASSERT_TRUE(cache_provider.StopBuildStep());
}

TEST_P(BuildMMapWeightCacheProviderTest, MemoryLockCacheWorks) {
  enum { kWeightIndex, kBiasIndex };
  ASSERT_TRUE(cache_provider.StartBuildStep());
  ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1, kWeightIndex,
                  kBiasIndex);
  ASSERT_TRUE(cache_provider.StopBuildStep());

  EXPECT_TRUE(cache_provider.LockMemory());
  EXPECT_TRUE(cache_provider.UnlockMemory());
}

struct LoadMMapWeightCacheProviderTest : BuildMMapWeightCacheProviderTest {
  enum { kWeightIndex1, kBiasIndex, kWeightIndex2 };

  void SetUp() override {
    BuildMMapWeightCacheProviderTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    ASSERT_TRUE(cache_provider.StartBuildStep());
    pack_id_1 = ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                                kWeightIndex1, kBiasIndex);
    pack_id_2 = ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed2,
                                kWeightIndex2);
    ASSERT_TRUE(cache_provider.StopBuildStep());
    cache_provider.StopBuild();
  }

  xnn_weights_cache_look_up_key LookUpKey1() const {
    return ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex);
  }

  xnn_weights_cache_look_up_key LookUpKey2() const {
    return ctx.LookUpKey(kAlgoSeed2, kWeightIndex2);
  }

  xnn_weights_cache_look_up_key InvalidLookUpKey() const {
    return ctx.LookUpKey(kAlgoSeed3, kWeightIndex2);
  }

  PackIdentifier pack_id_1;
  PackIdentifier pack_id_2;
};

INSTANTIATE_TEST_SUITE_P(Test, LoadMMapWeightCacheProviderTest, TestVariants(),
                         TestVariant::Name);

TEST_P(LoadMMapWeightCacheProviderTest, LookUpFailsIfKeyDoesntMatch) {
  xnn_weights_cache_look_up_key look_up_key = InvalidLookUpKey();
  EXPECT_EQ(cache_provider.LookUp(&look_up_key), SIZE_MAX);
}

TEST_P(LoadMMapWeightCacheProviderTest,
       InsertOutsideOfBuildStepIsReachableAndMarksCacheAsStale) {
  char data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  xnn_weights_cache_look_up_key look_up_key = InvalidLookUpKey();
  EXPECT_THAT(cache_provider.ReserveSpace(sizeof(data)), Not(IsNull()));
  const size_t offset =
      cache_provider.LookUpOrInsert(&look_up_key, data, sizeof(data));
  EXPECT_THAT(offset, Not(Eq(SIZE_MAX)));
  EXPECT_THAT(cache_provider.OffsetToAddr(offset), Not(IsNull()));
  // Note: the in-memory weight cache doesn't write to a file so we don't care.
  if (use_explicit_fd || !use_in_memory_cache) {
    // Delete the cache provider to force a sync of the stale state.
    cache_provider = MMapWeightCacheProvider();
    EXPECT_THAT(IsCompatibleCacheFile(tmp_file), Eq(false));
  }
}

TEST_P(LoadMMapWeightCacheProviderTest, LoadingAStaleFileRestartsABuild) {
  char data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  xnn_weights_cache_look_up_key look_up_key = InvalidLookUpKey();
  ASSERT_FALSE(cache_provider.IsBuilding());
  EXPECT_THAT(cache_provider.ReserveSpace(sizeof(data)), Not(IsNull()));
  EXPECT_THAT(cache_provider.LookUpOrInsert(&look_up_key, data, sizeof(data)),
              Not(Eq(SIZE_MAX)));
  // Note: the in-memory weight cache doesn't write to a file so we don't care.
  if (use_in_memory_cache) {
    return;
  }
  // Delete the cache provider to force a sync of the stale state.
  cache_provider = MMapWeightCacheProvider();
  EXPECT_FALSE(IsCompatibleCacheFile(
      FileDescriptor::Open(tmp_file.GetCPath(), O_RDWR, 0644)));

  EXPECT_TRUE(cache_provider.LoadOrStartBuild(tmp_file.GetCPath(),
                                              tmp_file.Duplicate()));
  EXPECT_TRUE(cache_provider.CanStartBuildStep());

  // Delete the cache provider to force a sync of the stale state.
  cache_provider = MMapWeightCacheProvider();
  EXPECT_TRUE(IsCompatibleCacheFile(
      FileDescriptor::Open(tmp_file.GetCPath(), O_RDWR, 0644)));

  EXPECT_TRUE(cache_provider.Load(tmp_file.GetCPath(), tmp_file.Duplicate()));
}

TEST_P(LoadMMapWeightCacheProviderTest, LookUpSucceeds) {
  const auto& reference_1 = ctx.packed_buffers.find(pack_id_1)->second;
  const auto& reference_2 = ctx.packed_buffers.find(pack_id_2)->second;

  const xnn_weights_cache_look_up_key look_up_key_1 = LookUpKey1();
  const xnn_weights_cache_look_up_key look_up_key_2 = LookUpKey2();

  const uint64_t offset_1 = cache_provider.LookUp(&look_up_key_1);
  const uint64_t offset_2 = cache_provider.LookUp(&look_up_key_2);

  ASSERT_EQ(offset_1, reference_1.offset);
  ASSERT_EQ(offset_2, reference_2.offset);

  const void* const addr_1 = cache_provider.OffsetToAddr(offset_1);
  const void* const addr_2 = cache_provider.OffsetToAddr(offset_2);

  ASSERT_NE(addr_1, nullptr);
  ASSERT_NE(addr_2, nullptr);

  EXPECT_THAT(LightSpan<const uint8_t>(addr_1, reference_1.buffer.size()),
              ElementsAreArray(reference_1.buffer));
  EXPECT_THAT(LightSpan<const uint8_t>(addr_2, reference_2.buffer.size()),
              ElementsAreArray(reference_2.buffer));
}

struct MMapWeightCacheProviderTest : testing::TestWithParam<TestVariant> {
  void SetUp() override {
    if (use_in_memory_cache &&
        !TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider()) {
      GTEST_SKIP() << "In-memory weight cache isn't enabled for this build or "
                      "isn't supported by the current system, skipping test.";
    }
    xnn_set_fingerprint(kDefaultFingerprint);
  }
  bool use_explicit_fd = GetParam().use_explicit_fd;
  const char* const explicit_fd_path = GetParam().explicit_fd_path;
  const bool use_in_memory_cache = GetParam().use_in_memory_cache;
};

INSTANTIATE_TEST_SUITE_P(Test, MMapWeightCacheProviderTest, TestVariants(),
                         TestVariant::Name);

TEST_P(MMapWeightCacheProviderTest, XnnpackCApiJourney) {
  using std::size;
  TempFileDesc temp_fd;
  const char* temp_fd_cpath = explicit_fd_path;
  FileDescriptor temp_fd_value = temp_fd.Duplicate();
  if (!use_explicit_fd) {
    temp_fd.Close();
    temp_fd_cpath = temp_fd.GetCPath();
    temp_fd_value.Close();
    if (use_in_memory_cache) {
      temp_fd_cpath = kInMemoryCachePath;
    }
  }
  const int32_t fake_packing_algo_seed = 0xBA0BAB;
  const char packed_data_ref_1[] = "abcdefghij";
  const char packed_data_ref_2[] = "klmnopqr";
  const std::string packed_data_ref_3 =
      GenerateRandomString(/*10 MiB*/ 10 * 1024 * 1024);
  auto bytes = [](const auto& array) { return size(array) * sizeof(array[0]); };

  constexpr int kBufferCount = 10;
  // We are going to feed dummy packed data. We only need a valid pointer
  // address to map to a buffer identifier.
  char fake_buffer_pointer[kBufferCount] = {0};

  auto build_cache_provider = std::make_unique<MMapWeightCacheProvider>();
  auto load_cache_provider = std::make_unique<MMapWeightCacheProvider>();

  {  // Build and reload scenario.
    // This isn't factored between the two scenarios. When reloading the cache
    // in another process, the buffer addresses will have changed.
    TfLiteTensor tensors[kBufferCount];
    std::unordered_map<size_t, size_t> tensor_buffer_identifiers;
    for (int i = 0; i < kBufferCount; ++i) {
      tensors[i].data.data = (void*)(fake_buffer_pointer + i);
      tensors[i].bytes = 1;
      tensor_buffer_identifiers[i] = i;
    }

    MMapWeightCacheProvider& cache_provider = *build_cache_provider;
    ASSERT_TRUE(cache_provider.LoadOrStartBuild(temp_fd_cpath,
                                                temp_fd_value.Duplicate()));
    // 1st build step.
    ASSERT_TRUE(cache_provider.StartBuildStep());

    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    ASSERT_TRUE(cache_provider.MapTensorIdentifiers(tensors, size(tensors),
                                                    tensor_buffer_identifiers));

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    const xnn_weights_cache_look_up_key look_up_key_3{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[3].data.data,
        .bias = tensors[4].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    // Lookup non-packed tensor.
    ASSERT_EQ(cache->look_up(cache, &look_up_key_1), SIZE_MAX);
    // Reserve space, write data and add packed data.
    void* const reserved_ptr =
        cache->reserve_space(cache, bytes(packed_data_ref_1));
    ASSERT_NE(reserved_ptr, nullptr);
    std::memcpy(reserved_ptr, packed_data_ref_1, bytes(packed_data_ref_1));
    const size_t build_offset_1 = cache->look_up_or_insert(
        cache, &look_up_key_1, reserved_ptr, bytes(packed_data_ref_1));

    // Check that a second insertion with the same key returns the same
    // offset.
    const size_t build_offset_redundant = cache->look_up_or_insert(
        cache, &look_up_key_1, reserved_ptr, bytes(packed_data_ref_1));
    EXPECT_EQ(build_offset_1, build_offset_redundant);

    // Lookup and insert other tensor
    ASSERT_EQ(cache->look_up(cache, &look_up_key_3), SIZE_MAX);
    void* const reserved_ptr_3 =
        cache->reserve_space(cache, bytes(packed_data_ref_3));
    ASSERT_NE(reserved_ptr_3, nullptr);
    std::memcpy(reserved_ptr_3, packed_data_ref_3.data(),
                bytes(packed_data_ref_3));
    const size_t build_offset_3 = cache->look_up_or_insert(
        cache, &look_up_key_3, reserved_ptr_3, bytes(packed_data_ref_3));

    ASSERT_TRUE(cache_provider.StopBuildStep());

    // Lookup newly packed tensor.
    ASSERT_EQ(cache->look_up(cache, &look_up_key_1), build_offset_1);
    ASSERT_EQ(cache->look_up(cache, &look_up_key_3), build_offset_3);

    // 2nd build step.
    ASSERT_TRUE(cache_provider.StartBuildStep());

    // Add a tensor without reserving before.
    const xnn_weights_cache_look_up_key look_up_key_2{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[2].data.data,
        .bias = tensors[3].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    const size_t build_offset_2 = cache->look_up_or_insert(
        cache, &look_up_key_2, (void*)packed_data_ref_2,
        bytes(packed_data_ref_2));

    // Buffer inserted during build step 1 can be looked up.
    EXPECT_EQ(cache->look_up(cache, &look_up_key_3), build_offset_3);
    // Reinsert buffer inserted during build step 1 should be a no-op.
    EXPECT_EQ(cache->look_up_or_insert(cache, &look_up_key_3, reserved_ptr_3,
                                       bytes(packed_data_ref_3)),
              build_offset_3);

    // Save the cache to disk and reload.
    ASSERT_TRUE(cache_provider.StopBuildStep());

    ASSERT_TRUE(cache->is_finalized(cache));

    const size_t reload_offset_1 = cache->look_up(cache, &look_up_key_1);
    ASSERT_EQ(reload_offset_1, build_offset_1);

    const void* const loaded_packed_data_1 =
        cache->offset_to_addr(cache, reload_offset_1);
    ASSERT_NE(loaded_packed_data_1, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_1, size(packed_data_ref_1)),
        ElementsAreArray(packed_data_ref_1));

    const size_t reload_offset_2 = cache->look_up(cache, &look_up_key_2);
    ASSERT_EQ(reload_offset_2, build_offset_2);

    const void* const loaded_packed_data_2 =
        cache->offset_to_addr(cache, reload_offset_2);
    ASSERT_NE(loaded_packed_data_2, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_2, size(packed_data_ref_2)),
        ElementsAreArray(packed_data_ref_2));

    const size_t reload_offset_3 = cache->look_up(cache, &look_up_key_3);
    ASSERT_EQ(reload_offset_3, build_offset_3);

    const void* const loaded_packed_data_3 =
        cache->offset_to_addr(cache, reload_offset_3);
    ASSERT_NE(loaded_packed_data_3, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_3, size(packed_data_ref_3)),
        ElementsAreArray(packed_data_ref_3));
  }

  {  // Load existing cache scenario.
    TfLiteTensor tensors[kBufferCount];
    std::unordered_map<size_t, size_t> tensor_buffer_identifiers;
    for (int i = 0; i < kBufferCount; ++i) {
      tensors[i].data.data = (void*)(fake_buffer_pointer + i);
      tensors[i].bytes = 1;
      tensor_buffer_identifiers[i] = i;
    }

    // When testing the in-memory cache, we reuse the cache provider used for
    // building the cache. Otherwise we us a new one.
    MMapWeightCacheProvider* cache_provider = build_cache_provider.get();
    if (use_in_memory_cache) {
      load_cache_provider.reset();
    } else {
      build_cache_provider.reset();
      cache_provider = load_cache_provider.get();
      ASSERT_TRUE(cache_provider->LoadOrStartBuild(temp_fd_cpath,
                                                   temp_fd_value.Duplicate()));
      ASSERT_TRUE(cache_provider->MapTensorIdentifiers(
          tensors, size(tensors), tensor_buffer_identifiers));
    }
    xnn_weights_cache_t cache = &cache_provider->GetCacheProvider();

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    const xnn_weights_cache_look_up_key look_up_key_2{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[2].data.data,
        .bias = tensors[3].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    const xnn_weights_cache_look_up_key look_up_key_3{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[3].data.data,
        .bias = tensors[4].data.data,
        .fingerprint_id = kDefaultFingerprint.id};

    ASSERT_TRUE(cache->is_finalized(cache));

    const size_t offset_1 = cache->look_up(cache, &look_up_key_1);
    const void* const loaded_packed_data_1 =
        cache->offset_to_addr(cache, offset_1);
    ASSERT_NE(loaded_packed_data_1, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_1, size(packed_data_ref_1)),
        ElementsAreArray(packed_data_ref_1));

    const size_t offset_2 = cache->look_up(cache, &look_up_key_2);
    ASSERT_NE(offset_2, SIZE_MAX);
    const void* const loaded_packed_data_2 =
        cache->offset_to_addr(cache, offset_2);
    ASSERT_NE(loaded_packed_data_2, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_2, size(packed_data_ref_2)),
        ElementsAreArray(packed_data_ref_2));

    const size_t offset_3 = cache->look_up(cache, &look_up_key_3);
    const void* const loaded_packed_data_3 =
        cache->offset_to_addr(cache, offset_3);
    ASSERT_NE(loaded_packed_data_3, nullptr);
    EXPECT_THAT(
        LightSpan<const char>(loaded_packed_data_3, size(packed_data_ref_3)),
        ElementsAreArray(packed_data_ref_3));
  }
}

TEST_P(MMapWeightCacheProviderTest, CacheIsRebuiltOnFingerprintMismatch) {
  if (use_in_memory_cache) {
    GTEST_SUCCEED() << "In-memory cache is never reloaded.";
    return;
  }
  TempFileDesc temp_fd;
  const char* temp_fd_cpath = explicit_fd_path;

  xnn_fingerprint test_fingeprint{0x7357, 0xF33D};
  {  // Build a cache file with a specific fingerprint.
    // Clear fingerprints and add a test fingerprint to XNNPack.
    xnn_clear_fingerprints();
    xnn_set_fingerprint(test_fingeprint);

    // Build a cache file.
    MMapWeightCacheProvider cache_provider;

    const char kernel[] = "Fake data.";
    TfLiteTensor tensor;
    tensor.data.data = (void*)kernel;
    tensor.bytes = sizeof(kernel);
    ASSERT_TRUE(cache_provider.MapTensorIdentifiers(
        &tensor, /*size=*/1, /*tensor_index_to_identifier=*/{{0, 1}}));
    ASSERT_TRUE(
        cache_provider.LoadOrStartBuild(temp_fd_cpath, temp_fd.Duplicate()));
    ASSERT_TRUE(cache_provider.StartBuildStep());
    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = 1234,
        .kernel = kernel,
        .bias = nullptr,
        .fingerprint_id = test_fingeprint.id};
    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    const size_t build_offset_1 = cache->look_up_or_insert(
        cache, &look_up_key_1,
        const_cast<void*>(reinterpret_cast<const void*>(kernel)),
        sizeof(kernel));
    (void)build_offset_1;
    ASSERT_TRUE(cache_provider.StopBuildStep());
  }

  if (!use_explicit_fd) {
    temp_fd.Close();
    temp_fd_cpath = temp_fd.GetCPath();
  }

  // Change the test fingerprint value.
  test_fingeprint.value = 0xdeadb33f;
  xnn_set_fingerprint(test_fingeprint);

  // Reload the file.
  auto build_cache_provider = std::make_unique<MMapWeightCacheProvider>();
  MMapWeightCacheProvider& cache_provider = *build_cache_provider;
  ASSERT_TRUE(
      cache_provider.LoadOrStartBuild(temp_fd_cpath, temp_fd.Duplicate()));
  ASSERT_TRUE(cache_provider.StartBuildStep());
}

enum class IsCompatibleCacheFileTestOverload { kPath, kDescriptor };

class IsCompatibleCacheFileTest
    : public testing::TestWithParam<IsCompatibleCacheFileTestOverload> {
 public:
  using Param = IsCompatibleCacheFileTestOverload;

  void SetUp() override {
    xnn_clear_fingerprints();
    xnn_set_fingerprint(kDefaultFingerprint);

    // Build a cache file.
    MMapWeightCacheProvider cache_provider;

    const char kernel[] = "Fake data.";
    TfLiteTensor tensor;
    tensor.data.data = (void*)kernel;
    tensor.bytes = sizeof(kernel);
    ASSERT_TRUE(cache_provider.MapTensorIdentifiers(
        &tensor, /*size=*/1, /*tensor_index_to_identifier=*/{{0, 1}}));
    ASSERT_TRUE(
        cache_provider.LoadOrStartBuild(fd_.GetCPath(), fd_.Duplicate()));
    ASSERT_TRUE(cache_provider.StartBuildStep());
    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = 1234,
        .kernel = kernel,
        .bias = nullptr,
        .fingerprint_id = kDefaultFingerprint.id};
    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    const size_t build_offset_1 = cache->look_up_or_insert(
        cache, &look_up_key_1,
        const_cast<void*>(reinterpret_cast<const void*>(kernel)),
        sizeof(kernel));
    (void)build_offset_1;
    ASSERT_TRUE(cache_provider.StopBuildStep());
  }

  void ChangeRuntimeFingerprintValue() {
    xnn_set_fingerprint(
        {kDefaultFingerprint.id, kDefaultFingerprint.value + 1});
  }

  bool CallIsCompatibleCacheFile() {
    switch (GetParam()) {
      case Param::kPath:
        fd_.Close();
        return IsCompatibleCacheFile(fd_.GetCPath());
      case Param::kDescriptor: {
        const auto pos = fd_.GetPos();
        EXPECT_NE(pos, 0);  // We test with a non zero position.
        return IsCompatibleCacheFile(fd_);
        EXPECT_EQ(fd_.GetPos(), pos);
      }
    }
  }

  TempFileDesc fd_;
};

std::string Name(
    const testing::TestParamInfo<IsCompatibleCacheFileTestOverload>& info) {
  switch (info.param) {
    case IsCompatibleCacheFileTestOverload::kPath:
      return "WithPathOverload";
    case IsCompatibleCacheFileTestOverload::kDescriptor:
      return "WithFileDescriptorOverload";
  }
}

TEST_P(IsCompatibleCacheFileTest, ReturnsTrueWhenFingerprintMatches) {
  EXPECT_TRUE(CallIsCompatibleCacheFile());
}

TEST_P(IsCompatibleCacheFileTest, ReturnsFalseWhenFingerprintMismatches) {
  ChangeRuntimeFingerprintValue();
  EXPECT_FALSE(CallIsCompatibleCacheFile());
}

TEST_P(IsCompatibleCacheFileTest, ReturnsFalseWhenFingerprintIsNotFound) {
  xnn_clear_fingerprints();
  EXPECT_FALSE(CallIsCompatibleCacheFile());
}

TEST_P(IsCompatibleCacheFileTest, ReturnsFalseWhenVersionMismatches) {
  constexpr decltype(XNNPackCacheHeader::version) bad_version =
      XNNPackCacheHeader::kInvalidHeader;
  constexpr size_t version_offset = offsetof(XNNPackCacheHeader, version);
  ASSERT_EQ(fd_.SetPos(version_offset), version_offset);
  ASSERT_TRUE(fd_.Write(&bad_version, sizeof(bad_version)));
  EXPECT_FALSE(CallIsCompatibleCacheFile());
}

TEST_P(IsCompatibleCacheFileTest, ReturnsFalseWhenMagicValueMismatches) {
  constexpr decltype(XNNPackCacheHeader::magic_value) bad_magic =
      ~XNNPackCacheHeader::kMagicValue;
  constexpr size_t magic_offset = offsetof(XNNPackCacheHeader, magic_value);
  ASSERT_EQ(fd_.SetPos(magic_offset), magic_offset);
  ASSERT_TRUE(fd_.Write(&bad_magic, sizeof(bad_magic)));
  EXPECT_FALSE(CallIsCompatibleCacheFile());
}

INSTANTIATE_TEST_SUITE_P(
    Test, IsCompatibleCacheFileTest,
    testing::Values(IsCompatibleCacheFileTest::Param::kPath,
                    IsCompatibleCacheFileTest::Param::kDescriptor),
    Name);

class MaliciousCacheTest : public WeightCacheBuilderTest {
 public:
  void SetUp() override {
    WeightCacheBuilderTest::SetUp();
    ASSERT_TRUE(fd_.IsValid());
    {  // Build a "valid" cache.
      const PackIdentifier dummy_id{1, 2, 3};
      WeightCacheBuilder builder;
      ASSERT_TRUE(builder.Start(fd_.GetCPath(), fd_));
      ASSERT_TRUE(builder.StartBuildStep());
      const std::string payload = "payload data";
      auto loc = builder.Append(dummy_id, payload.c_str(), payload.size(),
                                kDefaultFingerprint.id);
      (void)loc;
      ASSERT_TRUE(builder.StopBuildStep());
    }
  }

  void TamperWithCacheFile(
      std::function<void(cache::schema::BufferListT&)> EditSchema) {
    fd_.SetPos(0);
    XNNPackCacheHeader header;
    ASSERT_TRUE(fd_.Read(&header, sizeof(header)));
    std::vector<uint8_t> fb_data(header.buffer_list_size);
    fd_.SetPos(header.buffer_list_offset);
    ASSERT_TRUE(fd_.Read(fb_data.data(), header.buffer_list_size));
    const cache::schema::BufferList* buffer_list =
        cache::schema::GetBufferList(fb_data.data());

    cache::schema::BufferListT schema;
    buffer_list->UnPackTo(&schema);

    EditSchema(schema);

    flatbuffers::FlatBufferBuilder fbb;
    cache::schema::FinishBufferListBuffer(
        fbb, cache::schema::BufferList::Pack(fbb, &schema));

    fd_.SetPos(header.buffer_list_offset);
    ASSERT_TRUE(fd_.Write(fbb.GetBufferPointer(), fbb.GetSize()));

    header.buffer_list_size = fbb.GetSize();
    fd_.SetPos(0);
    ASSERT_TRUE(fd_.Write(&header, sizeof(header)));
    fd_.Close();
  }

 protected:
  TempFileDesc fd_;
};

TEST_F(MaliciousCacheTest, BufferStartsOutOfBounds) {
  TamperWithCacheFile([this](cache::schema::BufferListT& schema) {
    ASSERT_GE(schema.buffers.size(), 1);
    schema.buffers[0]->offset = fd_.Size() + 1;
  });

  MMapWeightCacheProvider cache_provider;
  EXPECT_FALSE(cache_provider.Load(fd_.GetCPath()));
}

TEST_F(MaliciousCacheTest, BufferEndsOutOfBounds) {
  ASSERT_NO_FATAL_FAILURE(
      TamperWithCacheFile([this](cache::schema::BufferListT& schema) {
        ASSERT_GE(schema.buffers.size(), 1);
        ASSERT_TRUE(fd_.Size() > schema.base_offset);
        schema.buffers[0]->size = fd_.Size() - schema.base_offset + 1;
      }));

  MMapWeightCacheProvider cache_provider;
  EXPECT_FALSE(cache_provider.Load(fd_.GetCPath()));
}

TEST_F(MaliciousCacheTest, BufferOffsetPlusSizeOverflowsInt64) {
  ASSERT_NO_FATAL_FAILURE(
      TamperWithCacheFile([](cache::schema::BufferListT& schema) {
        ASSERT_GE(schema.buffers.size(), 1);
        // Overflow uint64_t computations with a wrap around.
        schema.buffers[0]->size = std::numeric_limits<size_t>::max();
      }));

  MMapWeightCacheProvider cache_provider;
  EXPECT_FALSE(cache_provider.Load(fd_.GetCPath()));
}

class InMemoryWeightCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    xnn_set_fingerprint(kDefaultFingerprint);
    ASSERT_TRUE(provider_.LoadOrStartBuild(kInMemoryCachePath));
  }

  static xnn_weights_cache_look_up_key Key(uint32_t seed,
                                           const std::string& weights) {
    return {.seed = seed,
            .kernel = weights.data(),
            .bias = nullptr,
            .fingerprint_id = kDefaultFingerprint.id};
  }

  // Packs `weights` the way XNNPack does and returns the cache offset.
  static size_t Insert(MMapWeightCacheProvider& provider,
                       const xnn_weights_cache_look_up_key& key,
                       const std::string& weights) {
    void* const reserved = provider.ReserveSpace(weights.size());
    if (reserved == nullptr) {
      return SIZE_MAX;
    }
    std::memcpy(reserved, weights.data(), weights.size());
    return provider.LookUpOrInsert(&key, reserved, weights.size());
  }

  size_t Insert(const xnn_weights_cache_look_up_key& key,
                const std::string& weights) {
    return Insert(provider_, key, weights);
  }

  MMapWeightCacheProvider provider_;
};

TEST(InMemoryWeightCacheProviderTest, IsAlwaysAvailable) {
  EXPECT_TRUE(TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider());
}

TEST_F(InMemoryWeightCacheTest, DoesntNeedBuildSteps) {
  EXPECT_TRUE(provider_.IsActive());
  EXPECT_FALSE(provider_.CanStartBuildStep());
  EXPECT_FALSE(provider_.IsBuilding());
  // Build steps are accepted but don't do anything.
  EXPECT_TRUE(provider_.StartBuildStep());
  EXPECT_FALSE(provider_.IsBuilding());
  EXPECT_TRUE(provider_.StopBuildStep());
}

TEST_F(InMemoryWeightCacheTest, BuffersHaveStableAddresses) {
  std::vector<std::string> weights;
  std::vector<size_t> offsets;
  std::vector<const void*> addresses;
  for (int i = 0; i < 16; ++i) {
    weights.push_back(GenerateRandomString(100000 + i));
    ASSERT_TRUE(provider_.MapBufferIdentifier(
        weights.back().data(), weights.back().size(), /*identifier=*/i))
        << i;
    offsets.push_back(Insert(Key(/*seed=*/1, weights.back()), weights.back()));
    ASSERT_NE(offsets.back(), SIZE_MAX) << i;
    // The address is usable as soon as the buffer is inserted.
    addresses.push_back(provider_.OffsetToAddr(offsets.back()));
    ASSERT_NE(addresses.back(), nullptr) << i;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(provider_.OffsetToAddr(offsets[i]), addresses[i]) << i;
    EXPECT_THAT(LightSpan<const char>(addresses[i], weights[i].size()),
                ElementsAreArray(weights[i]))
        << i;
  }
}

TEST_F(InMemoryWeightCacheTest, DeduplicatesBuffers) {
  const std::string weights = GenerateRandomString(10000);
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  const xnn_weights_cache_look_up_key key = Key(/*seed=*/1, weights);
  const size_t offset = Insert(key, weights);
  ASSERT_NE(offset, SIZE_MAX);
  EXPECT_EQ(provider_.LookUp(&key), offset);
  EXPECT_EQ(Insert(key, weights), offset);

  // The same weights packed differently get their own buffer.
  const xnn_weights_cache_look_up_key other_key = Key(/*seed=*/2, weights);
  const size_t other_offset = Insert(other_key, weights);
  ASSERT_NE(other_offset, SIZE_MAX);
  EXPECT_NE(other_offset, offset);
}

TEST_F(InMemoryWeightCacheTest, ReleaseClearsTheCache) {
  const std::string weights = GenerateRandomString(64);
  const xnn_weights_cache_look_up_key key = Key(/*seed=*/1, weights);
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  const size_t offset = Insert(key, weights);
  ASSERT_NE(offset, SIZE_MAX);

  provider_.Release();
  EXPECT_FALSE(provider_.IsActive());

  // A released provider can start a new cache whose offsets point to the new
  // data.
  const std::string other_weights = GenerateRandomString(64);
  ASSERT_TRUE(provider_.StartBuild(kInMemoryCachePath));
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  EXPECT_EQ(provider_.LookUp(&key), SIZE_MAX);
  const size_t new_offset = Insert(key, other_weights);
  ASSERT_NE(new_offset, SIZE_MAX);
  EXPECT_THAT(LightSpan<const char>(provider_.OffsetToAddr(new_offset),
                                    other_weights.size()),
              ElementsAreArray(other_weights));
}

TEST_F(InMemoryWeightCacheTest, StartingAFileBuildReplacesTheCache) {
  const std::string weights = GenerateRandomString(64);
  const xnn_weights_cache_look_up_key key = Key(/*seed=*/1, weights);
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  ASSERT_NE(Insert(key, weights), SIZE_MAX);

  // Starting a file cache without releasing the in-memory one first forgets
  // the in-memory buffers but keeps the buffer identifiers.
  TempFileDesc tmp_file{TempFileDesc::kAutoClose};
  ASSERT_TRUE(provider_.StartBuild(tmp_file.GetCPath()));
  EXPECT_TRUE(provider_.CanStartBuildStep());
  EXPECT_EQ(provider_.LookUp(&key), SIZE_MAX);

  const std::string file_weights = GenerateRandomString(64);
  ASSERT_TRUE(provider_.StartBuildStep());
  EXPECT_TRUE(provider_.IsBuilding());
  const size_t offset = Insert(key, file_weights);
  ASSERT_NE(offset, SIZE_MAX);
  ASSERT_TRUE(provider_.StopBuildStep());

  EXPECT_EQ(provider_.LookUp(&key), offset);
  EXPECT_THAT(LightSpan<const char>(provider_.OffsetToAddr(offset),
                                    file_weights.size()),
              ElementsAreArray(file_weights));
}

TEST_F(InMemoryWeightCacheTest, LoadingAFileReplacesTheCache) {
  const std::string weights = GenerateRandomString(64);
  const xnn_weights_cache_look_up_key key = Key(/*seed=*/1, weights);
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  ASSERT_NE(Insert(key, weights), SIZE_MAX);

  // Build a cache file that holds other data for the same key.
  TempFileDesc tmp_file{TempFileDesc::kAutoClose};
  const std::string file_weights = GenerateRandomString(64);
  size_t file_offset = SIZE_MAX;
  {
    MMapWeightCacheProvider file_provider;
    ASSERT_TRUE(file_provider.StartBuild(tmp_file.GetCPath()));
    ASSERT_TRUE(file_provider.MapBufferIdentifier(
        weights.data(), weights.size(), /*identifier=*/1));
    ASSERT_TRUE(file_provider.StartBuildStep());
    file_offset = Insert(file_provider, key, file_weights);
    ASSERT_TRUE(file_provider.StopBuildStep());
  }
  ASSERT_NE(file_offset, SIZE_MAX);

  // Loading the file without releasing the in-memory cache first must not
  // leave in-memory buffers behind for the file offsets.
  ASSERT_TRUE(provider_.Load(tmp_file.GetPath()));
  EXPECT_EQ(provider_.LookUp(&key), file_offset);
  EXPECT_THAT(LightSpan<const char>(provider_.OffsetToAddr(file_offset),
                                    file_weights.size()),
              ElementsAreArray(file_weights));
}

TEST_F(InMemoryWeightCacheTest, MovedFromProviderIsInactive) {
  MMapWeightCacheProvider moved_to(std::move(provider_));
  EXPECT_TRUE(moved_to.IsActive());
  EXPECT_FALSE(provider_.IsActive());  // NOLINT(bugprone-use-after-move)

  MMapWeightCacheProvider assigned_to;
  assigned_to = std::move(moved_to);
  EXPECT_TRUE(assigned_to.IsActive());
  EXPECT_FALSE(moved_to.IsActive());  // NOLINT(bugprone-use-after-move)
}

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
TEST_F(InMemoryWeightCacheTest, BuffersUseAnonymousMapping) {
  const std::string weights = GenerateRandomString(4096);
  ASSERT_TRUE(provider_.MapBufferIdentifier(weights.data(), weights.size(),
                                            /*identifier=*/1));
  const size_t offset = Insert(Key(/*seed=*/1, weights), weights);
  ASSERT_NE(offset, SIZE_MAX);
  const void* const addr = provider_.OffsetToAddr(offset);
  ASSERT_NE(addr, nullptr);

  const std::string matched_line = GetMapsLine(addr);
  ASSERT_FALSE(matched_line.empty());
  EXPECT_THAT(matched_line, Not(HasSubstr("[heap]")));
  if (AnonVmaNamingSupported()) {
    EXPECT_THAT(matched_line, EndsWith("[anon:xnnpack-weights-cache]"));
  }
}
#endif

TEST(MMapWeightCacheProviderReleaseTest, ReleaseForgetsTheCacheFile) {
  TempFileDesc tmp_file{TempFileDesc::kAutoClose};
  MMapWeightCacheProvider provider;
  ASSERT_TRUE(provider.StartBuild(tmp_file.GetCPath()));
  ASSERT_TRUE(provider.IsActive());

  provider.Release();
  EXPECT_FALSE(provider.IsActive());
  EXPECT_FALSE(provider.CanStartBuildStep());
  EXPECT_TRUE(provider.GetFilePath().empty());
}

}  // namespace
}  // namespace tflite::xnnpack
