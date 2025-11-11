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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
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

TEST(WeightCacheBuilderTest, ReserveAppendWriteWorks) {
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
  auto loc = builder.Append(dummy_id, buffer, payload_size);

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_GE(builder.capacity(), payload_size);

  ASSERT_TRUE(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
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

TEST(WeightCacheBuilderTest, AppendWithoutReserveWriteWorks) {
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
  auto loc = builder.Append(dummy_id, payload.c_str(), payload_size);

  EXPECT_EQ(loc.size, payload_size);

  ASSERT_TRUE(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(cache_path.c_str()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
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

TEST(WeightCacheBuilderTest, CorruptBufferListFailsGracefully) {
  const std::string cache_path = testing::TempDir() + "/cache";
  const std::string payload = "This is some data in the file.";
  const PackIdentifier dummy_id{1, 2, 3};

  FileDescriptor file_descriptor = FileDescriptor::Open(
      cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str(), file_descriptor));
  ASSERT_TRUE(builder.StartBuildStep());

  const size_t payload_size = size(payload);
  auto loc = builder.Append(dummy_id, payload.c_str(), payload_size);
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
}

TEST(WeightCacheBuilderTest, InvalidFileDescriptorFails) {
  WeightCacheBuilder builder;
  EXPECT_FALSE(builder.Start("", FileDescriptor()));
  EXPECT_FALSE(builder.Start("/seldf/sedsft", FileDescriptor()));
}

TEST(WeightCacheBuilderTest, InMemoryCacheCanBeBuilt) {
  if (!TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider()) {
    GTEST_SKIP() << "In-memory weight cache isn't enabled for this build or "
                    "isn't supported by the current system, skipping test.";
  }
  WeightCacheBuilder builder;
  EXPECT_TRUE(
      builder.Start(kInMemoryCachePath, CreateInMemoryFileDescriptor(nullptr)));
  EXPECT_TRUE(builder.IsStarted());
  const FileDescriptor file_fd =
      FileDescriptor::Open(kInMemoryCachePath, O_RDONLY);
  EXPECT_FALSE(file_fd.IsValid());
  EXPECT_EQ(errno, ENOENT);
}

TEST(WeightCacheBuilderTest, MultipleStepBuild) {
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
    const auto loc = builder.Append(dummy_id1, buffer, payload_size);
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }
  {
    const size_t payload_size = size(payload3);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload3.c_str(), payload_size);
    const auto loc = builder.Append(dummy_id3, buffer, payload_size);
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
    const auto loc = builder.Append(dummy_id2, buffer, payload_size);
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }

  ASSERT_TRUE(builder.StopBuildStep());

  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

  const XNNPackCacheHeader& header =
      *reinterpret_cast<const XNNPackCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, XNNPackCacheHeader::kVersion);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

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
            .bias = nullptr};
  }

  // Creates a look up key for the XNNPack weight provider C interface.
  xnn_weights_cache_look_up_key LookUpKey(const uint32_t algorithm_seed,
                                          const int weights_index,
                                          const int bias_index) const {
    return {.seed = algorithm_seed,
            .kernel = buffers[weights_index].data(),
            .bias = buffers[bias_index].data()};
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
    cache_provider.MapTensorIdentifiers(ctx.tensors.data(), ctx.tensors.size(),
                                        ctx.tensor_buffer_identifiers);
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
  EXPECT_TRUE(cache_provider.IsBuilding());
  ASSERT_TRUE(cache_provider.StopBuildStep());

  ASSERT_TRUE(cache_provider.IsActive());
  EXPECT_FALSE(cache_provider.IsBuilding());
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
  }

  xnn_weights_cache_look_up_key LookUpKey1() const {
    return ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex);
  }

  xnn_weights_cache_look_up_key LookUpKey2() const {
    return ctx.LookUpKey(kAlgoSeed2, kWeightIndex2);
  }

  PackIdentifier pack_id_1;
  PackIdentifier pack_id_2;
};

INSTANTIATE_TEST_SUITE_P(Test, LoadMMapWeightCacheProviderTest, TestVariants(),
                         TestVariant::Name);

TEST_P(LoadMMapWeightCacheProviderTest, LookUpFailsIfKeyDoesntMatch) {
  xnn_weights_cache_look_up_key look_up_key{};
  EXPECT_EQ(cache_provider.LookUp(&look_up_key), SIZE_MAX);
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
      tensor_buffer_identifiers[i] = i;
    }

    MMapWeightCacheProvider& cache_provider = *build_cache_provider;
    ASSERT_TRUE(cache_provider.LoadOrStartBuild(temp_fd_cpath,
                                                temp_fd_value.Duplicate()));
    // 1st build step.
    ASSERT_TRUE(cache_provider.StartBuildStep());

    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    cache_provider.MapTensorIdentifiers(tensors, size(tensors),
                                        tensor_buffer_identifiers);

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data};

    const xnn_weights_cache_look_up_key look_up_key_3{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[3].data.data,
        .bias = tensors[4].data.data};

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
        .bias = tensors[3].data.data};

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
      cache_provider->MapTensorIdentifiers(tensors, size(tensors),
                                           tensor_buffer_identifiers);
    }
    xnn_weights_cache_t cache = &cache_provider->GetCacheProvider();

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data};

    const xnn_weights_cache_look_up_key look_up_key_2{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[2].data.data,
        .bias = tensors[3].data.data};

    const xnn_weights_cache_look_up_key look_up_key_3{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[3].data.data,
        .bias = tensors[4].data.data};

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

TEST_P(MMapWeightCacheProviderTest, XnnpackRebuildOnVersionMismatch) {
  TempFileDesc temp_fd;
  const char* temp_fd_cpath = explicit_fd_path;
  FileDescriptor temp_fd_value = temp_fd.Duplicate();

  {  // Set bad build identifier
    XNNPackCacheHeader header{.version = XNNPackCacheHeader::kVersion};
    header.xnnpack_build_identifier[0] += 1;
    ASSERT_TRUE(temp_fd_value.Write(&header, sizeof(header)));
  }

  if (!use_explicit_fd) {
    temp_fd.Close();
    temp_fd_cpath = temp_fd.GetCPath();
    temp_fd_value.Close();
    if (use_in_memory_cache) {
      temp_fd_cpath = kInMemoryCachePath;
    }
  }

  auto build_cache_provider = std::make_unique<MMapWeightCacheProvider>();
  MMapWeightCacheProvider& cache_provider = *build_cache_provider;
  ASSERT_TRUE(cache_provider.LoadOrStartBuild(temp_fd_cpath,
                                              temp_fd_value.Duplicate()));
  ASSERT_TRUE(cache_provider.StartBuildStep());
}

class IsCompatibleCacheFileTest : public testing::Test {
 public:
  void SetUp() override {
    header_.version = XNNPackCacheHeader::kVersion;
    memcpy(header_.xnnpack_build_identifier,
           xnn_experimental_get_build_identifier_data(),
           xnn_experimental_get_build_identifier_size());
  }

  bool WriteHeaderAndReturnIsCompatibleCacheFile() {
    const bool res = fd_.Write(&header_, sizeof(header_));
    fd_.Close();
    return res && IsCompatibleCacheFile(fd_.GetCPath());
  }

  XNNPackCacheHeader header_{};
  TempFileDesc fd_;
};

TEST_F(IsCompatibleCacheFileTest, ReturnsTrueForACorrectHeader) {
  EXPECT_TRUE(WriteHeaderAndReturnIsCompatibleCacheFile());
}

TEST_F(IsCompatibleCacheFileTest, ReturnsFalseForWrongHeaderVersion) {
  header_.version += 1;
  EXPECT_FALSE(WriteHeaderAndReturnIsCompatibleCacheFile());
}

TEST_F(IsCompatibleCacheFileTest, ReturnsFalseForWrongBuildIdentifier) {
  header_.xnnpack_build_identifier[0] += 1;
  EXPECT_FALSE(WriteHeaderAndReturnIsCompatibleCacheFile());
}

}  // namespace
}  // namespace tflite::xnnpack
