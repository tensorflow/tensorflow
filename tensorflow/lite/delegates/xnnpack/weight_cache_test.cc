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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
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
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"

namespace tflite::xnnpack {

std::ostream& operator<<(std::ostream& os, const PackIdentifier& p) {
  return os << "PackIdentifier{pack_algo: " << p.pack_algorithm_id
            << ", weights_id: " << p.weights_id << ", bias_id: " << p.bias_id
            << "}";
}

namespace {

using testing::ElementsAreArray;
using testing::Ge;

// Wraps a call to `mkstemp` to create temporary files.
class TempFileDesc {
 public:
  static constexpr struct AutoClose {
  } kAutoCLose{};

#if defined(_MSC_VER)
  TempFileDesc() : fd_() {
    char filename[L_tmpnam_s];
    errno_t err = tmpnam_s(filename, L_tmpnam_s);
    if (err) {
      fprintf(stderr, "Could not create temporary filename.\n");
      std::abort();
    }
    path_ = filename;
    fd_ = open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "Could not create temporary filename.\n");
      std::abort();
    }
  }
#else
  TempFileDesc() : fd_(mkstemp(path_.data())) {
    if (GetFd() < 0) {
      perror("Could not create temporary file");
    }
  }
#endif

  explicit TempFileDesc(AutoClose) : TempFileDesc() { Close(); }

  TempFileDesc(const TempFileDesc&) = delete;
  TempFileDesc& operator=(const TempFileDesc&) = delete;

  friend void swap(TempFileDesc& a, TempFileDesc& b) {
    std::swap(a.path_, b.path_);
    std::swap(a.fd_, b.fd_);
  }

  TempFileDesc(TempFileDesc&& other) { swap(*this, other); }
  TempFileDesc& operator=(TempFileDesc&& other) {
    swap(*this, other);
    return *this;
  }

  ~TempFileDesc() { Close(); }

  void Close() {
    if (GetFd() >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  const std::string& GetPath() const { return path_; }

  const char* GetCPath() const { return path_.c_str(); }

  int GetFd() const { return fd_; }

  bool IsOpen() const { return fd_ >= 0; }

 private:
  std::string path_ = testing::TempDir() + "/weight_cache_test_file.XXXXXX";
  int fd_ = -1;
};

TEST(MMapHandleTest, DefaultConstructs) {
  MMapHandle handle;
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MapNonExitxingFileFails) {
  // I hope this path doesn't exist...
  const char* file_path = "sdbgfd";
  MMapHandle handle;
  EXPECT_FALSE(handle.Map(file_path));
}

TEST(MMapHandleTest, MapExistingFileWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), size(payload)),
            size(payload));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));
  EXPECT_TRUE(handle.IsMapped());
  EXPECT_NE(handle.data(), nullptr);
  EXPECT_THAT(handle.size(), Ge(size(payload)));
  EXPECT_THAT(handle, ElementsAreArray(payload));

  handle.UnMap();
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MoveConstructs) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file;
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), size(payload)),
            size(payload));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_TRUE(handle.Map(tmp_file.GetCPath()));

  MMapHandle handle2(std::move(handle));

  // We are checking that the moved from handle has lost control over the data.
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
  // NOLINTEND(bugprone-use-after-move)

  EXPECT_TRUE(handle2.IsMapped());
  EXPECT_NE(handle2.data(), nullptr);
  EXPECT_THAT(handle2.size(), Ge(size(payload)));
  EXPECT_THAT(handle2, ElementsAreArray(payload));
}

TEST(WeightCacheBuilderTest, ReserveAppendWriteWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const PackIdentifier dummy_id{1, 2, 3};

  WeightCacheBuilder builder;
  const std::string cache_path = testing::TempDir() + "/cache";
  ASSERT_TRUE(builder.Start(cache_path.c_str()));

  const size_t payload_size = size(payload);
  void* buffer = builder.Reserve(payload_size);
  std::memcpy(buffer, payload.c_str(), payload_size);
  auto loc = builder.Append(dummy_id, buffer, payload_size);

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_GE(builder.capacity(), payload_size);
  EXPECT_TRUE(builder.ShouldFinalize());

  ASSERT_TRUE(builder.Finalize());

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
  WeightCacheBuilder builder;
  ASSERT_TRUE(builder.Start(cache_path.c_str()));

  const size_t payload_size = size(payload);
  auto loc = builder.Append(dummy_id, payload.c_str(), payload_size);

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_TRUE(builder.ShouldFinalize());

  ASSERT_TRUE(builder.Finalize());

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

TEST(WeightCacheBuilderTest, NonExistingPathFails) {
  using std::size;
  WeightCacheBuilder builder;
  EXPECT_FALSE(builder.Start(""));
  EXPECT_FALSE(builder.Start("/seldf/sedsft"));
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

    // Add the packed buffer to the XNNPack cache. Normaly you would pack in
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

struct BuildMMapWeightCacheProviderTest : testing::Test {
  enum { kAlgoSeed1, kAlgoSeed2, kAlgoSeed3 };
  enum { kBufferId1, kBufferId2, kBufferId3, kBufferId4 };

  void SetUp() override {
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
    const std::string cache_path = testing::TempDir() + "/cache";
    ASSERT_TRUE(cache_provider.StartBuild(cache_path.c_str()));
  }

  FakeContext ctx;
  MMapWeightCacheProvider cache_provider;
};

TEST_F(BuildMMapWeightCacheProviderTest, LookUpFailsIfKeyDoesntMatch) {
  xnn_weights_cache_look_up_key look_up_key{};
  EXPECT_EQ(cache_provider.LookUp(&look_up_key), SIZE_MAX);
}

TEST_F(BuildMMapWeightCacheProviderTest, LookUpSucceeds) {
  enum { kWeightIndex, kBiasIndex };
  const auto pack_id = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                       kAlgoSeed1, kWeightIndex, kBiasIndex);
  const xnn_weights_cache_look_up_key look_up_key =
      ctx.LookUpKey(kAlgoSeed1, kWeightIndex, kBiasIndex);

  EXPECT_EQ(cache_provider.LookUp(&look_up_key),
            ctx.packed_buffers.find(pack_id)->second.offset);
}

TEST_F(BuildMMapWeightCacheProviderTest,
       DifferentAlgoSeedsSameTensorsDontConflict) {
  enum { kWeightIndex, kBiasIndex };
  const auto pack_id_1 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed1, kWeightIndex, kBiasIndex);
  const auto pack_id_2 = ctx.PackTensors(&cache_provider.GetCacheProvider(),
                                         kAlgoSeed2, kWeightIndex, kBiasIndex);

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

TEST_F(BuildMMapWeightCacheProviderTest,
       SameAlgoSeedDifferentTensorsDontConflict) {
  enum { kWeightIndex1, kWeightIndex2, kBiasIndex1, kBiasIndex2 };
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

TEST_F(BuildMMapWeightCacheProviderTest, FinalizeWorks) {
  enum { kWeightIndex1, kBiasIndex, kWeightIndex2 };
  TempFileDesc tmp_file(TempFileDesc::kAutoCLose);
  ASSERT_TRUE(cache_provider.StartBuild(tmp_file.GetCPath()));

  ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1, kWeightIndex1,
                  kBiasIndex);
  ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed2,
                  kWeightIndex2);

  EXPECT_TRUE(cache_provider.IsActive());
  EXPECT_TRUE(cache_provider.IsBuilding());
  ASSERT_TRUE(cache_provider.Finalize());

  ASSERT_TRUE(cache_provider.IsFinalized());
}

struct LoadMMapWeightCacheProviderTest : BuildMMapWeightCacheProviderTest {
  enum { kWeightIndex1, kBiasIndex, kWeightIndex2 };

  void SetUp() override {
    BuildMMapWeightCacheProviderTest::SetUp();
    ASSERT_TRUE(cache_provider.StartBuild(tmp_file.GetCPath()));

    pack_id_1 = ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed1,
                                kWeightIndex1, kBiasIndex);
    pack_id_2 = ctx.PackTensors(&cache_provider.GetCacheProvider(), kAlgoSeed2,
                                kWeightIndex2);

    ASSERT_TRUE(cache_provider.Finalize());
    ASSERT_TRUE(cache_provider.IsFinalized());
  }

  xnn_weights_cache_look_up_key LookUpKey1() const {
    return ctx.LookUpKey(kAlgoSeed1, kWeightIndex1, kBiasIndex);
  }

  xnn_weights_cache_look_up_key LookUpKey2() const {
    return ctx.LookUpKey(kAlgoSeed2, kWeightIndex2);
  }

  TempFileDesc tmp_file;
  PackIdentifier pack_id_1;
  PackIdentifier pack_id_2;
};

TEST_F(LoadMMapWeightCacheProviderTest, LookUpFailsIfKeyDoesntMatch) {
  xnn_weights_cache_look_up_key look_up_key{};
  EXPECT_EQ(cache_provider.LookUp(&look_up_key), SIZE_MAX);
}

template <class T>
class LightSpan {
 public:
  using value_type = T;

  LightSpan(const void* data, const size_t size)
      : ptr_(reinterpret_cast<T*>(data)), size_(size) {}

  size_t size() const { return size(); }
  const T* begin() const { return ptr_; }
  const T* end() const { return ptr_ + size_; }

  friend std::ostream& operator<<(std::ostream& os, const LightSpan<T>& s) {
    os << '[';
    auto it = s.begin();
    if (it != s.end()) {
      os << +*it;
    }
    ++it;
    for (; it != s.end(); ++it) {
      os << ", " << +*it;
    }
    return os << ']';
  }

 private:
  T* ptr_;
  size_t size_;
};

TEST_F(LoadMMapWeightCacheProviderTest, LookUpSucceeds) {
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

TEST(MMapWeightCacheProviderTest, XnnpackCApiJourney) {
  using std::size;
  TempFileDesc temp_fd(TempFileDesc::kAutoCLose);
  const int32_t fake_packing_algo_seed = 0xBA0BAB;
  const char packed_data_ref_1[] = "abcdefghij";
  const char packed_data_ref_2[] = "klmnopqr";
  auto bytes = [](const auto& array) { return size(array) * sizeof(array[0]); };

  constexpr int kBufferCount = 10;
  // We are going to feed dummy packed data. We only need a valid pointer
  // address to map to a buffer identifier.
  char fake_buffer_pointer[kBufferCount] = {0};

  {  // Build and reload scenario.
    TfLiteTensor tensors[kBufferCount];
    std::unordered_map<size_t, size_t> tensor_buffer_identifiers;
    for (int i = 0; i < kBufferCount; ++i) {
      tensors[i].data.data = (void*)(fake_buffer_pointer + i);
      tensor_buffer_identifiers[i] = i;
    }

    MMapWeightCacheProvider cache_provider;
    ASSERT_TRUE(cache_provider.StartBuild(temp_fd.GetCPath()));

    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    cache_provider.MapTensorIdentifiers(tensors, size(tensors),
                                        tensor_buffer_identifiers);

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data};

    // Lookup non-packed tensor.
    ASSERT_EQ(cache->look_up(cache, &look_up_key_1), SIZE_MAX);
    // Reserve space, write data and add packed data.
    void* const reserved_ptr =
        cache->reserve_space(cache, bytes(packed_data_ref_1));
    ASSERT_NE(reserved_ptr, nullptr);
    std::memcpy(reserved_ptr, packed_data_ref_1, bytes(packed_data_ref_1));
    const size_t build_offset_1 = cache->look_up_or_insert(
        cache, &look_up_key_1, reserved_ptr, bytes(packed_data_ref_1));

    // Check that a second insertion with the same key returns the same offset.
    const size_t build_offset_redundant = cache->look_up_or_insert(
        cache, &look_up_key_1, reserved_ptr, bytes(packed_data_ref_1));
    EXPECT_EQ(build_offset_1, build_offset_redundant);

    // Lookup newly packed tensor.
    ASSERT_EQ(cache->look_up(cache, &look_up_key_1), build_offset_1);

    // Add a tensor without reserving before.
    const xnn_weights_cache_look_up_key look_up_key_2{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[2].data.data,
        .bias = tensors[3].data.data};
    const size_t build_offset_2 = cache->look_up_or_insert(
        cache, &look_up_key_2, (void*)packed_data_ref_2,
        bytes(packed_data_ref_2));

    // Save the cache to disk and reload.
    ASSERT_TRUE(cache_provider.Finalize());

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
  }

  {  // Load existing cache scenario.
    TfLiteTensor tensors[kBufferCount];
    std::unordered_map<size_t, size_t> tensor_buffer_identifiers;
    for (int i = 0; i < kBufferCount; ++i) {
      tensors[i].data.data = (void*)(fake_buffer_pointer + i);
      tensor_buffer_identifiers[i] = i;
    }

    MMapWeightCacheProvider cache_provider;
    ASSERT_TRUE(cache_provider.Load(temp_fd.GetCPath()));

    xnn_weights_cache_t cache = &cache_provider.GetCacheProvider();
    cache_provider.MapTensorIdentifiers(tensors, size(tensors),
                                        tensor_buffer_identifiers);

    const xnn_weights_cache_look_up_key look_up_key_1{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[0].data.data,
        .bias = tensors[1].data.data};

    const xnn_weights_cache_look_up_key look_up_key_2{
        .seed = fake_packing_algo_seed,
        .kernel = tensors[2].data.data,
        .bias = tensors[3].data.data};

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
  }
}

}  // namespace
}  // namespace tflite::xnnpack
