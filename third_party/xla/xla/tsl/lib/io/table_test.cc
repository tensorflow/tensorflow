/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/io/table.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/block.h"
#include "xla/tsl/lib/io/block_builder.h"
#include "xla/tsl/lib/io/format.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/lib/random/philox_random.h"
#include "xla/tsl/lib/random/simple_philox.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/snappy.h"

namespace tsl {
namespace table {

namespace {
typedef std::pair<absl::string_view, absl::string_view> StringPiecePair;
}

namespace test {
static absl::string_view RandomString(random::SimplePhilox* rnd, int len,
                                      std::string* dst) {
  dst->resize(len);
  for (int i = 0; i < len; i++) {
    (*dst)[i] = static_cast<char>(' ' + rnd->Uniform(95));  // ' ' .. '~'
  }
  return absl::string_view(*dst);
}
static std::string RandomKey(random::SimplePhilox* rnd, int len) {
  // Make sure to generate a wide variety of characters so we
  // test the boundary conditions for short-key optimizations.
  static const char kTestChars[] = {'\0', '\1', 'a',    'b',    'c',
                                    'd',  'e',  '\xfd', '\xfe', '\xff'};
  std::string result;
  for (int i = 0; i < len; i++) {
    result += kTestChars[rnd->Uniform(sizeof(kTestChars))];
  }
  return result;
}
static absl::string_view CompressibleString(random::SimplePhilox* rnd,
                                            double compressed_fraction,
                                            size_t len, std::string* dst) {
  int raw = static_cast<int>(len * compressed_fraction);
  if (raw < 1) raw = 1;
  std::string raw_data;
  RandomString(rnd, raw, &raw_data);

  // Duplicate the random data until we have filled "len" bytes
  dst->clear();
  while (dst->size() < len) {
    dst->append(raw_data);
  }
  dst->resize(len);
  return absl::string_view(*dst);
}
}  // namespace test

static void Increment(std::string* key) { key->push_back('\0'); }

// An STL comparator that compares two StringPieces
namespace {
struct STLLessThan {
  STLLessThan() {}
  bool operator()(const std::string& a, const std::string& b) const {
    return absl::string_view(a).compare(absl::string_view(b)) < 0;
  }
};
}  // namespace

class StringSink : public WritableFile {
 public:
  ~StringSink() override {}

  const std::string& contents() const { return contents_; }

  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Name(absl::string_view* result) const override {
    return absl::UnimplementedError("StringSink does not support Name()");
  }
  absl::Status Sync() override { return absl::OkStatus(); }
  absl::Status Tell(int64_t* pos) override {
    *pos = contents_.size();
    return absl::OkStatus();
  }

  absl::Status Append(absl::string_view data) override {
    contents_.append(data.data(), data.size());
    return absl::OkStatus();
  }

 private:
  std::string contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(absl::string_view contents)
      : contents_(contents.data(), contents.size()), bytes_read_(0) {}

  ~StringSource() override {}

  uint64_t Size() const { return contents_.size(); }

  absl::Status Name(absl::string_view* result) const override {
    return absl::UnimplementedError("StringSource does not support Name()");
  }

  absl::Status Read(uint64_t offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    if (offset > contents_.size()) {
      return absl::InvalidArgumentError("invalid Read offset");
    }
    if (offset + n > contents_.size()) {
      n = contents_.size() - offset;
    }
    memcpy(scratch, &contents_[offset], n);
    *result = absl::string_view(scratch, n);
    bytes_read_ += n;
    return absl::OkStatus();
  }

  uint64_t BytesRead() const { return bytes_read_; }

 private:
  std::string contents_;
  mutable uint64_t bytes_read_;
};

typedef std::map<std::string, std::string, STLLessThan> KVMap;

// Helper class for tests to unify the interface between
// BlockBuilder/TableBuilder and Block/Table.
class Constructor {
 public:
  explicit Constructor() : data_(STLLessThan()) {}
  virtual ~Constructor() {}

  void Add(const std::string& key, absl::string_view value) {
    data_[key] = std::string(value);
  }

  // Finish constructing the data structure with all the keys that have
  // been added so far.  Returns the keys in sorted order in "*keys"
  // and stores the key/value pairs in "*kvmap"
  void Finish(const Options& options, std::vector<std::string>* keys,
              KVMap* kvmap) {
    *kvmap = data_;
    keys->clear();
    for (KVMap::const_iterator it = data_.begin(); it != data_.end(); ++it) {
      keys->push_back(it->first);
    }
    data_.clear();
    absl::Status s = FinishImpl(options, *kvmap);
    ASSERT_TRUE(s.ok()) << s;
  }

  // Construct the data structure from the data in "data"
  virtual absl::Status FinishImpl(const Options& options,
                                  const KVMap& data) = 0;

  virtual Iterator* NewIterator() const = 0;

  virtual const KVMap& data() { return data_; }

 private:
  KVMap data_;
};

class BlockConstructor : public Constructor {
 public:
  BlockConstructor() : block_(nullptr) {}
  ~BlockConstructor() override { delete block_; }
  absl::Status FinishImpl(const Options& options, const KVMap& data) override {
    delete block_;
    block_ = nullptr;
    BlockBuilder builder(&options);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
    }
    // Open the block
    data_ = std::string(builder.Finish());
    BlockContents contents;
    contents.data = data_;
    contents.cacheable = false;
    contents.heap_allocated = false;
    block_ = new Block(contents);
    return absl::OkStatus();
  }
  Iterator* NewIterator() const override { return block_->NewIterator(); }

 private:
  std::string data_;
  Block* block_;
};

class TableConstructor : public Constructor {
 public:
  TableConstructor() : source_(nullptr), table_(nullptr) {}
  ~TableConstructor() override { Reset(); }
  absl::Status FinishImpl(const Options& options, const KVMap& data) override {
    Reset();
    StringSink sink;
    TableBuilder builder(options, &sink);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
      CHECK_OK(builder.status());
    }
    absl::Status s = builder.Finish();
    CHECK_OK(s) << s;

    CHECK_EQ(sink.contents().size(), builder.FileSize());

    // Open the table
    source_ = new StringSource(sink.contents());
    Options table_options;
    return Table::Open(table_options, source_, sink.contents().size(), &table_);
  }

  Iterator* NewIterator() const override { return table_->NewIterator(); }

  uint64_t ApproximateOffsetOf(absl::string_view key) const {
    return table_->ApproximateOffsetOf(key);
  }

  uint64_t BytesRead() const { return source_->BytesRead(); }

 private:
  void Reset() {
    delete table_;
    delete source_;
    table_ = nullptr;
    source_ = nullptr;
  }

  StringSource* source_;
  Table* table_;
};

enum TestType { TABLE_TEST, BLOCK_TEST };

struct TestArgs {
  TestType type;
  int restart_interval;
};

static const TestArgs kTestArgList[] = {
    {TABLE_TEST, 16}, {TABLE_TEST, 1}, {TABLE_TEST, 1024},
    {BLOCK_TEST, 16}, {BLOCK_TEST, 1}, {BLOCK_TEST, 1024},
};
static const int kNumTestArgs = sizeof(kTestArgList) / sizeof(kTestArgList[0]);

class Harness : public ::testing::Test {
 public:
  Harness() : constructor_(nullptr) {}

  void Init(const TestArgs& args) {
    delete constructor_;
    constructor_ = nullptr;
    options_ = Options();

    options_.block_restart_interval = args.restart_interval;
    // Use shorter block size for tests to exercise block boundary
    // conditions more.
    options_.block_size = 256;
    switch (args.type) {
      case TABLE_TEST:
        constructor_ = new TableConstructor();
        break;
      case BLOCK_TEST:
        constructor_ = new BlockConstructor();
        break;
    }
  }

  ~Harness() override { delete constructor_; }

  void Add(const std::string& key, const std::string& value) {
    constructor_->Add(key, value);
  }

  void Test(random::SimplePhilox* rnd, int num_random_access_iters = 200) {
    std::vector<std::string> keys;
    KVMap data;
    constructor_->Finish(options_, &keys, &data);

    TestForwardScan(keys, data);
    TestRandomAccess(rnd, keys, data, num_random_access_iters);
  }

  void TestForwardScan(const std::vector<std::string>& keys,
                       const KVMap& data) {
    Iterator* iter = constructor_->NewIterator();
    ASSERT_TRUE(!iter->Valid());
    iter->SeekToFirst();
    for (KVMap::const_iterator model_iter = data.begin();
         model_iter != data.end(); ++model_iter) {
      ASSERT_EQ(ToStringPiecePair(data, model_iter), ToStringPiecePair(iter));
      iter->Next();
    }
    ASSERT_TRUE(!iter->Valid());
    delete iter;
  }

  void TestRandomAccess(random::SimplePhilox* rnd,
                        const std::vector<std::string>& keys, const KVMap& data,
                        int num_random_access_iters) {
    static const bool kVerbose = false;
    Iterator* iter = constructor_->NewIterator();
    ASSERT_TRUE(!iter->Valid());
    KVMap::const_iterator model_iter = data.begin();
    if (kVerbose) fprintf(stderr, "---\n");
    for (int i = 0; i < num_random_access_iters; i++) {
      const int toss = rnd->Uniform(3);
      switch (toss) {
        case 0: {
          if (iter->Valid()) {
            if (kVerbose) fprintf(stderr, "Next\n");
            iter->Next();
            ++model_iter;
            ASSERT_EQ(ToStringPiecePair(data, model_iter),
                      ToStringPiecePair(iter));
          }
          break;
        }

        case 1: {
          if (kVerbose) fprintf(stderr, "SeekToFirst\n");
          iter->SeekToFirst();
          model_iter = data.begin();
          ASSERT_EQ(ToStringPiecePair(data, model_iter),
                    ToStringPiecePair(iter));
          break;
        }

        case 2: {
          std::string key = PickRandomKey(rnd, keys);
          model_iter = data.lower_bound(key);
          if (kVerbose)
            fprintf(stderr, "Seek '%s'\n", absl::CEscape(key).c_str());
          iter->Seek(absl::string_view(key));
          ASSERT_EQ(ToStringPiecePair(data, model_iter),
                    ToStringPiecePair(iter));
          break;
        }
      }
    }
    delete iter;
  }

  StringPiecePair ToStringPiecePair(const KVMap& data,
                                    const KVMap::const_iterator& it) {
    if (it == data.end()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->first, it->second);
    }
  }

  StringPiecePair ToStringPiecePair(const KVMap& data,
                                    const KVMap::const_reverse_iterator& it) {
    if (it == data.rend()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->first, it->second);
    }
  }

  StringPiecePair ToStringPiecePair(const Iterator* it) {
    if (!it->Valid()) {
      return StringPiecePair("END", "");
    } else {
      return StringPiecePair(it->key(), it->value());
    }
  }

  std::string PickRandomKey(random::SimplePhilox* rnd,
                            const std::vector<std::string>& keys) {
    if (keys.empty()) {
      return "foo";
    } else {
      const int index = rnd->Uniform(keys.size());
      std::string result = keys[index];
      switch (rnd->Uniform(3)) {
        case 0:
          // Return an existing key
          break;
        case 1: {
          // Attempt to return something smaller than an existing key
          if (!result.empty() && result[result.size() - 1] > '\0') {
            result[result.size() - 1]--;
          }
          break;
        }
        case 2: {
          // Return something larger than an existing key
          Increment(&result);
          break;
        }
      }
      return result;
    }
  }

 private:
  Options options_;
  Constructor* constructor_;
};

// Test empty table/block.
TEST_F(Harness, Empty) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 1, 17);
    random::SimplePhilox rnd(&philox);
    Test(&rnd);
  }
}

// Special test for a block with no restart entries.  The C++ leveldb
// code never generates such blocks, but the Java version of leveldb
// seems to.
TEST_F(Harness, ZeroRestartPointsInBlock) {
  char data[sizeof(uint32_t)];
  memset(data, 0, sizeof(data));
  BlockContents contents;
  contents.data = absl::string_view(data, sizeof(data));
  contents.cacheable = false;
  contents.heap_allocated = false;
  Block block(contents);
  Iterator* iter = block.NewIterator();
  iter->SeekToFirst();
  ASSERT_TRUE(!iter->Valid());
  iter->Seek("foo");
  ASSERT_TRUE(!iter->Valid());
  delete iter;
}

// Test the empty key
TEST_F(Harness, SimpleEmptyKey) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 1, 17);
    random::SimplePhilox rnd(&philox);
    Add("", "v");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleSingle) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 2, 17);
    random::SimplePhilox rnd(&philox);
    Add("abc", "v");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleMulti) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 3, 17);
    random::SimplePhilox rnd(&philox);
    Add("abc", "v");
    Add("abcd", "v");
    Add("ac", "v2");
    Test(&rnd);
  }
}

TEST_F(Harness, SimpleMultiBigValues) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 3, 17);
    random::SimplePhilox rnd(&philox);
    Add("ainitial", "tiny");
    Add("anext", std::string(10000000, 'a'));
    Add("anext2", std::string(10000000, 'b'));
    Add("azz", "tiny");
    Test(&rnd, 100 /* num_random_access_iters */);
  }
}

TEST_F(Harness, SimpleSpecialKey) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 4, 17);
    random::SimplePhilox rnd(&philox);
    Add("\xff\xff", "v3");
    Test(&rnd);
  }
}

TEST_F(Harness, Randomized) {
  for (int i = 0; i < kNumTestArgs; i++) {
    Init(kTestArgList[i]);
    random::PhiloxRandom philox(testing::RandomSeed() + 5, 17);
    random::SimplePhilox rnd(&philox);
    for (int num_entries = 0; num_entries < 2000;
         num_entries += (num_entries < 50 ? 1 : 200)) {
      if ((num_entries % 10) == 0) {
        fprintf(stderr, "case %d of %d: num_entries = %d\n", (i + 1),
                int(kNumTestArgs), num_entries);
      }
      for (int e = 0; e < num_entries; e++) {
        std::string v;
        Add(test::RandomKey(&rnd, rnd.Skewed(4)),
            std::string(test::RandomString(&rnd, rnd.Skewed(5), &v)));
      }
      Test(&rnd);
    }
  }
}

static bool Between(uint64_t val, uint64_t low, uint64_t high) {
  bool result = (val >= low) && (val <= high);
  if (!result) {
    fprintf(stderr, "Value %llu is not in range [%llu, %llu]\n",
            static_cast<unsigned long long>(val),
            static_cast<unsigned long long>(low),
            static_cast<unsigned long long>(high));
  }
  return result;
}

class TableTest {};

TEST(TableTest, ApproximateOffsetOfPlain) {
  TableConstructor c;
  c.Add("k01", "hello");
  c.Add("k02", "hello2");
  c.Add("k03", std::string(10000, 'x'));
  c.Add("k04", std::string(200000, 'x'));
  c.Add("k05", std::string(300000, 'x'));
  c.Add("k06", "hello3");
  c.Add("k07", std::string(100000, 'x'));
  std::vector<std::string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kNoCompression;
  c.Finish(options, &keys, &kvmap);

  ASSERT_TRUE(Between(c.ApproximateOffsetOf("abc"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01a"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k02"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k03"), 10, 500));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04"), 10000, 11000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04a"), 210000, 211000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k05"), 210000, 211000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k06"), 510000, 511000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k07"), 510000, 511000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("xyz"), 610000, 612000));
}

static bool SnappyCompressionSupported() {
  std::string out;
  absl::string_view in = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  return port::Snappy_Compress(in.data(), in.size(), &out);
}

TEST(TableTest, ApproximateOffsetOfCompressed) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  TableConstructor c;
  std::string tmp;
  c.Add("k01", "hello");
  c.Add("k02", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  c.Add("k03", "hello3");
  c.Add("k04", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  std::vector<std::string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kSnappyCompression;
  c.Finish(options, &keys, &kvmap);
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("abc"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k01"), 0, 0));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k02"), 10, 100));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k03"), 2000, 4000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("k04"), 2000, 4000));
  ASSERT_TRUE(Between(c.ApproximateOffsetOf("xyz"), 4000, 7000));
}

// A RandomAccessFile over a string that records the largest read requested.
// ReadBlock allocates a buffer of the requested size before reading, so this
// is also the largest block buffer allocated.
class LargestReadStringSource : public RandomAccessFile {
 public:
  explicit LargestReadStringSource(absl::string_view contents)
      : contents_(contents.data(), contents.size()), largest_read_(0) {}

  uint64_t Size() const { return contents_.size(); }
  uint64_t LargestRead() const { return largest_read_; }

  absl::Status Name(absl::string_view* result) const override {
    return absl::UnimplementedError(
        "LargestReadStringSource does not support Name()");
  }

  absl::Status Read(uint64_t offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    if (n > largest_read_) largest_read_ = n;
    if (offset > contents_.size()) {
      return absl::InvalidArgumentError("invalid Read offset");
    }
    if (n > contents_.size() - offset) n = contents_.size() - offset;
    memcpy(scratch, &contents_[offset], n);
    *result = absl::string_view(scratch, n);
    return absl::OkStatus();
  }

 private:
  std::string contents_;
  mutable uint64_t largest_read_;
};

static std::string EncodeFooter(uint64_t index_offset, uint64_t index_size) {
  BlockHandle metaindex_handle;
  metaindex_handle.set_offset(0);
  metaindex_handle.set_size(0);
  BlockHandle index_handle;
  index_handle.set_offset(index_offset);
  index_handle.set_size(index_size);
  Footer footer;
  footer.set_metaindex_handle(metaindex_handle);
  footer.set_index_handle(index_handle);
  std::string encoded;
  footer.EncodeTo(&encoded);
  return encoded;
}

TEST(TableTest, IndexBlockHandlePastEndOfFileIsRejected) {
  const std::string padding(100, 'x');
  const uint64_t file_size = padding.size() + Footer::kEncodedLength;
  const struct {
    uint64_t offset;
    uint64_t size;
  } kHandles[] = {
      {0, uint64_t{1} << 20},                  // Block larger than the file.
      {0, file_size - kBlockTrailerSize + 1},  // Trailer ends past the file.
      {file_size + 1, 0},                      // Offset past the file.
      {~uint64_t{0} - 1, 2},                   // Offset plus size wraps.
  };
  for (const auto& h : kHandles) {
    LargestReadStringSource source(padding + EncodeFooter(h.offset, h.size));
    ASSERT_EQ(source.Size(), file_size);
    Table* table = nullptr;
    absl::Status s = Table::Open(Options(), &source, source.Size(), &table);
    EXPECT_TRUE(absl::IsDataLoss(s)) << s;
    EXPECT_EQ(table, nullptr);
    EXPECT_LE(source.LargestRead(), source.Size());
    delete table;
  }
}

TEST(TableTest, DataBlockHandlePastEndOfFileIsRejected) {
  // Build a table with one data block of about 1 MiB, then keep only its index
  // block and a new footer, so the index entry points past the end of the file.
  StringSink sink;
  Options options;
  options.compression = kNoCompression;
  TableBuilder builder(options, &sink);
  builder.Add("k01", std::string(1 << 20, 'x'));
  ASSERT_TRUE(builder.Finish().ok());
  const std::string& table_contents = sink.contents();
  absl::string_view footer_input =
      absl::string_view(table_contents)
          .substr(table_contents.size() - Footer::kEncodedLength,
                  Footer::kEncodedLength);
  Footer footer;
  ASSERT_TRUE(footer.DecodeFrom(&footer_input).ok());
  const BlockHandle& index_handle = footer.index_handle();
  LargestReadStringSource source(
      table_contents.substr(index_handle.offset(),
                            index_handle.size() + kBlockTrailerSize) +
      EncodeFooter(0, index_handle.size()));

  Table* table = nullptr;
  ASSERT_TRUE(Table::Open(options, &source, source.Size(), &table).ok());
  Iterator* iter = table->NewIterator();
  iter->SeekToFirst();
  EXPECT_FALSE(iter->Valid());
  EXPECT_TRUE(absl::IsDataLoss(iter->status())) << iter->status();
  EXPECT_LE(source.LargestRead(), source.Size());
  delete iter;
  delete table;
}

TEST(TableTest, SeekToFirstKeyDoesNotReadTooMuch) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::string tmp;
  TableConstructor c;
  c.Add("k01", "firstvalue");
  c.Add("k03", test::CompressibleString(&rnd, 0.25, 1000000, &tmp));
  c.Add("k04", "abc");
  std::vector<std::string> keys;
  KVMap kvmap;
  Options options;
  options.block_size = 1024;
  options.compression = kNoCompression;
  c.Finish(options, &keys, &kvmap);

  Iterator* iter = c.NewIterator();
  iter->Seek("k01");
  delete iter;
  // Make sure we don't read the big second block when just trying to
  // retrieve the data in the first key
  EXPECT_LT(c.BytesRead(), 200);
}

}  // namespace table
}  // namespace tsl
