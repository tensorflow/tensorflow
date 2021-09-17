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

#include "tensorflow/core/lib/io/table.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/block.h"
#include "tensorflow/core/lib/io/block_builder.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace table {

namespace {
typedef std::pair<StringPiece, StringPiece> StringPiecePair;
}

namespace test {
static StringPiece RandomString(random::SimplePhilox* rnd, int len,
                                string* dst) {
  dst->resize(len);
  for (int i = 0; i < len; i++) {
    (*dst)[i] = static_cast<char>(' ' + rnd->Uniform(95));  // ' ' .. '~'
  }
  return StringPiece(*dst);
}
static string RandomKey(random::SimplePhilox* rnd, int len) {
  // Make sure to generate a wide variety of characters so we
  // test the boundary conditions for short-key optimizations.
  static const char kTestChars[] = {'\0', '\1', 'a',    'b',    'c',
                                    'd',  'e',  '\xfd', '\xfe', '\xff'};
  string result;
  for (int i = 0; i < len; i++) {
    result += kTestChars[rnd->Uniform(sizeof(kTestChars))];
  }
  return result;
}
static StringPiece CompressibleString(random::SimplePhilox* rnd,
                                      double compressed_fraction, size_t len,
                                      string* dst) {
  int raw = static_cast<int>(len * compressed_fraction);
  if (raw < 1) raw = 1;
  string raw_data;
  RandomString(rnd, raw, &raw_data);

  // Duplicate the random data until we have filled "len" bytes
  dst->clear();
  while (dst->size() < len) {
    dst->append(raw_data);
  }
  dst->resize(len);
  return StringPiece(*dst);
}
}  // namespace test

static void Increment(string* key) { key->push_back('\0'); }

// An STL comparator that compares two StringPieces
namespace {
struct STLLessThan {
  STLLessThan() {}
  bool operator()(const string& a, const string& b) const {
    return StringPiece(a).compare(StringPiece(b)) < 0;
  }
};
}  // namespace

class StringSink : public WritableFile {
 public:
  ~StringSink() override {}

  const string& contents() const { return contents_; }

  Status Close() override { return Status::OK(); }
  Status Flush() override { return Status::OK(); }
  Status Name(StringPiece* result) const override {
    return errors::Unimplemented("StringSink does not support Name()");
  }
  Status Sync() override { return Status::OK(); }
  Status Tell(int64_t* pos) override {
    *pos = contents_.size();
    return Status::OK();
  }

  Status Append(StringPiece data) override {
    contents_.append(data.data(), data.size());
    return Status::OK();
  }

 private:
  string contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(const StringPiece& contents)
      : contents_(contents.data(), contents.size()), bytes_read_(0) {}

  ~StringSource() override {}

  uint64 Size() const { return contents_.size(); }

  Status Name(StringPiece* result) const override {
    return errors::Unimplemented("StringSource does not support Name()");
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    if (offset > contents_.size()) {
      return errors::InvalidArgument("invalid Read offset");
    }
    if (offset + n > contents_.size()) {
      n = contents_.size() - offset;
    }
    memcpy(scratch, &contents_[offset], n);
    *result = StringPiece(scratch, n);
    bytes_read_ += n;
    return Status::OK();
  }

  uint64 BytesRead() const { return bytes_read_; }

 private:
  string contents_;
  mutable uint64 bytes_read_;
};

typedef std::map<string, string, STLLessThan> KVMap;

// Helper class for tests to unify the interface between
// BlockBuilder/TableBuilder and Block/Table.
class Constructor {
 public:
  explicit Constructor() : data_(STLLessThan()) {}
  virtual ~Constructor() {}

  void Add(const string& key, const StringPiece& value) {
    data_[key] = string(value);
  }

  // Finish constructing the data structure with all the keys that have
  // been added so far.  Returns the keys in sorted order in "*keys"
  // and stores the key/value pairs in "*kvmap"
  void Finish(const Options& options, std::vector<string>* keys, KVMap* kvmap) {
    *kvmap = data_;
    keys->clear();
    for (KVMap::const_iterator it = data_.begin(); it != data_.end(); ++it) {
      keys->push_back(it->first);
    }
    data_.clear();
    Status s = FinishImpl(options, *kvmap);
    ASSERT_TRUE(s.ok()) << s.ToString();
  }

  // Construct the data structure from the data in "data"
  virtual Status FinishImpl(const Options& options, const KVMap& data) = 0;

  virtual Iterator* NewIterator() const = 0;

  virtual const KVMap& data() { return data_; }

 private:
  KVMap data_;
};

class BlockConstructor : public Constructor {
 public:
  BlockConstructor() : block_(nullptr) {}
  ~BlockConstructor() override { delete block_; }
  Status FinishImpl(const Options& options, const KVMap& data) override {
    delete block_;
    block_ = nullptr;
    BlockBuilder builder(&options);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
    }
    // Open the block
    data_ = string(builder.Finish());
    BlockContents contents;
    contents.data = data_;
    contents.cacheable = false;
    contents.heap_allocated = false;
    block_ = new Block(contents);
    return Status::OK();
  }
  Iterator* NewIterator() const override { return block_->NewIterator(); }

 private:
  string data_;
  Block* block_;
};

class TableConstructor : public Constructor {
 public:
  TableConstructor() : source_(nullptr), table_(nullptr) {}
  ~TableConstructor() override { Reset(); }
  Status FinishImpl(const Options& options, const KVMap& data) override {
    Reset();
    StringSink sink;
    TableBuilder builder(options, &sink);

    for (KVMap::const_iterator it = data.begin(); it != data.end(); ++it) {
      builder.Add(it->first, it->second);
      TF_CHECK_OK(builder.status());
    }
    Status s = builder.Finish();
    TF_CHECK_OK(s) << s.ToString();

    CHECK_EQ(sink.contents().size(), builder.FileSize());

    // Open the table
    source_ = new StringSource(sink.contents());
    Options table_options;
    return Table::Open(table_options, source_, sink.contents().size(), &table_);
  }

  Iterator* NewIterator() const override { return table_->NewIterator(); }

  uint64 ApproximateOffsetOf(const StringPiece& key) const {
    return table_->ApproximateOffsetOf(key);
  }

  uint64 BytesRead() const { return source_->BytesRead(); }

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

  void Add(const string& key, const string& value) {
    constructor_->Add(key, value);
  }

  void Test(random::SimplePhilox* rnd, int num_random_access_iters = 200) {
    std::vector<string> keys;
    KVMap data;
    constructor_->Finish(options_, &keys, &data);

    TestForwardScan(keys, data);
    TestRandomAccess(rnd, keys, data, num_random_access_iters);
  }

  void TestForwardScan(const std::vector<string>& keys, const KVMap& data) {
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
                        const std::vector<string>& keys, const KVMap& data,
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
          string key = PickRandomKey(rnd, keys);
          model_iter = data.lower_bound(key);
          if (kVerbose)
            fprintf(stderr, "Seek '%s'\n", absl::CEscape(key).c_str());
          iter->Seek(StringPiece(key));
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

  string PickRandomKey(random::SimplePhilox* rnd,
                       const std::vector<string>& keys) {
    if (keys.empty()) {
      return "foo";
    } else {
      const int index = rnd->Uniform(keys.size());
      string result = keys[index];
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
  char data[sizeof(uint32)];
  memset(data, 0, sizeof(data));
  BlockContents contents;
  contents.data = StringPiece(data, sizeof(data));
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
    Add("anext", string(10000000, 'a'));
    Add("anext2", string(10000000, 'b'));
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
        string v;
        Add(test::RandomKey(&rnd, rnd.Skewed(4)),
            string(test::RandomString(&rnd, rnd.Skewed(5), &v)));
      }
      Test(&rnd);
    }
  }
}

static bool Between(uint64 val, uint64 low, uint64 high) {
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
  c.Add("k03", string(10000, 'x'));
  c.Add("k04", string(200000, 'x'));
  c.Add("k05", string(300000, 'x'));
  c.Add("k06", "hello3");
  c.Add("k07", string(100000, 'x'));
  std::vector<string> keys;
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
  string out;
  StringPiece in = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
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
  string tmp;
  c.Add("k01", "hello");
  c.Add("k02", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  c.Add("k03", "hello3");
  c.Add("k04", test::CompressibleString(&rnd, 0.25, 10000, &tmp));
  std::vector<string> keys;
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

TEST(TableTest, SeekToFirstKeyDoesNotReadTooMuch) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  string tmp;
  TableConstructor c;
  c.Add("k01", "firstvalue");
  c.Add("k03", test::CompressibleString(&rnd, 0.25, 1000000, &tmp));
  c.Add("k04", "abc");
  std::vector<string> keys;
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
}  // namespace tensorflow
