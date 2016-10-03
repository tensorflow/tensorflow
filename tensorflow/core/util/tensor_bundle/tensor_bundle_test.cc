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

#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#include <random>
#include <vector>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

template <typename T>
Tensor Constant(T v, TensorShape shape) {
  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

template <typename T>
Tensor Constant_2x3(T v) {
  return Constant(v, TensorShape({2, 3}));
}

template <typename T>
void Expect(BundleReader* reader, const string& key,
            const Tensor& expected_val) {
  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupTensorShape().
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupTensorShape(key, &shape));
  EXPECT_EQ(expected_val.shape(), shape);
  // Tests for Lookup(), checking tensor contents.
  Tensor val(expected_val.dtype(), shape);
  TF_ASSERT_OK(reader->Lookup(key, &val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    ret.push_back(reader->key().ToString());
  }
  return ret;
}

// Writes out the metadata file of a bundle again, with the endianness marker
// bit flipped.
Status FlipEndiannessBit(const string& prefix) {
  Env* env = Env::Default();
  const string metadata_tmp_path = Prefix("some_tmp_path");
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(metadata_tmp_path, &file));
  table::TableBuilder builder(table::Options(), file.get());

  // Reads the existing metadata file, and fills the builder.
  {
    const string filename = MetaFilename(prefix);
    uint64 file_size;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    table::Table* table = nullptr;
    TF_RETURN_IF_ERROR(
        table::Table::Open(table::Options(), file.get(), file_size, &table));
    std::unique_ptr<table::Table> table_deleter(table);
    std::unique_ptr<table::Iterator> iter(table->NewIterator());

    // Reads the header entry.
    iter->Seek(kHeaderEntryKey);
    CHECK(iter->Valid());
    BundleHeaderProto header;
    CHECK(header.ParseFromArray(iter->value().data(), iter->value().size()));
    // Flips the endianness.
    if (header.endianness() == BundleHeaderProto::LITTLE) {
      header.set_endianness(BundleHeaderProto::BIG);
    } else {
      header.set_endianness(BundleHeaderProto::LITTLE);
    }
    builder.Add(iter->key(), header.SerializeAsString());
    iter->Next();

    // Adds the non-header entries unmodified.
    for (; iter->Valid(); iter->Next()) builder.Add(iter->key(), iter->value());
  }
  TF_RETURN_IF_ERROR(builder.Finish());
  TF_RETURN_IF_ERROR(env->RenameFile(metadata_tmp_path, MetaFilename(prefix)));
  return file->Close();
}

template <typename T>
void TestBasic() {
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    writer.Add("foo_003", Constant_2x3<T>(3));
    writer.Add("foo_000", Constant_2x3<T>(0));
    writer.Add("foo_002", Constant_2x3<T>(2));
    writer.Add("foo_001", Constant_2x3<T>(1));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(0));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(1));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(2));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(3));
  }
  {
    BundleWriter writer(Env::Default(), Prefix("bar"));
    writer.Add("bar_003", Constant_2x3<T>(3));
    writer.Add("bar_000", Constant_2x3<T>(0));
    writer.Add("bar_002", Constant_2x3<T>(2));
    writer.Add("bar_001", Constant_2x3<T>(1));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003"}));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(3));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(2));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(1));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(0));
  }
  TF_ASSERT_OK(MergeBundles(Env::Default(), {Prefix("foo"), Prefix("bar")},
                            Prefix("merged")));
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003",
                             "foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(0));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(1));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(2));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(3));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(0));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(1));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(2));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(3));
  }
}

template <typename T>
void TestNonStandardShapes() {
  {
    BundleWriter writer(Env::Default(), Prefix("nonstandard"));
    writer.Add("scalar", Constant<T>(0, TensorShape()));
    writer.Add("non_standard0", Constant<T>(0, TensorShape({0, 1618})));
    writer.Add("non_standard1", Constant<T>(0, TensorShape({16, 0, 18})));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("nonstandard"));
    TF_ASSERT_OK(reader.status());
    Expect<T>(&reader, "scalar", Constant<T>(0, TensorShape()));
    Expect<T>(&reader, "non_standard0", Constant<T>(0, TensorShape({0, 1618})));
    Expect<T>(&reader, "non_standard1",
              Constant<T>(0, TensorShape({16, 0, 18})));
  }
}

// Writes a bundle to disk with a bad "version"; checks for "expected_error".
void VersionTest(const VersionDef& version, StringPiece expected_error) {
  const string path = Prefix("version_test");
  {
    // Prepare an empty bundle with the given version information.
    BundleHeaderProto header;
    *header.mutable_version() = version;

    // Write the metadata file to disk.
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewWritableFile(MetaFilename(path), &file));
    table::TableBuilder builder(table::Options(), file.get());
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    TF_ASSERT_OK(builder.Finish());
  }
  // Read it back in and verify that we get the expected error.
  BundleReader reader(Env::Default(), path);
  EXPECT_TRUE(errors::IsInvalidArgument(reader.status()));
  EXPECT_TRUE(
      StringPiece(reader.status().error_message()).starts_with(expected_error));
}

}  // namespace

TEST(TensorBundleTest, Basic) {
  TestBasic<float>();
  TestBasic<double>();
  TestBasic<int32>();
  TestBasic<uint8>();
  TestBasic<int16>();
  TestBasic<int8>();
  TestBasic<complex64>();
  TestBasic<complex128>();
  TestBasic<int64>();
  TestBasic<bool>();
  TestBasic<qint32>();
  TestBasic<quint8>();
  TestBasic<qint8>();
}

TEST(TensorBundleTest, PartitionedVariables) {
  const TensorShape kFullShape({5, 10});
  // Adds two slices.
  // First slice: column 0, all zeros.
  // Second slice: column 1 to rest, all ones.
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TensorSlice slice = TensorSlice::ParseOrDie("-:0,1");

    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape,
                                 TensorSlice::ParseOrDie("-:0,1"),
                                 Constant<float>(0., TensorShape({5, 1}))));
    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape,
                                 TensorSlice::ParseOrDie("-:1,9"),
                                 Constant<float>(1., TensorShape({5, 9}))));
    TF_ASSERT_OK(writer.Finish());
  }
  // Reads in full.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    Tensor expected_val(DT_FLOAT, kFullShape);
    test::FillFn<float>(&expected_val, [](int offset) -> float {
      if (offset % 10 == 0) {
        return 0;  // First column zeros.
      }
      return 1;  // Other columns ones.
    });

    Tensor val(DT_FLOAT, kFullShape);
    TF_ASSERT_OK(reader.Lookup("foo", &val));
    test::ExpectTensorEqual<float>(val, expected_val);
  }
  // Reads a slice consisting of first two columns, "cutting" both slices.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    // First two columns, "cutting" both slices.
    const TensorSlice distinct_slice = TensorSlice::ParseOrDie("-:0,2");
    Tensor expected_val(DT_FLOAT, TensorShape({5, 2}));
    test::FillFn<float>(&expected_val, [](int offset) -> float {
      if (offset % 2 == 0) {
        return 0;  // First column zeros.
      }
      return 1;  // Other columns ones.
    });

    Tensor val(DT_FLOAT, TensorShape({5, 2}));
    TF_ASSERT_OK(reader.LookupSlice("foo", distinct_slice, &val));
    test::ExpectTensorEqual<float>(val, expected_val);
  }
  // Reads a slice consisting of columns 2-4, "cutting" the second slice only.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    const TensorSlice distinct_slice = TensorSlice::ParseOrDie("-:2,2");
    Tensor val(DT_FLOAT, TensorShape({5, 2}));
    TF_ASSERT_OK(reader.LookupSlice("foo", distinct_slice, &val));
    test::ExpectTensorEqual<float>(val,
                                   Constant<float>(1., TensorShape({5, 2})));
  }
}

TEST(TensorBundleTest, NonStandardShapes) {
  TestNonStandardShapes<float>();
  TestNonStandardShapes<double>();
  TestNonStandardShapes<int32>();
  TestNonStandardShapes<uint8>();
  TestNonStandardShapes<int16>();
  TestNonStandardShapes<int8>();
  TestNonStandardShapes<complex64>();
  TestNonStandardShapes<complex128>();
  TestNonStandardShapes<int64>();
  TestNonStandardShapes<bool>();
  TestNonStandardShapes<qint32>();
  TestNonStandardShapes<quint8>();
  TestNonStandardShapes<qint8>();
}

TEST(TensorBundleTest, StringTensors) {
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    writer.Add("string_tensor", Tensor(DT_STRING, TensorShape({1})));  // Empty.
    writer.Add("scalar", test::AsTensor<string>({"hello"}));
    writer.Add("strs", test::AsTensor<string>(
                           {"hello", "", "x01", string(1 << 25, 'c')}));
    // Mixes in some floats.
    writer.Add("floats", Constant_2x3<float>(16.18));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"floats", "scalar", "string_tensor", "strs"}));

    Expect<string>(&reader, "string_tensor",
                   Tensor(DT_STRING, TensorShape({1})));
    Expect<string>(&reader, "scalar", test::AsTensor<string>({"hello"}));
    Expect<string>(
        &reader, "strs",
        test::AsTensor<string>({"hello", "", "x01", string(1 << 25, 'c')}));
    Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));
  }
}

TEST(TensorBundleTest, DirectoryStructure) {
  Env* env = Env::Default();
  // Writes two bundles.
  const std::vector<string> kBundlePrefixes = {Prefix("worker0"),
                                               Prefix("worker1")};
  for (int i = 0; i < 2; ++i) {
    BundleWriter writer(env, kBundlePrefixes[i]);
    writer.Add(strings::StrCat("tensor", i), Constant_2x3<float>(0.));
    TF_ASSERT_OK(writer.Finish());
  }

  // Ensures we have the expected files.
  auto CheckDirFiles = [env](const string& bundle_prefix,
                             gtl::ArraySlice<string> expected_files) {
    StringPiece dir = io::Dirname(bundle_prefix);
    for (const string& expected_file : expected_files) {
      EXPECT_TRUE(env->FileExists(io::JoinPath(dir, expected_file)));
    }
  };

  // Check we have:
  //   worker<i>.index
  //   worker<i>.data-00000-of-00001
  CheckDirFiles(kBundlePrefixes[0],
                {"worker0.index", "worker0.data-00000-of-00001"});
  CheckDirFiles(kBundlePrefixes[1],
                {"worker1.index", "worker1.data-00000-of-00001"});

  // Trivially "merge" one bundle to some other location (i.e., a renaming).
  const string kAnotherPrefix = Prefix("another");
  TF_ASSERT_OK(MergeBundles(env, {kBundlePrefixes[0]}, kAnotherPrefix));
  CheckDirFiles(kAnotherPrefix,
                {"another.index", "another.data-00000-of-00001"});

  // Performs actual merge of the two bundles.  Check we have:
  //   merged.index
  //   merged.data-00000-of-00002
  //   merged.data-00001-of-00002
  const string kMerged = Prefix("merged");
  TF_ASSERT_OK(
      MergeBundles(env, {kAnotherPrefix, kBundlePrefixes[1]}, kMerged));
  CheckDirFiles(kMerged, {"merged.index", "merged.data-00000-of-00002",
                          "merged.data-00001-of-00002"});
}

TEST(TensorBundleTest, Error) {
  {  // Dup keys.
    BundleWriter writer(Env::Default(), Prefix("dup"));
    writer.Add("foo", Constant_2x3(1.f));
    writer.Add("foo", Constant_2x3(2.f));
    EXPECT_TRUE(
        StringPiece(writer.status().ToString()).contains("duplicate key"));
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Double finish
    BundleWriter writer(Env::Default(), Prefix("bad"));
    EXPECT_TRUE(writer.Finish().ok());
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Not found.
    BundleReader reader(Env::Default(), Prefix("nonexist"));
    EXPECT_TRUE(StringPiece(reader.status().ToString()).contains("Not found"));
  }
}

TEST(TensorBundleTest, Checksum) {
  // Randomly flips a byte in [pos_lhs, end of data file), or exactly byte
  // pos_lhs if exact_pos == True.
  auto FlipByte = [](const string& prefix, int pos_lhs,
                     bool exact_pos = false) {
    DCHECK_GE(pos_lhs, 0);
    const string& datafile = DataFilename(Prefix(prefix), 0, 1);
    string data;
    TF_ASSERT_OK(ReadFileToString(Env::Default(), datafile, &data));

    int byte_pos = 0;
    if (!exact_pos) {
      std::mt19937 rng;
      std::uniform_int_distribution<int> dist(pos_lhs, data.size() - 1);
      byte_pos = dist(rng);
    } else {
      byte_pos = pos_lhs;
    }
    data[byte_pos] = ~data[byte_pos];
    TF_ASSERT_OK(WriteStringToFile(Env::Default(), datafile, data));
  };
  // The lookup should fail with a checksum-related message.
  auto ExpectLookupFails = [](const string& prefix, const string& key,
                              const string& expected_msg, Tensor& val) {
    BundleReader reader(Env::Default(), Prefix(prefix));
    Status status = reader.Lookup(key, &val);
    EXPECT_TRUE(errors::IsDataLoss(status));
    EXPECT_TRUE(StringPiece(status.ToString()).contains(expected_msg));
  };

  // Corrupts a float tensor.
  {
    BundleWriter writer(Env::Default(), Prefix("singleton"));
    writer.Add("foo", Constant_2x3(1.f));
    TF_ASSERT_OK(writer.Finish());

    FlipByte("singleton", 0 /* corrupts any byte */);
    Tensor val(DT_FLOAT, TensorShape({2, 3}));
    ExpectLookupFails("singleton", "foo",
                      "Checksum does not match" /* expected fail msg */, val);
  }
  // Corrupts a string tensor.
  {
    auto WriteStrings = []() {
      BundleWriter writer(Env::Default(), Prefix("strings"));
      writer.Add("foo", test::AsTensor<string>({"hello", "world"}));
      TF_ASSERT_OK(writer.Finish());
    };
    // Corrupts the first two bytes, which are the varint32-encoded lengths
    // of the two string elements.  Should hit mismatch on length cksum.
    for (int i = 0; i < 2; ++i) {
      WriteStrings();
      FlipByte("strings", i, true /* corrupts exactly byte i */);
      Tensor val(DT_STRING, TensorShape({2}));
      ExpectLookupFails(
          "strings", "foo",
          "length checksum does not match" /* expected fail msg */, val);
    }
    // Corrupts the string bytes, should hit an overall cksum mismatch.
    WriteStrings();
    FlipByte("strings", 2 /* corrupts starting from byte 2 */);
    Tensor val(DT_STRING, TensorShape({2}));
    ExpectLookupFails("strings", "foo",
                      "Checksum does not match" /* expected fail msg */, val);
  }
}

TEST(TensorBundleTest, Endianness) {
  BundleWriter writer(Env::Default(), Prefix("end"));
  writer.Add("key", Constant_2x3<float>(1.0));
  TF_ASSERT_OK(writer.Finish());

  // Flips the endianness bit.
  TF_ASSERT_OK(FlipEndiannessBit(Prefix("end")));

  BundleReader reader(Env::Default(), Prefix("end"));
  EXPECT_TRUE(errors::IsUnimplemented(reader.status()));
  EXPECT_TRUE(StringPiece(reader.status().ToString())
                  .contains("different endianness from the reader"));
}

TEST(TensorBundleTest, TruncatedTensorContents) {
  Env* env = Env::Default();
  BundleWriter writer(env, Prefix("end"));
  writer.Add("key", Constant_2x3<float>(1.0));
  TF_ASSERT_OK(writer.Finish());

  // Truncates the data file by one byte, so that we hit EOF.
  const string datafile = DataFilename(Prefix("end"), 0, 1);
  string data;
  TF_ASSERT_OK(ReadFileToString(env, datafile, &data));
  ASSERT_TRUE(!data.empty());
  TF_ASSERT_OK(WriteStringToFile(env, datafile,
                                 StringPiece(data.data(), data.size() - 1)));

  BundleReader reader(env, Prefix("end"));
  TF_ASSERT_OK(reader.status());
  Tensor val(DT_FLOAT, TensorShape({2, 3}));
#if defined(PLATFORM_GOOGLE)
  EXPECT_EQ("Data loss: Requested 24 bytes but read 23 bytes.",
            reader.Lookup("key", &val).ToString());
#else
  EXPECT_TRUE(errors::IsOutOfRange(reader.Lookup("key", &val)));
#endif
}

TEST(TensorBundleTest, HeaderEntry) {
  {
    BundleWriter writer(Env::Default(), Prefix("b"));
    writer.Add("key", Constant_2x3<float>(1.0));
    TF_ASSERT_OK(writer.Finish());
  }

  // Extracts out the header.
  BundleHeaderProto header;
  {
    BundleReader reader(Env::Default(), Prefix("b"));
    TF_ASSERT_OK(reader.status());
    reader.Seek(kHeaderEntryKey);
    ASSERT_TRUE(reader.Valid());
    ASSERT_TRUE(ParseProtoUnlimited(&header, reader.value().data(),
                                    reader.value().size()));
  }

  // num_shards
  EXPECT_EQ(1, header.num_shards());
  // endianness
  if (port::kLittleEndian) {
    EXPECT_EQ(BundleHeaderProto::LITTLE, header.endianness());
  } else {
    EXPECT_EQ(BundleHeaderProto::BIG, header.endianness());
  }
  // version
  EXPECT_GT(kTensorBundleVersion, 0);
  EXPECT_EQ(kTensorBundleVersion, header.version().producer());
  EXPECT_EQ(kTensorBundleMinConsumer, header.version().min_consumer());
}

TEST(TensorBundleTest, VersionTest) {
  // Min consumer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleVersion + 1);
    versions.set_min_consumer(kTensorBundleVersion + 1);
    VersionTest(
        versions,
        strings::StrCat("Checkpoint min consumer version ",
                        kTensorBundleVersion + 1, " above current version ",
                        kTensorBundleVersion, " for TensorFlow"));
  }
  // Min producer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleMinProducer - 1);
    VersionTest(
        versions,
        strings::StrCat("Checkpoint producer version ",
                        kTensorBundleMinProducer - 1, " below min producer ",
                        kTensorBundleMinProducer, " supported by TensorFlow"));
  }
  // Bad consumer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleVersion + 1);
    versions.add_bad_consumers(kTensorBundleVersion);
    VersionTest(
        versions,
        strings::StrCat(
            "Checkpoint disallows consumer version ", kTensorBundleVersion,
            ".  Please upgrade TensorFlow: this version is likely buggy."));
  }
}

}  // namespace tensorflow
