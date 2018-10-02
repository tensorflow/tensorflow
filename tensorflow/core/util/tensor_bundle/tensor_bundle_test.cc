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
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

string TestdataPrefix(const string& prefix) {
  return strings::StrCat(testing::TensorFlowSrcRoot(),
                         "/core/util/tensor_bundle/testdata/", prefix);
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
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  EXPECT_EQ(expected_val.dtype(), dtype);
  EXPECT_EQ(expected_val.shape(), shape);
  // Tests for Lookup(), checking tensor contents.
  Tensor val(expected_val.dtype(), shape);
  TF_ASSERT_OK(reader->Lookup(key, &val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

template <class T>
void ExpectVariant(BundleReader* reader, const string& key,
                   const Tensor& expected_t) {
  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  // Tests for Lookup(), checking tensor contents.
  EXPECT_EQ(expected_t.dtype(), dtype);
  EXPECT_EQ(expected_t.shape(), shape);
  Tensor actual_t(dtype, shape);
  TF_ASSERT_OK(reader->Lookup(key, &actual_t));
  for (int i = 0; i < expected_t.NumElements(); i++) {
    Variant actual_var = actual_t.flat<Variant>()(i);
    Variant expected_var = expected_t.flat<Variant>()(i);
    EXPECT_EQ(actual_var.TypeName(), expected_var.TypeName());
    auto* actual_val = actual_var.get<T>();
    auto* expected_val = expected_var.get<T>();
    EXPECT_EQ(*expected_val, *actual_val);
  }
}

template <typename T>
void ExpectNext(BundleReader* reader, const Tensor& expected_val) {
  EXPECT_TRUE(reader->Valid());
  reader->Next();
  TF_ASSERT_OK(reader->status());
  Tensor val;
  TF_ASSERT_OK(reader->ReadCurrent(&val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    ret.emplace_back(reader->key());
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
    TF_EXPECT_OK(writer.Add("foo_003", Constant_2x3<T>(3)));
    TF_EXPECT_OK(writer.Add("foo_000", Constant_2x3<T>(0)));
    TF_EXPECT_OK(writer.Add("foo_002", Constant_2x3<T>(2)));
    TF_EXPECT_OK(writer.Add("foo_001", Constant_2x3<T>(1)));
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
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(0));
    ExpectNext<T>(&reader, Constant_2x3<T>(1));
    ExpectNext<T>(&reader, Constant_2x3<T>(2));
    ExpectNext<T>(&reader, Constant_2x3<T>(3));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    BundleWriter writer(Env::Default(), Prefix("bar"));
    TF_EXPECT_OK(writer.Add("bar_003", Constant_2x3<T>(3)));
    TF_EXPECT_OK(writer.Add("bar_000", Constant_2x3<T>(0)));
    TF_EXPECT_OK(writer.Add("bar_002", Constant_2x3<T>(2)));
    TF_EXPECT_OK(writer.Add("bar_001", Constant_2x3<T>(1)));
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
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(0));
    ExpectNext<T>(&reader, Constant_2x3<T>(1));
    ExpectNext<T>(&reader, Constant_2x3<T>(2));
    ExpectNext<T>(&reader, Constant_2x3<T>(3));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
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
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(0));
    ExpectNext<T>(&reader, Constant_2x3<T>(1));
    ExpectNext<T>(&reader, Constant_2x3<T>(2));
    ExpectNext<T>(&reader, Constant_2x3<T>(3));
    ExpectNext<T>(&reader, Constant_2x3<T>(0));
    ExpectNext<T>(&reader, Constant_2x3<T>(1));
    ExpectNext<T>(&reader, Constant_2x3<T>(2));
    ExpectNext<T>(&reader, Constant_2x3<T>(3));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
}

template <typename T>
void TestNonStandardShapes() {
  {
    BundleWriter writer(Env::Default(), Prefix("nonstandard"));
    TF_EXPECT_OK(writer.Add("scalar", Constant<T>(0, TensorShape())));
    TF_EXPECT_OK(
        writer.Add("non_standard0", Constant<T>(0, TensorShape({0, 1618}))));
    TF_EXPECT_OK(
        writer.Add("non_standard1", Constant<T>(0, TensorShape({16, 0, 18}))));
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
      str_util::StartsWith(reader.status().error_message(), expected_error));
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
  TensorSlice slice1 = TensorSlice::ParseOrDie("-:0,1");
  TensorSlice slice2 = TensorSlice::ParseOrDie("-:1,9");
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));

    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape, slice1,
                                 Constant<float>(0., TensorShape({5, 1}))));
    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape, slice2,
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
  // Reads all slices.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    std::vector<TensorSlice> slices;
    TF_ASSERT_OK(reader.LookupTensorSlices("foo", &slices));

    EXPECT_EQ(2, slices.size());
    EXPECT_EQ(slice1.DebugString(), slices[0].DebugString());
    EXPECT_EQ(slice2.DebugString(), slices[1].DebugString());
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

TEST(TensorBundleTest, EquivalentSliceTest) {
  const TensorShape kFullShape({5, 10});
  const Tensor kExpected(Constant<float>(1., kFullShape));
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(writer.AddSlice("no_extents", kFullShape,
                                 TensorSlice::ParseOrDie("-:-"), kExpected));
    TF_ASSERT_OK(writer.AddSlice("both_extents", kFullShape,
                                 TensorSlice::ParseOrDie("0,5:0,10"),
                                 kExpected));
    TF_ASSERT_OK(writer.Finish());
  }
  // Slices match exactly and are fully abbreviated.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("no_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Slice match exactly and are fully specified.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("0,5:0,10");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("both_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Stored slice has no extents, spec has extents.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("0,5:0,10");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("no_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Stored slice has both extents, spec has no extents.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("both_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
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

TEST(TensorBundleTest, StringTensorsOldFormat) {
  // Test string tensor bundle made with previous version of code that use
  // varint32s to store string lengths (we now use varint64s).
  BundleReader reader(Env::Default(), TestdataPrefix("old_string_tensors/foo"));
  TF_ASSERT_OK(reader.status());
  EXPECT_EQ(AllTensorKeys(&reader),
            std::vector<string>({"floats", "scalar", "string_tensor", "strs"}));

  Expect<string>(&reader, "string_tensor", Tensor(DT_STRING, TensorShape({1})));
  Expect<string>(&reader, "scalar", test::AsTensor<string>({"hello"}));
  Expect<string>(
      &reader, "strs",
      test::AsTensor<string>({"hello", "", "x01", string(1 << 10, 'c')}));
  Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));
}

TEST(TensorBundleTest, StringTensors) {
  constexpr size_t kLongLength = static_cast<size_t>(UINT32_MAX) + 1;
  Tensor long_string_tensor(DT_STRING, TensorShape({1}));

  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("string_tensor",
                            Tensor(DT_STRING, TensorShape({1}))));  // Empty.
    TF_EXPECT_OK(writer.Add("scalar", test::AsTensor<string>({"hello"})));
    TF_EXPECT_OK(writer.Add(
        "strs",
        test::AsTensor<string>({"hello", "", "x01", string(1 << 25, 'c')})));

    // Requires a 64-bit length.
    string* backing_string = long_string_tensor.flat<string>().data();
    backing_string->assign(kLongLength, 'd');
    TF_EXPECT_OK(writer.Add("long_scalar", long_string_tensor));

    // Mixes in some floats.
    TF_EXPECT_OK(writer.Add("floats", Constant_2x3<float>(16.18)));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(AllTensorKeys(&reader),
              std::vector<string>({"floats", "long_scalar", "scalar",
                                   "string_tensor", "strs"}));

    Expect<string>(&reader, "string_tensor",
                   Tensor(DT_STRING, TensorShape({1})));
    Expect<string>(&reader, "scalar", test::AsTensor<string>({"hello"}));
    Expect<string>(
        &reader, "strs",
        test::AsTensor<string>({"hello", "", "x01", string(1 << 25, 'c')}));

    Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));

    // We don't use the Expect function so we can re-use the
    // `long_string_tensor` buffer for reading out long_scalar to keep memory
    // usage reasonable.
    EXPECT_TRUE(reader.Contains("long_scalar"));
    DataType dtype;
    TensorShape shape;
    TF_ASSERT_OK(reader.LookupDtypeAndShape("long_scalar", &dtype, &shape));
    EXPECT_EQ(DT_STRING, dtype);
    EXPECT_EQ(TensorShape({1}), shape);

    // Zero-out the string so that we can be sure the new one is read in.
    string* backing_string = long_string_tensor.flat<string>().data();
    backing_string->assign("");

    // Read long_scalar and check it contains kLongLength 'd's.
    TF_ASSERT_OK(reader.Lookup("long_scalar", &long_string_tensor));
    ASSERT_EQ(backing_string, long_string_tensor.flat<string>().data());
    EXPECT_EQ(kLongLength, backing_string->length());
    for (char c : *backing_string) {
      // Not using ASSERT_EQ('d', c) because this way is twice as fast due to
      // compiler optimizations.
      if (c != 'd') {
        FAIL() << "long_scalar is not full of 'd's as expected.";
        break;
      }
    }
  }
}

class VariantObject {
 public:
  VariantObject() {}
  VariantObject(const string& metadata, int64 value)
      : metadata_(metadata), value_(value) {}

  string TypeName() const { return "TEST VariantObject"; }
  void Encode(VariantTensorData* data) const {
    data->set_type_name(TypeName());
    data->set_metadata(metadata_);
    Tensor val_t = Tensor(DT_INT64, TensorShape({}));
    val_t.scalar<int64>()() = value_;
    *(data->add_tensors()) = val_t;
  }
  bool Decode(const VariantTensorData& data) {
    EXPECT_EQ(data.type_name(), TypeName());
    data.get_metadata(&metadata_);
    EXPECT_EQ(data.tensors_size(), 1);
    value_ = data.tensors(0).scalar<int64>()();
    return true;
  }
  bool operator==(const VariantObject other) const {
    return metadata_ == other.metadata_ && value_ == other.value_;
  }
  string metadata_;
  int64 value_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VariantObject, "TEST VariantObject");

TEST(TensorBundleTest, VariantTensors) {
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(
        writer.Add("variant_tensor",
                   test::AsTensor<Variant>({VariantObject("test", 10),
                                            VariantObject("test1", 20)})));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectVariant<VariantObject>(
        &reader, "variant_tensor",
        test::AsTensor<Variant>(
            {VariantObject("test", 10), VariantObject("test1", 20)}));
  }
}

TEST(TensorBundleTest, DirectoryStructure) {
  Env* env = Env::Default();
  // Writes two bundles.
  const std::vector<string> kBundlePrefixes = {Prefix("worker0"),
                                               Prefix("worker1")};
  for (int i = 0; i < 2; ++i) {
    BundleWriter writer(env, kBundlePrefixes[i]);
    TF_EXPECT_OK(
        writer.Add(strings::StrCat("tensor", i), Constant_2x3<float>(0.)));
    TF_ASSERT_OK(writer.Finish());
  }

  // Ensures we have the expected files.
  auto CheckDirFiles = [env](const string& bundle_prefix,
                             gtl::ArraySlice<string> expected_files) {
    StringPiece dir = io::Dirname(bundle_prefix);
    for (const string& expected_file : expected_files) {
      TF_EXPECT_OK(env->FileExists(io::JoinPath(dir, expected_file)));
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
    TF_EXPECT_OK(writer.Add("foo", Constant_2x3(1.f)));
    EXPECT_FALSE(writer.Add("foo", Constant_2x3(2.f)).ok());
    EXPECT_TRUE(
        str_util::StrContains(writer.status().ToString(), "duplicate key"));
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Double finish
    BundleWriter writer(Env::Default(), Prefix("bad"));
    EXPECT_TRUE(writer.Finish().ok());
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Not found.
    BundleReader reader(Env::Default(), Prefix("nonexist"));
    EXPECT_TRUE(str_util::StrContains(reader.status().ToString(), "Not found"));
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
    EXPECT_TRUE(str_util::StrContains(status.ToString(), expected_msg));
  };

  // Corrupts a float tensor.
  {
    BundleWriter writer(Env::Default(), Prefix("singleton"));
    TF_EXPECT_OK(writer.Add("foo", Constant_2x3(1.f)));
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
      TF_EXPECT_OK(
          writer.Add("foo", test::AsTensor<string>({"hello", "world"})));
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
  TF_EXPECT_OK(writer.Add("key", Constant_2x3<float>(1.0)));
  TF_ASSERT_OK(writer.Finish());

  // Flips the endianness bit.
  TF_ASSERT_OK(FlipEndiannessBit(Prefix("end")));

  BundleReader reader(Env::Default(), Prefix("end"));
  EXPECT_TRUE(errors::IsUnimplemented(reader.status()));
  EXPECT_TRUE(str_util::StrContains(reader.status().ToString(),
                                    "different endianness from the reader"));
}

TEST(TensorBundleTest, TruncatedTensorContents) {
  Env* env = Env::Default();
  BundleWriter writer(env, Prefix("end"));
  TF_EXPECT_OK(writer.Add("key", Constant_2x3<float>(1.0)));
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
  EXPECT_TRUE(errors::IsOutOfRange(reader.Lookup("key", &val)));
}

TEST(TensorBundleTest, HeaderEntry) {
  {
    BundleWriter writer(Env::Default(), Prefix("b"));
    TF_EXPECT_OK(writer.Add("key", Constant_2x3<float>(1.0)));
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

class TensorBundleAlignmentTest : public ::testing::Test {
 protected:
  template <typename T>
  void ExpectAlignment(BundleReader* reader, const string& key, int alignment) {
    BundleEntryProto full_tensor_entry;
    TF_ASSERT_OK(reader->GetBundleEntryProto(key, &full_tensor_entry));
    EXPECT_EQ(0, full_tensor_entry.offset() % alignment);
  }
};

TEST_F(TensorBundleAlignmentTest, AlignmentTest) {
  {
    BundleWriter::Options opts;
    opts.data_alignment = 42;
    BundleWriter writer(Env::Default(), Prefix("foo"), opts);
    TF_EXPECT_OK(writer.Add("foo_003", Constant_2x3<float>(3)));
    TF_EXPECT_OK(writer.Add("foo_000", Constant_2x3<float>(0)));
    TF_EXPECT_OK(writer.Add("foo_002", Constant_2x3<float>(2)));
    TF_EXPECT_OK(writer.Add("foo_001", Constant_2x3<float>(1)));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<float>(&reader, "foo_000", Constant_2x3<float>(0));
    Expect<float>(&reader, "foo_001", Constant_2x3<float>(1));
    Expect<float>(&reader, "foo_002", Constant_2x3<float>(2));
    Expect<float>(&reader, "foo_003", Constant_2x3<float>(3));
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<float>(&reader, Constant_2x3<float>(0));
    ExpectNext<float>(&reader, Constant_2x3<float>(1));
    ExpectNext<float>(&reader, Constant_2x3<float>(2));
    ExpectNext<float>(&reader, Constant_2x3<float>(3));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectAlignment<float>(&reader, "foo_000", 42);
    ExpectAlignment<float>(&reader, "foo_001", 42);
    ExpectAlignment<float>(&reader, "foo_002", 42);
    ExpectAlignment<float>(&reader, "foo_003", 42);
  }
}

static void BM_BundleAlignmentByteOff(int iters, int alignment,
                                      int tensor_size) {
  testing::StopTiming();
  {
    BundleWriter::Options opts;
    opts.data_alignment = alignment;
    BundleWriter writer(Env::Default(), Prefix("foo"), opts);
    TF_CHECK_OK(writer.Add("small", Constant(true, TensorShape({1}))));
    TF_CHECK_OK(writer.Add("big", Constant(32.1, TensorShape({tensor_size}))));
    TF_CHECK_OK(writer.Finish());
  }
  BundleReader reader(Env::Default(), Prefix("foo"));
  TF_CHECK_OK(reader.status());
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    Tensor t;
    TF_CHECK_OK(reader.Lookup("big", &t));
  }
  testing::StopTiming();
}

#define BM_BundleAlignment(ALIGN, SIZE)                        \
  static void BM_BundleAlignment_##ALIGN##_##SIZE(int iters) { \
    BM_BundleAlignmentByteOff(iters, ALIGN, SIZE);             \
  }                                                            \
  BENCHMARK(BM_BundleAlignment_##ALIGN##_##SIZE)

BM_BundleAlignment(1, 512);
BM_BundleAlignment(1, 4096);
BM_BundleAlignment(1, 1048576);
BM_BundleAlignment(4096, 512);
BM_BundleAlignment(4096, 4096);
BM_BundleAlignment(4096, 1048576);

}  // namespace tensorflow
