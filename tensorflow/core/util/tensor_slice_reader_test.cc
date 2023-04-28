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

#include "tensorflow/core/util/tensor_slice_reader.h"

#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/iterator.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

namespace checkpoint {

namespace {

// A simple test where we write a few tensor slices with a number of tensor
// slice writers and then read them back from a tensor slice reader.
//
// We have a 2-d tensor of shape 4 X 5 that looks like this:
//
//   0   1   2   3   4
//   5   6   7   8   9
//  10  11  12  13  14
//  15  16  17  18  19
//
// We assume this is a row-major matrix.

void SimpleFloatHelper(
    const TensorSliceWriter::CreateBuilderFunction& create_function,
    TensorSliceReader::OpenTableFunction open_function) {
  const string fname_base = io::JoinPath(testing::TmpDir(), "float_checkpoint");

  TensorShape shape({4, 5});

  // File #0 contains a slice that is the top two rows:
  //
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    const string fname = strings::StrCat(fname_base, "_0");
    TensorSliceWriter writer(fname, create_function);
    const float data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TensorSlice slice = TensorSlice::ParseOrDie("0,2:-");
    TF_CHECK_OK(writer.Add("test", shape, slice, data));
    TF_CHECK_OK(writer.Finish());
  }

  // File #1 contains two slices:
  //
  // slice #0 is the bottom left corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //  10  11  12   .   .
  //  15  16  17   .   .
  //
  // slice #1 is the bottom right corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .  18  19
  {
    const string fname = strings::StrCat(fname_base, "_1");
    TensorSliceWriter writer(fname, create_function);
    // slice #0
    {
      const float data[] = {10, 11, 12, 15, 16, 17};
      TensorSlice slice = TensorSlice::ParseOrDie("2,2:0,3");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    // slice #1
    {
      const float data[] = {18, 19};
      TensorSlice slice = TensorSlice::ParseOrDie("3,1:3,2");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    TF_CHECK_OK(writer.Finish());
  }

  // Notice that we leave a hole in the tensor
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   . (13) (14)
  //   .   .   .   .   .

  // Now we need to read the tensor slices
  const string filepattern = strings::StrCat(fname_base, "_*");
  TensorSliceReader reader(filepattern, std::move(open_function));
  TF_EXPECT_OK(reader.status());
  EXPECT_EQ(2, reader.num_files());

  // We query some of the tensors
  {
    TensorShape shape;
    DataType type;
    EXPECT_TRUE(reader.HasTensor("test", &shape, &type));
    EXPECT_EQ("[4,5]", shape.DebugString());
    EXPECT_EQ(DT_FLOAT, type);
    EXPECT_FALSE(reader.HasTensor("don't exist", nullptr, nullptr));
  }

  // Now we query some slices
  //
  // Slice #1 is an exact match
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("0,2:-");
    float expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float results[10];
    EXPECT_TRUE(reader.CopySliceData("test", s, results));
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(expected[i], results[i]);
    }
  }

  // Slice #2 is a subset match
  //   .   .   .   .   .
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,1:-");
    float expected[] = {5, 6, 7, 8, 9};
    float results[5];
    EXPECT_TRUE(reader.CopySliceData("test", s, results));
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(expected[i], results[i]);
    }
  }

  // Slice #4 includes the hole and so there is no match
  //   .   .   .   .   .
  //   .   .   7   8   9
  //   .   .  12  13  14
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:2,3");
    float results[6];
    EXPECT_FALSE(reader.CopySliceData("test", s, results));
  }
}

TEST(TensorSliceReaderTest, SimpleFloat) {
  SimpleFloatHelper(CreateTableTensorSliceBuilder, OpenTableTensorSliceReader);
}

template <typename T, typename U>
void SimpleIntXHelper(
    const TensorSliceWriter::CreateBuilderFunction& create_function,
    TensorSliceReader::OpenTableFunction open_function,
    const string& checkpoint_file) {
  const string fname_base = io::JoinPath(testing::TmpDir(), checkpoint_file);

  TensorShape shape({4, 5});

  // File #0 contains a slice that is the top two rows:
  //
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    const string fname = strings::StrCat(fname_base, "_0");
    TensorSliceWriter writer(fname, create_function);
    const T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TensorSlice slice = TensorSlice::ParseOrDie("0,2:-");
    TF_CHECK_OK(writer.Add("test", shape, slice, data));
    TF_CHECK_OK(writer.Finish());
  }

  // File #1 contains two slices:
  //
  // slice #0 is the bottom left corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //  10  11  12   .   .
  //  15  16  17   .   .
  //
  // slice #1 is the bottom right corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .  18  19
  {
    const string fname = strings::StrCat(fname_base, "_1");
    TensorSliceWriter writer(fname, create_function);
    // slice #0
    {
      const T data[] = {10, 11, 12, 15, 16, 17};
      TensorSlice slice = TensorSlice::ParseOrDie("2,2:0,3");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    // slice #1
    {
      const T data[] = {18, 19};
      TensorSlice slice = TensorSlice::ParseOrDie("3,1:3,2");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    TF_CHECK_OK(writer.Finish());
  }

  // Notice that we leave a hole in the tensor
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   . (13) (14)
  //   .   .   .   .   .

  // Now we need to read the tensor slices
  const string filepattern = strings::StrCat(fname_base, "_*");
  TensorSliceReader reader(filepattern, std::move(open_function));
  TF_EXPECT_OK(reader.status());
  EXPECT_EQ(2, reader.num_files());

  // We query some of the tensors
  {
    TensorShape shape;
    DataType type;
    EXPECT_TRUE(reader.HasTensor("test", &shape, &type));
    EXPECT_EQ("[4,5]", shape.DebugString());
    EXPECT_EQ(DataTypeToEnum<T>::v(), type);
    EXPECT_FALSE(reader.HasTensor("don't exist", nullptr, nullptr));
  }

  // Now we query some slices
  //
  // Slice #1 is an exact match
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("0,2:-");
    T expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    U results[10];
    EXPECT_TRUE(reader.CopySliceData("test", s, results));
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(expected[i], results[i]);
    }
  }

  // Slice #2 is a subset match
  //   .   .   .   .   .
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,1:-");
    T expected[] = {5, 6, 7, 8, 9};
    U results[5];
    EXPECT_TRUE(reader.CopySliceData("test", s, results));
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(expected[i], results[i]);
    }
  }

  // Slice #4 includes the hole and so there is no match
  //   .   .   .   .   .
  //   .   .   7   8   9
  //   .   .  12  13  14
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:2,3");
    U results[6];
    EXPECT_FALSE(reader.CopySliceData("test", s, results));
  }
}

#define TEST_SIMPLE_INT(TYPE, SAVED_TYPE)                             \
  TEST(TensorSliceReaderTest, Simple##TYPE) {                         \
    SimpleIntXHelper<TYPE, SAVED_TYPE>(CreateTableTensorSliceBuilder, \
                                       OpenTableTensorSliceReader,    \
                                       #TYPE "_checkpoint");          \
  }

TEST_SIMPLE_INT(int32, int32)
TEST_SIMPLE_INT(int64_t, int64_t)
TEST_SIMPLE_INT(int16, int32)
TEST_SIMPLE_INT(int8, int32)
TEST_SIMPLE_INT(uint8, int32)

// Modifies the SavedTensorSlices messages in a checkpoint to allow creating
// malformed or unsupported checkpoints.
void MutateSavedTensorSlices(
    const std::string& fname,
    const std::function<std::string(SavedTensorSlices)>& mutator) {
  table::Options options;
  options.compression = table::kNoCompression;

  // Read all entres from the table.
  std::vector<std::pair<std::string, std::string>> entries;
  {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(Env::Default()->NewRandomAccessFile(fname, &file));
    uint64 file_size;
    TF_CHECK_OK(Env::Default()->GetFileSize(fname, &file_size));
    table::Table* t;
    TF_CHECK_OK(table::Table::Open(options, file.get(), file_size, &t));
    std::unique_ptr<table::Table> table(t);
    std::unique_ptr<table::Iterator> it(table->NewIterator());
    for (it->Seek(""); it->Valid(); it->Next()) {
      entries.emplace_back(it->key(), it->value());
    }
    TF_CHECK_OK(it->status());
  }

  // Rewrite the table, mutating each value.
  {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(fname, &file));
    table::TableBuilder builder(options, file.get());
    for (const auto& entry : entries) {
      SavedTensorSlices sts;
      CHECK(sts.ParseFromString(entry.second));
      builder.Add(entry.first, mutator(std::move(sts)));
    }
    TF_CHECK_OK(builder.Finish());
    TF_CHECK_OK(file->Close());
  }
}

TEST(TensorSliceReaderTest, MissingTensorType) {
  const string fname = io::JoinPath(testing::TmpDir(), "invalid_checkpoint");
  TensorSliceWriter writer(fname, CreateTableTensorSliceBuilder);
  const int32 data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TensorShape shape({4, 5});
  TensorSlice slice = TensorSlice::ParseOrDie("0,2:-");
  TF_CHECK_OK(writer.Add("test", shape, slice, data));
  TF_CHECK_OK(writer.Finish());

  MutateSavedTensorSlices(fname, [](SavedTensorSlices sts) {
    if (sts.has_meta()) {
      for (auto& tensor : *sts.mutable_meta()->mutable_tensor()) {
        tensor.clear_type();
      }
    }
    return sts.SerializeAsString();
  });

  TensorSliceReader reader(fname, OpenTableTensorSliceReader);
  TF_CHECK_OK(reader.status());

  // The tensor should be present, but loading it should fail due to the
  // unset (invalid) type.
  EXPECT_TRUE(reader.HasTensor("test", nullptr, nullptr));
  std::unique_ptr<Tensor> tensor;
  EXPECT_FALSE(reader.GetTensor("test", &tensor).ok());
}

TEST(TensorSliceReaderTest, UnsupportedTensorType) {
  const string fname = io::JoinPath(testing::TmpDir(), "int32_ref_checkpoint");
  TensorSliceWriter writer(fname, CreateTableTensorSliceBuilder);
  const int32 data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TensorShape shape({4, 5});
  TensorSlice slice = TensorSlice::ParseOrDie("0,2:-");
  TF_CHECK_OK(writer.Add("test", shape, slice, data));
  TF_CHECK_OK(writer.Finish());

  MutateSavedTensorSlices(fname, [](SavedTensorSlices sts) {
    if (sts.has_meta()) {
      for (auto& tensor : *sts.mutable_meta()->mutable_tensor()) {
        tensor.set_type(DT_INT32_REF);
      }
    }
    return sts.SerializeAsString();
  });

  TensorSliceReader reader(fname, OpenTableTensorSliceReader);
  TF_CHECK_OK(reader.status());

  // The tensor should be present, but loading it should fail due to the
  // unsupported type.
  EXPECT_TRUE(reader.HasTensor("test", nullptr, nullptr));
  std::unique_ptr<Tensor> tensor;
  EXPECT_FALSE(reader.GetTensor("test", &tensor).ok());
}

TEST(TensorSliceReaderTest, NegativeTensorShapeDimension) {
  const string fname =
      io::JoinPath(testing::TmpDir(), "negative_dim_checkpoint");
  TensorSliceWriter writer(fname, CreateTableTensorSliceBuilder);
  const int32 data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TF_CHECK_OK(writer.Add("test", TensorShape({4, 5}),
                         TensorSlice::ParseOrDie("0,2:-"), data));
  TF_CHECK_OK(writer.Finish());

  MutateSavedTensorSlices(fname, [](SavedTensorSlices sts) {
    if (sts.has_meta()) {
      for (auto& tensor : *sts.mutable_meta()->mutable_tensor()) {
        for (auto& dim : *tensor.mutable_shape()->mutable_dim()) {
          dim.set_size(-dim.size());
        }
      }
    }
    return sts.SerializeAsString();
  });

  TensorSliceReader reader(fname, OpenTableTensorSliceReader);
  // The negative dimension should cause loading to fail.
  EXPECT_FALSE(reader.status().ok());
}

TEST(TensorSliceReaderTest, InvalidTensorSlice) {
  const string fname =
      io::JoinPath(testing::TmpDir(), "invalid_slice_checkpoint");
  TensorSliceWriter writer(fname, CreateTableTensorSliceBuilder);
  const int32 data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TF_CHECK_OK(writer.Add("test", TensorShape({4, 5}),
                         TensorSlice::ParseOrDie("0,2:-"), data));
  TF_CHECK_OK(writer.Finish());

  MutateSavedTensorSlices(fname, [](SavedTensorSlices sts) {
    if (sts.has_meta()) {
      for (auto& tensor : *sts.mutable_meta()->mutable_tensor()) {
        tensor.mutable_slice(0)->mutable_extent(0)->set_length(-10);
      }
    }
    return sts.SerializeAsString();
  });

  TensorSliceReader reader(fname, OpenTableTensorSliceReader);
  // The negative exent length should cause loading to fail.
  EXPECT_FALSE(reader.status().ok());
}

TEST(TensorSliceReaderTest, MissingTensorData) {
  const string fname =
      io::JoinPath(testing::TmpDir(), "missing_data_checkpoint");
  TensorSliceWriter writer(fname, CreateTableTensorSliceBuilder);
  const int32 data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TF_ASSERT_OK(writer.Add("test", TensorShape({4, 5}),
                          TensorSlice::ParseOrDie("0,2:-"), data));
  TF_ASSERT_OK(writer.Finish());

  MutateSavedTensorSlices(fname, [&](SavedTensorSlices sts) {
    if (sts.has_data()) {
      // Replace the data with only 4 elements.
      Fill(data, 4, sts.mutable_data()->mutable_data());
    }
    return sts.SerializeAsString();
  });

  TensorSliceReader reader(fname, OpenTableTensorSliceReader);
  TF_ASSERT_OK(reader.status());

  // The tensor should be present, but loading it should fail due to the missing
  // data.
  EXPECT_TRUE(reader.HasTensor("test", nullptr, nullptr));
  std::unique_ptr<Tensor> tensor;
  EXPECT_FALSE(reader.GetTensor("test", &tensor).ok());
}

void CachedTensorSliceReaderTesterHelper(
    const TensorSliceWriter::CreateBuilderFunction& create_function,
    const TensorSliceReader::OpenTableFunction& open_function) {
  const string fname_base = io::JoinPath(testing::TmpDir(), "float_checkpoint");

  TensorShape shape({4, 5});

  // File #0 contains a slice that is the top two rows:
  //
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  {
    const string fname = strings::StrCat(fname_base, "_0");
    TensorSliceWriter writer(fname, create_function);
    const float data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TensorSlice slice = TensorSlice::ParseOrDie("0,2:-");
    TF_CHECK_OK(writer.Add("test", shape, slice, data));
    TF_CHECK_OK(writer.Finish());
  }

  // File #1 contains two slices:
  //
  // slice #0 is the bottom left corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //  10  11  12   .   .
  //  15  16  17   .   .
  //
  // slice #1 is the bottom right corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .  18  19
  {
    const string fname = strings::StrCat(fname_base, "_1");
    TensorSliceWriter writer(fname, create_function);
    // slice #0
    {
      const float data[] = {10, 11, 12, 15, 16, 17};
      TensorSlice slice = TensorSlice::ParseOrDie("2,2:0,3");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    // slice #1
    {
      const float data[] = {18, 19};
      TensorSlice slice = TensorSlice::ParseOrDie("3,1:3,2");
      TF_CHECK_OK(writer.Add("test", shape, slice, data));
    }
    TF_CHECK_OK(writer.Finish());
  }

  // Notice that we leave a hole in the tensor
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   . (13) (14)
  //   .   .   .   .   .

  // Now we need to read the tensor slices
  TensorSliceReaderCache cache;
  const string filepattern = strings::StrCat(fname_base, "_*");
  const TensorSliceReader* reader = cache.GetReader(
      filepattern, open_function, TensorSliceReader::kLoadAllShards);
  EXPECT_TRUE(reader != nullptr);
  EXPECT_EQ(2, reader->num_files());

  // We query some of the tensors
  {
    TensorShape shape;
    DataType type;
    EXPECT_TRUE(reader->HasTensor("test", &shape, &type));
    EXPECT_EQ("[4,5]", shape.DebugString());
    EXPECT_EQ(DT_FLOAT, type);
    EXPECT_FALSE(reader->HasTensor("don't exist", nullptr, nullptr));
  }

  // Make sure the reader is cached.
  const TensorSliceReader* reader2 = cache.GetReader(
      filepattern, open_function, TensorSliceReader::kLoadAllShards);
  EXPECT_EQ(reader, reader2);

  reader = cache.GetReader("file_does_not_exist", open_function,
                           TensorSliceReader::kLoadAllShards);
  EXPECT_TRUE(reader == nullptr);
}

TEST(CachedTensorSliceReaderTest, SimpleFloat) {
  CachedTensorSliceReaderTesterHelper(CreateTableTensorSliceBuilder,
                                      OpenTableTensorSliceReader);
}

static void VersionTest(const VersionDef& versions, const string& error) {
  const string path = io::JoinPath(testing::TmpDir(), "checkpoint");

  {
    // Prepare an empty checkpoint with some version information
    SavedTensorSlices sts;
    *sts.mutable_meta()->mutable_versions() = versions;
    string contents;
    EXPECT_TRUE(sts.SerializeToString(&contents));

    // Write it to disk
    TensorSliceWriter::Builder* builder;
    TF_ASSERT_OK(CreateTableTensorSliceBuilder(path, &builder));
    builder->Add(kSavedTensorSlicesKey, contents);
    int64_t file_size;
    TF_EXPECT_OK(builder->Finish(&file_size));
    delete builder;
  }

  // Read it back in and verify that we get the expected error
  TensorSliceReader reader(path, OpenTableTensorSliceReader);
  EXPECT_TRUE(reader.status().code() == error::INVALID_ARGUMENT &&
              absl::StartsWith(reader.status().message(), error))
      << "Expected error starting with '" << errors::InvalidArgument(error)
      << "', got '" << reader.status() << "'";
}

TEST(CheckpointVersionTest, MinConsumer) {
  VersionDef versions;
  versions.set_producer(TF_CHECKPOINT_VERSION + 1);
  versions.set_min_consumer(TF_CHECKPOINT_VERSION + 1);
  VersionTest(
      versions,
      strings::StrCat("Checkpoint min consumer version ",
                      TF_CHECKPOINT_VERSION + 1, " above current version ",
                      TF_CHECKPOINT_VERSION, " for TensorFlow"));
}

TEST(CheckpointVersionTest, MinProducer) {
  VersionDef versions;
  versions.set_producer(TF_CHECKPOINT_VERSION_MIN_PRODUCER - 1);
  VersionTest(versions, strings::StrCat("Checkpoint producer version ",
                                        TF_CHECKPOINT_VERSION_MIN_PRODUCER - 1,
                                        " below min producer ",
                                        TF_CHECKPOINT_VERSION_MIN_PRODUCER,
                                        " supported by TensorFlow"));
}

TEST(CheckpointVersionTest, BadConsumer) {
  VersionDef versions;
  versions.set_producer(TF_CHECKPOINT_VERSION + 1);
  versions.add_bad_consumers(TF_CHECKPOINT_VERSION);
  VersionTest(
      versions,
      strings::StrCat(
          "Checkpoint disallows consumer version ", TF_CHECKPOINT_VERSION,
          ".  Please upgrade TensorFlow: this version is likely buggy."));
}

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
