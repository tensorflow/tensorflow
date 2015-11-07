#include "tensorflow/core/util/tensor_slice_reader.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_writer.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include <gtest/gtest.h>

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

void SimpleFloatHelper(TensorSliceWriter::CreateBuilderFunction create_function,
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
  TensorSliceReader reader(filepattern, open_function);
  EXPECT_OK(reader.status());
  EXPECT_EQ(2, reader.num_files());

  // We query some of the tensors
  {
    TensorShape shape;
    DataType type;
    EXPECT_TRUE(reader.HasTensor("test", &shape, &type));
    EXPECT_EQ(
        "dim { size: 4 } "
        "dim { size: 5 }",
        shape.DebugString());
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
void SimpleIntXHelper(TensorSliceWriter::CreateBuilderFunction create_function,
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
  TensorSliceReader reader(filepattern, open_function);
  EXPECT_OK(reader.status());
  EXPECT_EQ(2, reader.num_files());

  // We query some of the tensors
  {
    TensorShape shape;
    DataType type;
    EXPECT_TRUE(reader.HasTensor("test", &shape, &type));
    EXPECT_EQ(
        "dim { size: 4 } "
        "dim { size: 5 }",
        shape.DebugString());
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
TEST_SIMPLE_INT(int64, int64)
TEST_SIMPLE_INT(int16, int32)
TEST_SIMPLE_INT(int8, int32)
TEST_SIMPLE_INT(uint8, int32)

void CachedTensorSliceReaderTesterHelper(
    TensorSliceWriter::CreateBuilderFunction create_function,
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
    EXPECT_EQ(
        "dim { size: 4 } "
        "dim { size: 5 }",
        shape.DebugString());
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

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
