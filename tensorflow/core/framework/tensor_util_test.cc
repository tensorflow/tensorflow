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

#include "tensorflow/core/framework/tensor_util.h"

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorUtil, DeepCopy0d) {
  Tensor x(DT_FLOAT, TensorShape({}));
  x.scalar<float>()() = 10.0;

  // Make y a deep copy of x and then change it.
  Tensor y = tensor::DeepCopy(x);
  y.scalar<float>()() = 20.0;

  // x doesn't change
  EXPECT_EQ(10.0, x.scalar<float>()());

  // Change x.
  x.scalar<float>()() = 30.0;

  // Y doesn't change.
  EXPECT_EQ(20.0, y.scalar<float>()());

  Tensor z = tensor::DeepCopy(y);

  // Change y.
  y.scalar<float>()() = 40.0;

  // The final states should all be different.
  EXPECT_EQ(20.0, z.scalar<float>()());
  EXPECT_EQ(30.0, x.scalar<float>()());
  EXPECT_EQ(40.0, y.scalar<float>()());

  // Should have the same shape and type.
  EXPECT_EQ(TensorShape({}), x.shape());
  EXPECT_EQ(TensorShape({}), y.shape());
  EXPECT_EQ(TensorShape({}), z.shape());

  EXPECT_EQ(DT_FLOAT, x.dtype());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(DT_FLOAT, z.dtype());
}

TEST(TensorUtil, DeepCopyZeroElements) {
  Tensor x;
  Tensor y = tensor::DeepCopy(x);
  EXPECT_EQ(TensorShape({0}), y.shape());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(0, y.NumElements());
}

TEST(TensorUtil, DeepCopy) {
  Tensor x(DT_FLOAT, TensorShape({1}));
  x.flat<float>()(0) = 10.0;

  // Make y a deep copy of x and then change it.
  Tensor y = tensor::DeepCopy(x);
  y.flat<float>()(0) = 20.0;

  // x doesn't change
  EXPECT_EQ(10.0, x.flat<float>()(0));

  // Change x.
  x.flat<float>()(0) = 30.0;

  // Y doesn't change.
  EXPECT_EQ(20.0, y.flat<float>()(0));

  Tensor z = tensor::DeepCopy(y);

  // Change y.
  y.flat<float>()(0) = 40.0;

  // The final states should all be different.
  EXPECT_EQ(20.0, z.flat<float>()(0));
  EXPECT_EQ(30.0, x.flat<float>()(0));
  EXPECT_EQ(40.0, y.flat<float>()(0));

  // Should have the same shape and type.
  EXPECT_EQ(TensorShape({1}), x.shape());
  EXPECT_EQ(TensorShape({1}), y.shape());
  EXPECT_EQ(TensorShape({1}), z.shape());

  EXPECT_EQ(DT_FLOAT, x.dtype());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(DT_FLOAT, z.dtype());

  // Test string deep copy
  Tensor str1(DT_STRING, TensorShape({2}));
  str1.flat<tstring>()(0) = "foo1";
  str1.flat<tstring>()(1) = "foo2";
  Tensor str2 = tensor::DeepCopy(str1);
  str2.flat<tstring>()(0) = "bar1";
  str2.flat<tstring>()(1) = "bar2";
  EXPECT_NE(str2.flat<tstring>()(0), str1.flat<tstring>()(0));
}

TEST(TensorUtil, DeepCopySlice) {
  Tensor x(DT_INT32, TensorShape({10}));
  x.flat<int32>().setConstant(1);

  // Slice 'x' -- y still refers to the same buffer.
  Tensor y = x.Slice(2, 6);

  // Do a deep copy of y, which is a slice.
  Tensor z = tensor::DeepCopy(y);

  // Set x to be different.
  x.flat<int32>().setConstant(2);

  EXPECT_EQ(TensorShape({10}), x.shape());
  EXPECT_EQ(TensorShape({4}), y.shape());
  EXPECT_EQ(TensorShape({4}), z.shape());
  EXPECT_EQ(DT_INT32, x.dtype());
  EXPECT_EQ(DT_INT32, y.dtype());
  EXPECT_EQ(DT_INT32, z.dtype());

  // x and y should now all be '2', but z should be '1'.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(2, x.flat<int32>()(i));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(2, y.unaligned_flat<int32>()(i));
    EXPECT_EQ(1, z.flat<int32>()(i));
  }
}

TEST(TensorUtil, DeepCopySliceString) {
  Tensor x(DT_STRING, TensorShape({10}));
  x.flat<tstring>().setConstant("hello");

  // Slice 'x' -- y still refers to the same buffer.
  Tensor y = x.Slice(3, 7);

  // Do a deep copy of y, which is a slice.
  Tensor z = tensor::DeepCopy(y);

  // Set x to be different.
  x.flat<tstring>().setConstant("goodbye");

  EXPECT_EQ(TensorShape({10}), x.shape());
  EXPECT_EQ(TensorShape({4}), y.shape());
  EXPECT_EQ(TensorShape({4}), z.shape());
  EXPECT_EQ(DT_STRING, x.dtype());
  EXPECT_EQ(DT_STRING, y.dtype());
  EXPECT_EQ(DT_STRING, z.dtype());

  // x and y should now all be 'goodbye', but z should be 'hello'.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ("goodbye", x.flat<tstring>()(i));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ("goodbye", y.unaligned_flat<tstring>()(i));
    EXPECT_EQ("hello", z.flat<tstring>()(i));
  }
}

TEST(TensorUtil, DeepCopySliceVariant) {
  Tensor x(DT_VARIANT, TensorShape({10}));
  x.flat<Variant>().setConstant(Tensor(42.0f));

  // Slice 'x' -- y still refers to the same buffer.
  Tensor y = x.Slice(3, 7);

  // Do a deep copy of y, which is a slice.
  Tensor z = tensor::DeepCopy(y);

  // Set x to be different.
  x.flat<Variant>().setConstant(Tensor("foo"));

  EXPECT_EQ(TensorShape({10}), x.shape());
  EXPECT_EQ(TensorShape({4}), y.shape());
  EXPECT_EQ(TensorShape({4}), z.shape());
  EXPECT_EQ(DT_VARIANT, x.dtype());
  EXPECT_EQ(DT_VARIANT, y.dtype());
  EXPECT_EQ(DT_VARIANT, z.dtype());

  // Each element of x and y should now be a DT_STRING Tensor containing "foo",
  // but each element of z should be a DT_FLOAT tensor containing 42.0.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ("foo", x.flat<Variant>()(i).get<Tensor>()->scalar<tstring>()());
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        "foo",
        y.unaligned_flat<Variant>()(i).get<Tensor>()->scalar<tstring>()());
    EXPECT_EQ(42.0, z.flat<Variant>()(i).get<Tensor>()->scalar<float>()());
  }
}

TEST(TensorUtil, Concat) {
  std::vector<int64> sizes = {1, 4, 5};
  std::vector<Tensor> to_concat;
  int64 total_size = 0;
  int offset = 0;
  for (size_t entry = 0; entry < sizes.size(); ++entry) {
    const int64 size = sizes[entry];
    Tensor tensor(DT_INT32, TensorShape({size, 2}));
    for (int i = offset; i < offset + size; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.matrix<int32>()(i - offset, j) = 2 * i + j;
      }
    }
    to_concat.push_back(tensor);
    total_size += size;
    offset += size;
  }

  Tensor concated;
  TF_ASSERT_OK(tensor::Concat(to_concat, &concated));
  ASSERT_EQ(TensorShape({total_size, 2}), concated.shape());
  for (int i = 0; i < total_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(2 * i + j, concated.matrix<int32>()(i, j));
    }
  }
}

TEST(TensorUtil, Split) {
  Tensor to_split(DT_INT64, TensorShape({10, 2}));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 2; ++j) {
      to_split.matrix<int64>()(i, j) = 2 * i + j;
    }
  }

  std::vector<int64> sizes = {1, 4, 5};
  std::vector<Tensor> splits;
  TF_ASSERT_OK(tensor::Split(to_split, sizes, &splits));
  ASSERT_EQ(sizes.size(), splits.size());

  int offset = 0;
  for (size_t entry = 0; entry < splits.size(); ++entry) {
    const int64 size = sizes[entry];
    const Tensor& split = splits[entry];

    ASSERT_EQ(TensorShape({size, 2}), split.shape());
    for (int i = offset; i < offset + size; ++i) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_EQ(2 * i + j, split.matrix<int64>()(i - offset, j));
      }
    }

    offset += size;
  }
}

TEST(TensorUtil, ConcatSplitStrings) {
  Tensor x(DT_STRING, TensorShape({4, 3}));
  for (int i = 0; i < 4 * 3; ++i) {
    x.flat<tstring>()(i) = strings::StrCat("foo_", i);
  }

  std::vector<Tensor> split;
  TF_ASSERT_OK(tensor::Split(x, {2, 1, 1}, &split));
  Tensor x_round_tripped;
  TF_ASSERT_OK(tensor::Concat(split, &x_round_tripped));
  ASSERT_EQ(x.shape(), x_round_tripped.shape());
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_EQ(x.flat<tstring>()(i), x_round_tripped.flat<tstring>()(i));
  }

  // Ensure that no memory is being shared between 'x' and 'x_round_tripped'.
  for (int i = 0; i < 4 * 3; ++i) {
    x_round_tripped.flat<tstring>()(i) = strings::StrCat("bar_", i);
  }
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_NE(x.flat<tstring>()(i), x_round_tripped.flat<tstring>()(i));
  }
}

TEST(TensorProtoUtil, CreatesStringTensorProto) {
  std::vector<string> values{"a", "b", "c"};
  std::vector<size_t> shape{1, 3};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_STRING\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 1\n"
            "  }\n"
            "  dim {\n"
            "    size: 3\n"
            "  }\n"
            "}\n"
            "string_val: \"a\"\n"
            "string_val: \"b\"\n"
            "string_val: \"c\"\n");
}

TEST(TensorProtoUtil, CreatesInt32TensorProto) {
  std::vector<int32> values{1, 2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_INT32\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "int_val: 1\n"
            "int_val: 2\n");
}

TEST(TensorProtoUtil, CreatesInt64TensorProto) {
  std::vector<int64> values{1, 2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_INT64\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "int64_val: 1\n"
            "int64_val: 2\n");
}

TEST(TensorProtoUtil, CreatesUInt32TensorProto) {
  std::vector<uint32> values{1, 2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_UINT32\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "uint32_val: 1\n"
            "uint32_val: 2\n");
}

TEST(TensorProtoUtil, CreatesUInt64TensorProto) {
  std::vector<uint64> values{1, 2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_UINT64\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "uint64_val: 1\n"
            "uint64_val: 2\n");
}

TEST(TensorProtoUtil, CreatesFloatTensorProto) {
  std::vector<float> values{1.1, 2.2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_FLOAT\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "float_val: 1.1\n"
            "float_val: 2.2\n");
}

TEST(TensorProtoUtil, CreatesDoubleTensorProto) {
  std::vector<double> values{1.1, 2.2};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_DOUBLE\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "double_val: 1.1\n"
            "double_val: 2.2\n");
}

TEST(TensorProtoUtil, CreatesBoolTensorProto) {
  std::vector<bool> values{true, false};
  std::vector<size_t> shape{2};

  auto proto = tensor::CreateTensorProto(values, shape);

  EXPECT_EQ(proto.DebugString(),
            "dtype: DT_BOOL\n"
            "tensor_shape {\n"
            "  dim {\n"
            "    size: 2\n"
            "  }\n"
            "}\n"
            "bool_val: true\n"
            "bool_val: false\n");
}

TEST(TensorProtoUtil, CompressTensorProtoInPlaceTooSmall) {
  const int kLength = 63;
  TensorProto tensor_proto =
      tensor::CreateTensorProto(std::vector<float>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  tensor_proto =
      tensor::CreateTensorProto(std::vector<int>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  tensor_proto =
      tensor::CreateTensorProto(std::vector<uint8>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  tensor_proto =
      tensor::CreateTensorProto(std::vector<bool>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  tensor_proto =
      tensor::CreateTensorProto(std::vector<Eigen::half>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  tensor_proto = tensor::CreateTensorProto(
      std::vector<std::complex<float>>(kLength), {kLength});
  EXPECT_FALSE(tensor::CompressTensorProtoInPlace(&tensor_proto));
}

TEST(TensorProtoUtil, CompressTensorProtoInPlaceAllEqual) {
  const int kLength = 64;
  TensorProto tensor_proto =
      tensor::CreateTensorProto(std::vector<float>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(tensor::internal::TensorProtoHelper<float>::NumValues(tensor_proto),
            0);

  tensor_proto =
      tensor::CreateTensorProto(std::vector<int>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(tensor::internal::TensorProtoHelper<int>::NumValues(tensor_proto),
            0);

  tensor_proto =
      tensor::CreateTensorProto(std::vector<uint8>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(tensor::internal::TensorProtoHelper<uint8>::NumValues(tensor_proto),
            0);
  tensor_proto =
      tensor::CreateTensorProto(std::vector<bool>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(tensor::internal::TensorProtoHelper<bool>::NumValues(tensor_proto),
            0);

  tensor_proto =
      tensor::CreateTensorProto(std::vector<Eigen::half>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(
      tensor::internal::TensorProtoHelper<Eigen::half>::NumValues(tensor_proto),
      0);

  tensor_proto = tensor::CreateTensorProto(
      std::vector<std::complex<float>>(kLength), {kLength});
  EXPECT_TRUE(tensor::CompressTensorProtoInPlace(&tensor_proto));
  EXPECT_EQ(tensor::internal::TensorProtoHelper<std::complex<float>>::NumValues(
                tensor_proto),
            0);
}

template <typename T>
void VectorWithConstantTail(int size, int tail_length, std::vector<T>* v) {
  CHECK_LE(tail_length, size);
  v->clear();
  for (int i = 0; i < size; ++i) {
    T vi = (i >= size - tail_length) ? T() : T(i);
    v->push_back(vi);
  }
}

template <>
void VectorWithConstantTail(int size, int tail_length,
                            std::vector<std::complex<float>>* v) {
  CHECK_LE(tail_length, size);
  v->clear();
  for (int i = 0; i < size; ++i) {
    std::complex<float> vi(
        0.0f, (i >= (size - tail_length)) ? 0.f : static_cast<float>(i));
    v->push_back(vi);
  }
}

template <typename T>
TensorProto CreateAsProtoTensorContent(int size, int tail_length) {
  std::vector<T> values;
  VectorWithConstantTail<T>(size, tail_length, &values);
  Tensor tensor(DataTypeToEnum<T>::value, TensorShape({size}));
  std::copy(values.begin(), values.end(), tensor.flat<T>().data());
  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  return tensor_proto;
}

template <typename T>
TensorProto CreateAsProtoField(int size, int tail_length) {
  std::vector<T> values;
  VectorWithConstantTail<T>(size, tail_length, &values);
  Tensor tensor(DataTypeToEnum<T>::value, TensorShape({size}));
  std::copy(values.begin(), values.end(), tensor.flat<T>().data());
  TensorProto tensor_proto;
  tensor.AsProtoField(&tensor_proto);
  return tensor_proto;
}

template <typename T>
void CompareTensorValues(const TensorProto& x, const TensorProto& y) {
  Tensor x_t;
  EXPECT_TRUE(x_t.FromProto(x));
  Tensor y_t;
  EXPECT_TRUE(y_t.FromProto(y));
  test::ExpectTensorEqual<T>(x_t, y_t);
}

template <typename T>
void ConstantTailTest(int64 length, int64 tail_length, bool as_field) {
  using TensorProtoHelper = tensor::internal::TensorProtoHelper<T>;
  using FieldType = typename TensorProtoHelper::FieldType;
  const float kMinCompressionRatio = 2.0;
  const int64 kMinSize = 64;
  TensorProto tensor_proto =
      as_field ? CreateAsProtoField<T>(length, tail_length)
               : CreateAsProtoTensorContent<T>(length, tail_length);
  TensorProto original_tensor_proto = tensor_proto;
  int64 original_size =
      length * (as_field ? (is_complex<T>::value ? 2 : 1) * sizeof(FieldType)
                         : sizeof(T));
  int64 size_as_tensor_content = length * sizeof(T);
  int64 size_as_field = std::min(length, (length - tail_length + 1)) *
                        (is_complex<T>::value ? 2 : 1) * sizeof(FieldType);
  bool will_compress = std::min(size_as_tensor_content, size_as_field) <=
                       static_cast<int64>(original_size / kMinCompressionRatio);

  EXPECT_EQ(tensor::CompressTensorProtoInPlace(kMinSize, kMinCompressionRatio,
                                               &tensor_proto),
            will_compress);
  if (will_compress) {
    if (size_as_tensor_content < size_as_field) {
      EXPECT_EQ(TensorProtoHelper::NumValues(tensor_proto), 0);
      EXPECT_FALSE(tensor_proto.tensor_content().empty());
    } else {
      EXPECT_LE(TensorProtoHelper::NumValues(tensor_proto),
                (length - tail_length + 1));
      EXPECT_TRUE(tensor_proto.tensor_content().empty());
    }
  }
  CompareTensorValues<T>(tensor_proto, original_tensor_proto);
}

TEST(TensorProtoUtil, CompressTensorProtoConstantTail) {
  const int kLength = 64;
  for (bool as_field : {true, false}) {
    for (int tail_length : {0, 1, 2, 32, 33, 63, 64}) {
      ConstantTailTest<float>(kLength, tail_length, as_field);
      ConstantTailTest<double>(kLength, tail_length, as_field);
      ConstantTailTest<complex64>(kLength, tail_length, as_field);
      ConstantTailTest<complex128>(kLength, tail_length, as_field);
      ConstantTailTest<int32>(kLength, tail_length, as_field);
      ConstantTailTest<uint32>(kLength, tail_length, as_field);
      ConstantTailTest<int64>(kLength, tail_length, as_field);
      ConstantTailTest<uint64>(kLength, tail_length, as_field);
      ConstantTailTest<int8>(kLength, tail_length, as_field);
      ConstantTailTest<uint8>(kLength, tail_length, as_field);
      ConstantTailTest<int16>(kLength, tail_length, as_field);
      ConstantTailTest<uint16>(kLength, tail_length, as_field);
      ConstantTailTest<Eigen::half>(kLength, tail_length, as_field);
      ConstantTailTest<bfloat16>(kLength, tail_length, as_field);
    }
  }
}

}  // namespace
}  // namespace tensorflow
