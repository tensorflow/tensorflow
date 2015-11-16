#include "tensorflow/core/framework/tensor_util.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"

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
  str1.flat<string>()(0) = "foo1";
  str1.flat<string>()(1) = "foo2";
  Tensor str2 = tensor::DeepCopy(str1);
  str2.flat<string>()(0) = "bar1";
  str2.flat<string>()(1) = "bar2";
  EXPECT_NE(str2.flat<string>()(0), str1.flat<string>()(0));
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

}  // namespace
}  // namespace tensorflow
