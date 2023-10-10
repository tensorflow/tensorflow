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

#include <complex>
#include <string>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace {

class SaveV2OpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("myop", "SaveV2")
                     .Input(FakeInput())  // prefix
                     .Input(FakeInput())  // tensor_names
                     .Input(FakeInput())  // shape_and_slices
                     .Input(FakeInput({DT_BOOL, DT_INT32, DT_FLOAT, DT_DOUBLE,
                                       DT_QINT8, DT_QINT32, DT_UINT8, DT_INT8,
                                       DT_INT16, DT_INT64, DT_COMPLEX64,
                                       DT_COMPLEX128, DT_HALF}))  // tensors
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SaveV2OpTest, Simple) {
  const string prefix = io::JoinPath(testing::TmpDir(), "tensor_simple");
  const string tensornames[] = {
      "tensor_bool",  "tensor_int",    "tensor_float",     "tensor_double",
      "tensor_qint8", "tensor_qint32", "tensor_uint8",     "tensor_int8",
      "tensor_int16", "tensor_int64",  "tensor_complex64", "tensor_complex128",
      "tensor_half"};

  MakeOp();
  // Add a file name
  AddInput<tstring>(TensorShape({}),
                    [&prefix](int x) -> tstring { return prefix; });

  // Add the tensor names
  AddInput<tstring>(TensorShape({13}), [&tensornames](int x) -> tstring {
    return tensornames[x];
  });

  // Add the slice specs
  AddInput<tstring>(TensorShape({13}),
                    [](int x) -> tstring { return "" /* saves in full */; });

  // Add a 1-d bool tensor
  AddInput<bool>(TensorShape({2}), [](int x) -> bool { return x != 0; });

  // Add a 1-d integer tensor
  AddInput<int32>(TensorShape({10}), [](int x) -> int32 { return x + 1; });

  // Add a 2-d float tensor
  AddInput<float>(TensorShape({2, 4}),
                  [](int x) -> float { return static_cast<float>(x) / 10; });

  // Add a 2-d double tensor
  AddInput<double>(TensorShape({2, 4}),
                   [](int x) -> double { return static_cast<double>(x) / 20; });

  // Add a 2-d qint8 tensor
  AddInput<qint8>(TensorShape({3, 2}),
                  [](int x) -> qint8 { return *reinterpret_cast<qint8*>(&x); });

  // Add a 2-d qint32 tensor
  AddInput<qint32>(TensorShape({2, 3}), [](int x) -> qint32 {
    return *reinterpret_cast<qint32*>(&x) * qint8(2);
  });

  // Add a 1-d uint8 tensor
  AddInput<uint8>(TensorShape({11}), [](int x) -> uint8 { return x + 1; });

  // Add a 1-d int8 tensor
  AddInput<int8>(TensorShape({7}), [](int x) -> int8 { return x - 7; });

  // Add a 1-d int16 tensor
  AddInput<int16>(TensorShape({7}), [](int x) -> int16 { return x - 8; });

  // Add a 1-d int64 tensor
  AddInput<int64_t>(TensorShape({9}), [](int x) -> int64 { return x - 9; });

  // Add a 2-d complex64 tensor
  AddInput<complex64>(TensorShape({2, 3}), [](int x) -> complex64 {
    return complex64(100 + x, 200 + x);
  });

  // Add a 2-d complex128 tensor
  AddInput<complex128>(TensorShape({2, 3}), [](int x) -> complex128 {
    return complex128(100 + x, 200 + x);
  });

  // Add a 2-d half tensor
  AddInput<Eigen::half>(TensorShape({2, 4}), [](int x) -> Eigen::half {
    return static_cast<Eigen::half>(x) / Eigen::half(2);
  });
  TF_ASSERT_OK(RunOpKernel());

  // Check that the checkpoint file is properly written
  BundleReader reader(Env::Default(), prefix);
  TF_EXPECT_OK(reader.status());

  // We expect to find all saved tensors
  {
    // The 1-d bool tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_bool", &shape));
    TensorShape expected({2});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_bool", &val));
    EXPECT_EQ(DT_BOOL, val.dtype());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ((i != 0), val.template flat<bool>()(i));
    }
  }

  {
    // The 1-d integer tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_int", &shape));
    TensorShape expected({10});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_int", &val));
    EXPECT_EQ(DT_INT32, val.dtype());
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i + 1, val.template flat<int>()(i));
    }
  }

  {
    // The 2-d float tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_float", &shape));
    TensorShape expected({2, 4});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_float", &val));
    EXPECT_EQ(DT_FLOAT, val.dtype());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(static_cast<float>(i) / 10, val.template flat<float>()(i));
    }
  }

  {
    // The 2-d double tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_double", &shape));
    TensorShape expected({2, 4});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_double", &val));
    EXPECT_EQ(DT_DOUBLE, val.dtype());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(static_cast<double>(i) / 20, val.template flat<double>()(i));
    }
  }

  {
    // The 2-d qint8 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_qint8", &shape));
    TensorShape expected({3, 2});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_qint8", &val));
    EXPECT_EQ(DT_QINT8, val.dtype());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(*reinterpret_cast<qint8*>(&i), val.template flat<qint8>()(i));
    }
  }

  {
    // The 2-d qint32 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_qint32", &shape));
    TensorShape expected({2, 3});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_qint32", &val));
    EXPECT_EQ(DT_QINT32, val.dtype());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(*reinterpret_cast<qint32*>(&i) * qint8(2),
                val.template flat<qint32>()(i));
    }
  }

  {
    // The 1-d uint8 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_uint8", &shape));
    TensorShape expected({11});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_uint8", &val));
    EXPECT_EQ(DT_UINT8, val.dtype());
    for (int i = 0; i < 11; ++i) {
      EXPECT_EQ(i + 1, val.template flat<uint8>()(i));
    }
  }

  {
    // The 1-d int8 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_int8", &shape));
    TensorShape expected({7});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_int8", &val));
    EXPECT_EQ(DT_INT8, val.dtype());
    for (int i = 0; i < 7; ++i) {
      EXPECT_EQ(i - 7, val.template flat<int8>()(i));
    }
  }

  {
    // The 1-d int16 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_int16", &shape));
    TensorShape expected({7});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_int16", &val));
    EXPECT_EQ(DT_INT16, val.dtype());
    for (int i = 0; i < 7; ++i) {
      EXPECT_EQ(i - 8, val.template flat<int16>()(i));
    }
  }

  {
    // The 1-d int64 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_int64", &shape));
    TensorShape expected({9});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_int64", &val));
    EXPECT_EQ(DT_INT64, val.dtype());
    for (int i = 0; i < 9; ++i) {
      EXPECT_EQ(i - 9, val.template flat<int64_t>()(i));
    }
  }

  {
    // The 2-d complex64 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_complex64", &shape));
    TensorShape expected({2, 3});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_complex64", &val));
    EXPECT_EQ(DT_COMPLEX64, val.dtype());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(100 + i, val.template flat<complex64>()(i).real());
      EXPECT_EQ(200 + i, val.template flat<complex64>()(i).imag());
    }
  }

  {
    // The 2-d complex128 tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_complex128", &shape));
    TensorShape expected({2, 3});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_complex128", &val));
    EXPECT_EQ(DT_COMPLEX128, val.dtype());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(100 + i, val.template flat<complex128>()(i).real());
      EXPECT_EQ(200 + i, val.template flat<complex128>()(i).imag());
    }
  }
  {
    // The 2-d half tensor
    TensorShape shape;
    TF_EXPECT_OK(reader.LookupTensorShape("tensor_half", &shape));
    TensorShape expected({2, 4});
    EXPECT_TRUE(shape.IsSameSize(expected));

    // We expect the tensor value to be correct.
    Tensor val;
    TF_EXPECT_OK(reader.Lookup("tensor_half", &val));
    EXPECT_EQ(DT_HALF, val.dtype());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(static_cast<Eigen::half>(i) / Eigen::half(2),
                val.template flat<Eigen::half>()(i));
    }
  }
}

}  // namespace
}  // namespace tensorflow
