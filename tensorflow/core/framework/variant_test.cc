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

#include <vector>

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

template <typename T>
struct Wrapper {
  T value;
  string TypeName() const { return "POD"; }
};

using Int = Wrapper<int>;
using Float = Wrapper<float>;

}  // end namespace

TEST(VariantTest, Int) {
  Variant x;
  EXPECT_EQ(x.MaybeDecodeAndGet<void>(), nullptr);
  x = 3;
  EXPECT_NE(x.MaybeDecodeAndGet<void>(), nullptr);
  EXPECT_EQ(*x.MaybeDecodeAndGet<int>(), 3);
  EXPECT_EQ(x.TypeName(), "int");
}

TEST(VariantTest, Basic) {
  Variant x;
  EXPECT_EQ(x.MaybeDecodeAndGet<void>(), nullptr);

  x = Int{42};

  EXPECT_NE(x.MaybeDecodeAndGet<void>(), nullptr);
  EXPECT_NE(x.MaybeDecodeAndGet<Int>(), nullptr);
  EXPECT_EQ(x.MaybeDecodeAndGet<Int>()->value, 42);
  EXPECT_EQ(x.TypeName(), "POD");
}

TEST(VariantTest, ConstGet) {
  Variant x;
  EXPECT_EQ(x.MaybeDecodeAndGet<void>(), nullptr);

  x = Int{42};

  const Variant y = x;

  EXPECT_NE(y.MaybeDecodeAndGet<void>(), nullptr);
  EXPECT_NE(y.MaybeDecodeAndGet<Int>(), nullptr);
  EXPECT_EQ(y.MaybeDecodeAndGet<Int>()->value, 42);
}

TEST(VariantTest, Clear) {
  Variant x;
  EXPECT_EQ(x.MaybeDecodeAndGet<void>(), nullptr);

  x = Int{42};

  EXPECT_NE(x.MaybeDecodeAndGet<void>(), nullptr);
  EXPECT_NE(x.MaybeDecodeAndGet<Int>(), nullptr);
  EXPECT_EQ(x.MaybeDecodeAndGet<Int>()->value, 42);

  x.clear();
  EXPECT_EQ(x.MaybeDecodeAndGet<void>(), nullptr);
}

TEST(VariantTest, Tensor) {
  Variant x;
  Tensor t(DT_FLOAT, {});
  t.flat<float>()(0) = 42.0f;
  x = t;

  EXPECT_NE(x.MaybeDecodeAndGet<Tensor>(), nullptr);
  EXPECT_EQ(x.MaybeDecodeAndGet<Tensor>()->flat<float>()(0), 42.0f);
  x.MaybeDecodeAndGet<Tensor>()->flat<float>()(0) += 1.0f;
  EXPECT_EQ(x.MaybeDecodeAndGet<Tensor>()->flat<float>()(0), 43.0f);
  EXPECT_EQ(x.TypeName(), "tensorflow::Tensor");
}

TEST(VariantTest, TensorProto) {
  Variant x;
  TensorProto t;
  t.set_dtype(DT_FLOAT);
  t.mutable_tensor_shape()->set_unknown_rank(true);
  x = t;

  EXPECT_EQ(x.TypeName(), "tensorflow.TensorProto");
  EXPECT_NE(x.MaybeDecodeAndGet<TensorProto>(), nullptr);
  EXPECT_EQ(x.MaybeDecodeAndGet<TensorProto>()->dtype(), DT_FLOAT);
  EXPECT_EQ(x.MaybeDecodeAndGet<TensorProto>()->tensor_shape().unknown_rank(),
            true);
}

TEST(VariantTest, CopyValue) {
  Variant x, y;
  x = Int{10};
  y = x;

  EXPECT_EQ(x.MaybeDecodeAndGet<Int>()->value, 10);
  EXPECT_EQ(x.MaybeDecodeAndGet<Int>()->value,
            y.MaybeDecodeAndGet<Int>()->value);
}

TEST(VariantTest, MoveValue) {
  Variant x;
  x = []() -> Variant {
    Variant y;
    y = Int{10};
    return y;
  }();
  EXPECT_EQ(x.MaybeDecodeAndGet<Int>()->value, 10);
}

TEST(VariantTest, TypeMismatch) {
  Variant x;
  x = Int{10};
  EXPECT_EQ(x.MaybeDecodeAndGet<float>(), nullptr);
  EXPECT_EQ(x.MaybeDecodeAndGet<int>(), nullptr);
  EXPECT_NE(x.MaybeDecodeAndGet<Int>(), nullptr);
}

struct TensorList {
  void Encode(VariantTensorData* data) const { data->tensors_ = vec; }

  bool Decode(const VariantTensorData& data) {
    vec = data.tensors_;
    return true;
  }

  string TypeName() const { return "TensorList"; }

  std::vector<Tensor> vec;
};

TEST(VariantTest, AccessingVariantTensorDataFails) {
  const Variant x = VariantTensorDataProto();
  EXPECT_DEATH(x.MaybeDecodeAndGet<int>(),
               "Cannot call MaybeDecodeAndGet on const Variant holding "
               "serialized data.");
}

TEST(VariantTest, TensorListTest) {
  Variant x;

  TensorList vec;
  for (int i = 0; i < 4; ++i) {
    Tensor elem(DT_INT32, {1});
    elem.flat<int>()(0) = i;
    vec.vec.push_back(elem);
  }

  for (int i = 0; i < 4; ++i) {
    Tensor elem(DT_FLOAT, {1});
    elem.flat<float>()(0) = 2 * i;
    vec.vec.push_back(elem);
  }

  x = vec;

  EXPECT_EQ(x.TypeName(), "TensorList");
  const TensorList& stored_vec = *x.MaybeDecodeAndGet<TensorList>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(stored_vec.vec[i].flat<int>()(0), i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(stored_vec.vec[i + 4].flat<float>()(0), 2 * i);
  }

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y = TensorList();
  y.Decode(serialized);

  const TensorList& decoded_vec = *y.MaybeDecodeAndGet<TensorList>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(decoded_vec.vec[i].flat<int>()(0), i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(decoded_vec.vec[i + 4].flat<float>()(0), 2 * i);
  }

  VariantTensorDataProto data;
  serialized.ToProto(&data);
  Variant y_unknown = data;
  EXPECT_EQ(y_unknown.TypeName(), "TensorList");
  EXPECT_EQ(y_unknown.TypeId(), MakeTypeIndex<VariantTensorDataProto>());

  // Now call a get that internally performs a decode.
  const TensorList& unknown_decoded_vec =
      *y_unknown.MaybeDecodeAndGet<TensorList>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(unknown_decoded_vec.vec[i].flat<int>()(0), i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(unknown_decoded_vec.vec[i + 4].flat<float>()(0), 2 * i);
  }

  // After the MaybeDecodeAndGet<>() which decoded the DataProto to a
  // TensorList, we can no longer access the original data proto
  EXPECT_EQ(y_unknown.TypeId(), MakeTypeIndex<TensorList>());
}

TEST(VariantTest, VariantArray) {
  Variant x[2];
  x[0] = Int{2};
  x[1] = Float{2.0f};

  EXPECT_EQ(x[0].MaybeDecodeAndGet<Int>()->value, 2);
  EXPECT_EQ(x[1].MaybeDecodeAndGet<Float>()->value, 2.0f);
}

TEST(VariantTest, PodUpdate) {
  struct Pod {
    int x;
    float y;

    string TypeName() const { return "POD"; }
  };

  Variant x = Pod{10, 20.f};
  EXPECT_NE(x.MaybeDecodeAndGet<Pod>(), nullptr);
  EXPECT_EQ(x.TypeName(), "POD");

  x.MaybeDecodeAndGet<Pod>()->x += x.MaybeDecodeAndGet<Pod>()->y;
  EXPECT_EQ(x.MaybeDecodeAndGet<Pod>()->x, 30);
}

TEST(VariantTest, EncodeDecodePod) {
  struct Pod {
    int x;
    float y;

    string TypeName() const { return "POD"; }
  };

  Variant x;
  Pod p{10, 20.0f};
  x = p;

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y;
  y = Pod();
  y.Decode(serialized);

  EXPECT_EQ(p.x, y.MaybeDecodeAndGet<Pod>()->x);
  EXPECT_EQ(p.y, y.MaybeDecodeAndGet<Pod>()->y);
}

TEST(VariantTest, EncodeDecodeTensor) {
  Variant x;
  Tensor t(DT_INT32, {});
  t.flat<int>()(0) = 42;
  x = t;

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y = Tensor();
  y.Decode(serialized);
  EXPECT_EQ(x.MaybeDecodeAndGet<Tensor>()->flat<int>()(0),
            y.MaybeDecodeAndGet<Tensor>()->flat<int>()(0));
}

}  // end namespace tensorflow
