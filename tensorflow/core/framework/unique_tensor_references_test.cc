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

#include "tensorflow/core/framework/unique_tensor_references.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(UniquifyTensors, TestUniqueVector) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(DT_FLOAT, TensorShape({2, 2}));

  EXPECT_FALSE(a.SharesBufferWith(b));

  refs.Add(a);
  refs.Add(b);
  TensorReferenceVector tensors;
  refs.FreezeAndReturnReferences(&tensors);
  EXPECT_EQ(2, tensors.size());
  if (tensors[0].SharesBufferWith(a)) {
    EXPECT_TRUE(tensors[1].SharesBufferWith(b));
  } else {
    EXPECT_TRUE(tensors[1].SharesBufferWith(a));
    EXPECT_TRUE(tensors[0].SharesBufferWith(b));
  }
  for (auto& t : tensors) {
    t.Unref();
  }
}

TEST(UniquifyTensors, TestNonUniqueVector) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(a);

  EXPECT_TRUE(a.SharesBufferWith(b));

  refs.Add(a);
  refs.Add(b);
  TensorReferenceVector tensors;
  refs.FreezeAndReturnReferences(&tensors);
  EXPECT_EQ(1, tensors.size());
  EXPECT_TRUE(tensors[0].SharesBufferWith(a));
  EXPECT_TRUE(tensors[0].SharesBufferWith(b));
  for (auto& t : tensors) {
    t.Unref();
  }
}

TEST(UniquifyTensors, TestNoLeakVector) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(DT_FLOAT, TensorShape({2, 2}));

  EXPECT_FALSE(a.SharesBufferWith(b));

  refs.Add(a);
  refs.Add(b);
}

TEST(UniquifyTensors, TestUniqueSet) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(DT_FLOAT, TensorShape({2, 2}));
  Tensor c(DT_FLOAT, TensorShape({2, 2}));
  Tensor d(DT_FLOAT, TensorShape({2, 2}));
  Tensor e(DT_FLOAT, TensorShape({2, 2}));

  EXPECT_FALSE(a.SharesBufferWith(b));

  refs.Add(a);
  refs.Add(b);
  refs.Add(c);
  refs.Add(d);
  refs.Add(e);
  TensorReferenceVector tensors;
  refs.FreezeAndReturnReferences(&tensors);
  EXPECT_EQ(5, tensors.size());
  for (auto& t : tensors) {
    t.Unref();
  }
}

TEST(UniquifyTensors, TestNonUniqueSet) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(DT_FLOAT, TensorShape({2, 2}));
  Tensor c(DT_FLOAT, TensorShape({2, 2}));
  Tensor d(DT_FLOAT, TensorShape({2, 2}));
  Tensor e(DT_FLOAT, TensorShape({2, 2}));
  Tensor f(c);

  EXPECT_TRUE(f.SharesBufferWith(c));

  refs.Add(a);
  refs.Add(b);
  refs.Add(c);
  refs.Add(d);
  refs.Add(e);
  refs.Add(f);
  TensorReferenceVector tensors;
  refs.FreezeAndReturnReferences(&tensors);
  EXPECT_EQ(5, tensors.size());
  for (auto& t : tensors) {
    t.Unref();
  }
}

TEST(UniquifyTensors, TestNoLeakSet) {
  UniqueTensorReferences refs;
  Tensor a(DT_FLOAT, TensorShape({2, 2}));
  Tensor b(DT_FLOAT, TensorShape({2, 2}));
  Tensor c(DT_FLOAT, TensorShape({2, 2}));
  Tensor d(DT_FLOAT, TensorShape({2, 2}));
  Tensor e(DT_FLOAT, TensorShape({2, 2}));

  refs.Add(a);
  refs.Add(b);
  refs.Add(c);
  refs.Add(d);
  refs.Add(e);
}

}  // namespace tensorflow
