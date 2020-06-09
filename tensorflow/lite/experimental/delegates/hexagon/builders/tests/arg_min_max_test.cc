/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class ArgBaseOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ArgBaseOpModel(TensorType input_type) {
    input_ = AddInput(input_type);
    output_ = AddOutput(TensorType_INT32);
  }

  int input() const { return input_; }

  std::vector<int> GetInt32Output() const {
    return ExtractVector<int>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  using SingleOpModelWithHexagon::builder_;

  int input_;
  int output_;
};

class ArgMinOpModel : public ArgBaseOpModel {
 public:
  ArgMinOpModel(std::initializer_list<int> input_shape, TensorType input_type)
      : ArgBaseOpModel(input_type /*input_type*/), input_shape_(input_shape) {}

  void Build() {
    SetBuiltinOp(BuiltinOperator_ARG_MIN, BuiltinOptions_ArgMinOptions,
                 CreateArgMinOptions(builder_, TensorType_INT32 /*output_type*/)
                     .Union());
    BuildInterpreter({input_shape_, {1}});
  }

 private:
  std::vector<int> input_shape_;
};

class ArgMaxOpModel : public ArgBaseOpModel {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type)
      : ArgBaseOpModel(input_type /*input_type*/), input_shape_(input_shape) {}

  void Build() {
    SetBuiltinOp(BuiltinOperator_ARG_MAX, BuiltinOptions_ArgMaxOptions,
                 CreateArgMaxOptions(builder_, TensorType_INT32 /*output_type*/)
                     .Union());
    BuildInterpreter({input_shape_, {1}});
  }

 private:
  std::vector<int> input_shape_;
};

template <typename integer_type, TensorType tensor_dtype>
void ArgMinTestImpl() {
  ArgMinOpModel model({1, 1, 1, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {3}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

template <typename integer_type, TensorType tensor_dtype>
void ArgMinNegativeTestImpl() {
  ArgMinOpModel model({1, 1, 2, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {-2}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(),
                                             {1, 2, 7, 8, 1, 9, 7, 3});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({0, 0, 0, 1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 4}));
}

template <typename integer_type, TensorType tensor_dtype>
void ArgMaxTestImpl() {
  ArgMaxOpModel model({1, 1, 1, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {3}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({3}));
}

TEST(ArgMinTest, GetArgMin_UInt8) {
  ArgMinTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMinTest, GetArgMin_Int8) { ArgMinTestImpl<int8_t, TensorType_INT8>(); }

TEST(ArgMinTest, GetArgMinNegative_UInt8) {
  ArgMinNegativeTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMinTest, GetArgMinNegative_Int8) {
  ArgMinNegativeTestImpl<int8_t, TensorType_INT8>();
}

TEST(ArgMaxTest, GetArgMax_UInt8) {
  ArgMaxTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMaxTest, GetArgMax_Int8) { ArgMaxTestImpl<int8_t, TensorType_INT8>(); }
}  // namespace tflite
