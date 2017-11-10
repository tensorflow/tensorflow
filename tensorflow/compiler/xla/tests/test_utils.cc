/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "tensorflow/compiler/xla/primitive_util.h"

namespace xla {

namespace {

template <typename FloatT>
void PopulateWithRandomFloatingPointData(Literal* literal) {
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<FloatT>());
  std::minstd_rand0 engine;
  std::uniform_real_distribution<FloatT> generator(0.0f, 1.0f);
  TF_CHECK_OK(literal->Populate<FloatT>(
      [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
        return generator(engine);
      }));
}

template <typename IntT>
void PopulateWithRandomIntegralData(Literal* literal) {
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  std::minstd_rand0 engine;
  std::uniform_int_distribution<IntT> generator(
      std::numeric_limits<IntT>::lowest(), std::numeric_limits<IntT>::max());
  TF_CHECK_OK(literal->Populate<IntT>(
      [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
        return generator(engine);
      }));
}

}  // namespace

StatusOr<std::unique_ptr<Literal>> MakeFakeLiteral(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    std::vector<std::unique_ptr<Literal>> elements;
    for (const Shape& element_shape : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> element,
                          MakeFakeLiteral(element_shape));
      elements.push_back(std::move(element));
    }
    return Literal::MakeTupleOwned(std::move(elements));
  }
  std::unique_ptr<Literal> literal = Literal::CreateFromShape(shape);
  switch (shape.element_type()) {
    case F32:
      PopulateWithRandomFloatingPointData<float>(literal.get());
      break;
    case F64:
      PopulateWithRandomFloatingPointData<double>(literal.get());
      break;
    case S8:
      PopulateWithRandomIntegralData<int8>(literal.get());
      break;
    case U8:
      PopulateWithRandomIntegralData<uint8>(literal.get());
      break;
    case S16:
      PopulateWithRandomIntegralData<int16>(literal.get());
      break;
    case U16:
      PopulateWithRandomIntegralData<uint16>(literal.get());
      break;
    case S32:
      PopulateWithRandomIntegralData<int32>(literal.get());
      break;
    case U32:
      PopulateWithRandomIntegralData<uint32>(literal.get());
      break;
    case S64:
      PopulateWithRandomIntegralData<int64>(literal.get());
      break;
    case U64:
      PopulateWithRandomIntegralData<uint64>(literal.get());
      break;
    case PRED: {
      std::uniform_int_distribution<int> generator(0, 1);
      std::minstd_rand0 engine;
      TF_CHECK_OK(literal->Populate<bool>(
          [&](tensorflow::gtl::ArraySlice<int64> /*indices*/) {
            return generator(engine);
          }));
      break;
    }
    default:
      return Unimplemented("Unsupported type for fake literal generation: %s",
                           ShapeUtil::HumanString(shape).c_str());
  }
  return std::move(literal);
}

StatusOr<std::vector<std::unique_ptr<Literal>>> MakeFakeArguments(
    const HloModule& module) {
  std::vector<std::unique_ptr<Literal>> arguments;
  for (const ShapeLayout& shape_layout :
       module.config().entry_computation_layout().parameter_layouts()) {
    TF_ASSIGN_OR_RETURN(auto literal, MakeFakeLiteral(shape_layout.shape()));
    arguments.push_back(std::move(literal));
  }
  return std::move(arguments);
}

}  // namespace xla
