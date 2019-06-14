/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/memory/memory.h"

#include <string>
#include <vector>

namespace xla {
namespace poplarplugin {
namespace {
template <typename TOutput, typename TValue>
void ConvertToType(char*& buffer, TValue value) {
  reinterpret_cast<TOutput*>(buffer)[0] = static_cast<TOutput>(value);
}

template <typename TValue>
void PutValueIntoBuffer(char*& buffer, const PrimitiveType type, TValue value) {
  switch (type) {
#define CONVERT_TO_TYPE(XLA_TYPE, NATIVE_TYPE) \
  case XLA_TYPE: {                             \
    ConvertToType<NATIVE_TYPE>(buffer, value); \
    break;                                     \
  }
    CONVERT_TO_TYPE(PRED, bool)
    CONVERT_TO_TYPE(S8, int8)
    CONVERT_TO_TYPE(U8, uint8)
    CONVERT_TO_TYPE(S16, int16)
    CONVERT_TO_TYPE(U16, uint16)
    CONVERT_TO_TYPE(S32, int32)
    CONVERT_TO_TYPE(U32, uint32)
    CONVERT_TO_TYPE(S64, int64)
    CONVERT_TO_TYPE(U64, uint64)
    CONVERT_TO_TYPE(F32, float)
    CONVERT_TO_TYPE(F16, Eigen::half)
#undef CONVERT_TO_TYPE
    default:
      LOG(FATAL) << "Unsupported primitive type " << type;
      break;
  }
}

void PutRandomUniformValueIntoBuffer(char*& buffer, const PrimitiveType type,
                                     std::mt19937& generator) {
  switch (type) {
    case PRED: {
      std::uniform_int_distribution<int8> dist(0, 1);
      PutValueIntoBuffer(buffer, type, dist(generator));
      break;
    }
#define GET_RANDOM_INT(XLA_TYPE, NATIVE_TYPE)                  \
  case XLA_TYPE: {                                             \
    auto max = std::numeric_limits<NATIVE_TYPE>::max();        \
    auto min = std::numeric_limits<NATIVE_TYPE>::min();        \
    std::uniform_int_distribution<NATIVE_TYPE> dist(min, max); \
    PutValueIntoBuffer(buffer, type, dist(generator));         \
    break;                                                     \
  }
      GET_RANDOM_INT(S8, int8)
      GET_RANDOM_INT(U8, uint8)
      GET_RANDOM_INT(S16, int16)
      GET_RANDOM_INT(U16, uint16)
      GET_RANDOM_INT(S32, int32)
      GET_RANDOM_INT(U32, uint32)
      GET_RANDOM_INT(S64, int64)
      GET_RANDOM_INT(U64, uint64)
#undef GET_RANDOM_INT
    case F16:
    case F32: {
      float max;
      float min;
      if (type == F16) {
        max = Eigen::half_impl::half_to_float(
            std::numeric_limits<Eigen::half>::max());
        min = Eigen::half_impl::half_to_float(
            std::numeric_limits<Eigen::half>::min());
      } else {
        max = std::numeric_limits<float>::max();
        min = std::numeric_limits<float>::min();
      }
      std::uniform_real_distribution<float> dist(min, max);
      PutValueIntoBuffer(buffer, type, dist(generator));
      break;
    }
    default:
      LOG(FATAL) << "Unsupported primitive type " << type;
      break;
  }
}

void PutRandomNormalValueIntoBuffer(char*& buffer, const PrimitiveType type,
                                    std::mt19937& generator) {
  switch (type) {
    case PRED:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64: {
      PutValueIntoBuffer(buffer, type, 1);
      break;
    }
    case F16:
    case F32: {
      std::normal_distribution<float> dist(0, 1);
      PutValueIntoBuffer(buffer, type, dist(generator));
      break;
    }
    default:
      LOG(FATAL) << "Unsupported primitive type " << type;
      break;
  }
}
}  // namespace

DataInitializer::DataInitializer(const std::string& type_string)
    : type_string_(type_string){};

std::unique_ptr<DataInitializer> DataInitializer::GetDataInitializer(
    const std::string& type_string) {
  if (type_string == "random" || type_string == "uniform") {
    return absl::make_unique<RandomDataInitializer>(type_string,
                                                    RandomType::UNIFORM);
  } else if (type_string == "normal") {
    return absl::make_unique<RandomDataInitializer>(type_string,
                                                    RandomType::NORMAL);
  } else {
    return absl::make_unique<ConstantDataInitializer>(type_string);
  }
}

DataInitializer& DataInitializer::GetSyntheticDataInitializer() {
  static auto initializer =
      GetDataInitializer(PoplarXlaFlags::Get().synthetic_data_initializer);
  return *initializer;
}

StatusOr<Literal> DataInitializer::GetData(const Shape& shape) {
  auto type = shape.element_type();
  auto flat_shape = ShapeUtil::MakeShape(type, {ShapeUtil::ElementsIn(shape)});
  auto flat_literal = Literal(flat_shape);
  char* dest_data = static_cast<char*>(flat_literal.untyped_data());
  const int64 primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(type);

  char* raw_ptr = new char[primitive_size];
  ShapeUtil::ForEachIndex(flat_shape,
                          [&](absl::Span<const int64> output_index) {
                            CHECK_EQ(output_index.size(), 1);
                            GetValue(raw_ptr, type);
                            memcpy(dest_data + primitive_size * output_index[0],
                                   raw_ptr, primitive_size);
                            return true;
                          });
  delete[] raw_ptr;
  TF_ASSIGN_OR_RETURN(auto literal, flat_literal.Reshape(shape.dimensions()));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Literal created is: ";
    VLOG(2) << literal.ToString();
  }
  return literal;
}

RandomDataInitializer::RandomDataInitializer(const std::string& type_string,
                                             RandomType random_type)
    : DataInitializer(type_string),
      random_type_(random_type),
      generator_(random_device_()) {
  VLOG(1) << "Created RandomDataInitializer given \"type_string\"="
          << type_string << ".";
};

void RandomDataInitializer::GetValue(char*& buffer, const PrimitiveType& type) {
  if (random_type_ == RandomType::UNIFORM) {
    PutRandomUniformValueIntoBuffer(buffer, type, generator_);
  } else {
    PutRandomNormalValueIntoBuffer(buffer, type, generator_);
  }
}

ConstantDataInitializer::ConstantDataInitializer(const std::string& type_string)
    : DataInitializer(type_string), value_(std::stoi(type_string)) {
  VLOG(1) << "Created ConstantDataInitializer with value " << value_
          << " given \"type_string\"=" << type_string << ".";
};

void ConstantDataInitializer::GetValue(char*& buffer,
                                       const PrimitiveType& type) {
  PutValueIntoBuffer(buffer, type, value_);
}

}  // namespace poplarplugin
}  // namespace xla
