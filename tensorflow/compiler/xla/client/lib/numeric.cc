/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/numeric.h"

#include <numeric>
#include <vector>

namespace xla {

namespace {

template <typename T>
XlaOp MakeIota(XlaBuilder* builder, int64 size) {
  std::vector<T> values(size);
  for (int64 i = 0; i < size; ++i) {
    values[i] = static_cast<T>(i);
  }
  return xla::ConstantR1<T>(builder, values);
}

}  // namespace

XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64 size) {
  switch (type) {
    case S8:
      return MakeIota<int8>(builder, size);
    case S16:
      return MakeIota<int16>(builder, size);
    case S32:
      return MakeIota<int32>(builder, size);
    case S64:
      return MakeIota<int64>(builder, size);
    case U8:
      return MakeIota<uint8>(builder, size);
    case U16:
      return MakeIota<uint16>(builder, size);
    case U32:
      return MakeIota<uint32>(builder, size);
    case U64:
      return MakeIota<uint64>(builder, size);
    case BF16:
      return MakeIota<bfloat16>(builder, size);
    case F16:
      return MakeIota<half>(builder, size);
    case F32:
      return MakeIota<float>(builder, size);
    case F64:
      return MakeIota<double>(builder, size);
    case C64:
      return MakeIota<complex64>(builder, size);
    default:
      return builder->ReportError(
          InvalidArgument("Unimplemented type for Iota: %s.",
                          PrimitiveType_Name(type).c_str()));
  }
}

}  // namespace xla
