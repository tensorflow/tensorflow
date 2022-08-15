/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/custom_call.h"

#include <string>
#include <string_view>

#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

using llvm::raw_ostream;

static void PrintArr(raw_ostream& os, std::string_view name,
                     llvm::ArrayRef<int64_t> arr) {
  os << " " << name << ": [";
  auto i64_to_string = [](int64_t v) { return std::to_string(v); };
  os << llvm::join(llvm::map_range(arr, i64_to_string), ", ");
  os << "]";
}

raw_ostream& operator<<(raw_ostream& os, const StridedMemrefView& view) {
  os << "StridedMemrefView: dtype: " << view.dtype;
  PrintArr(os, "sizes", view.sizes);
  PrintArr(os, "strides", view.strides);
  return os;
}

raw_ostream& operator<<(raw_ostream& os, const MemrefView& view) {
  os << "MemrefView: dtype: " << view.dtype;
  PrintArr(os, "sizes", view.sizes);
  return os;
}

raw_ostream& operator<<(raw_ostream& os, const FlatMemrefView& view) {
  return os << "FlatMemrefView: dtype: " << view.dtype
            << " size_in_bytes: " << view.size_in_bytes;
}

XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(uint8_t, "__type_id_uint8");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(uint32_t, "__type_id_uint32");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(uint64_t, "__type_id_uint64");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(bool, "__type_id_bool");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(int32_t, "__type_id_int32");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(int64_t, "__type_id_int64");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(Eigen::half,
                                            "__type_id_eigen_half");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(float, "__type_id_float");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(double, "__type_id_double");

XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::StringRef,
                                            "__type_id_string");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(MemrefView,
                                            "__type_id_memref_view");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(StridedMemrefView,
                                            "__type_id_strided_memref_view");

XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(EmptyArrayRef,
                                            "__type_id_empty_arrayref");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<int8_t>,
                                            "__type_id_arrayref_int8");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<int16_t>,
                                            "__type_id_arrayref_int16");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<int32_t>,
                                            "__type_id_arrayref_int32");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<int64_t>,
                                            "__type_id_arrayref_int64");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<float>,
                                            "__type_id_arrayref_float");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(llvm::ArrayRef<double>,
                                            "__type_id_arrayref_double");

XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(
    CustomCall::TensorRef<int32_t>, "__type_id_customcall_tensorref_int32_t");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(
    CustomCall::TensorRef<int64_t>, "__type_id_customcall_tensorref_int64_t");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(
    CustomCall::TensorRef<float>, "__type_id_customcall_tensorref_float");
XLA_RUNTIME_STATIC_TYPEID_NAME_REGISTRATION(
    CustomCall::TensorRef<double>, "__type_id_customcall_tensorref_double");

}  // namespace runtime
}  // namespace xla

XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(llvm::StringRef);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(xla::runtime::StridedMemrefView);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(xla::runtime::MemrefView);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(xla::runtime::FlatMemrefView);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(int32_t);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(int64_t);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(float);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(double);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<int32_t>);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<int64_t>);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<float>);
XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(ArrayRef<double>);
