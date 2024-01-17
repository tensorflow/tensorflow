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

#ifndef XLA_RUNTIME_RESULTS_H_
#define XLA_RUNTIME_RESULTS_H_

#include <type_traits>

#include "xla/runtime/logical_result.h"
#include "xla/runtime/types.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Conversions from XLA executable results to C++ types.
//===----------------------------------------------------------------------===//

// The result type defines its own ABI as a required number of bytes and
// alignment, and executable returns results by writing into the requested
// memory allocated in the call frame. The user is responsible for providing
// a conversion function that converts this opaque memory back to the C++
// data type. For example memrefs returned as a `StridedMemrefType` structure,
// and it is the user responsibiity to define a conversion function that can
// convert a memref to the run time Tensor/Buffer type.
//
// It is important that the type that is written into the call frame memory has
// a standard memory layout, because we rely on `reinterpret_cast` to reinterpet
// the opaque bytes to a C struct.
//
// See https://en.cppreference.com/w/cpp/types/is_standard_layout

// Result converter is responsible for taking a pointer to the memory location
// where the executable wrote the result, and converting it to the corresponding
// run time value expected by the caller (e.g. memref descriptor to Tensor).
class ResultConverter {
 public:
  virtual ~ResultConverter() = default;

  // Converts value `ret` of type `runtime_type` (runtime type derived from the
  // original `type`) returned from the executable at `result_index` result
  // position using registered conversion functions. Returns a logical result
  // telling if the conversion was successful.
  virtual LogicalResult ReturnValue(unsigned result_index, const Type* type,
                                    const Type* runtime_type,
                                    void* ret) const = 0;

  // Returns error for all results. This function is called if the runtime
  // fails to run the executable, or if the executable returns an error.
  //
  // This function is not called if an individual `ReturnValue` conversion
  // fails.
  //
  // It is the user's responsibility to handle the case where the return value
  // conversion fails, and some error value has to be returned for unhandled
  // results, e.g. error async value for unconverted results.
  virtual void ReturnError(const absl::Status& error) const = 0;
};

//===----------------------------------------------------------------------===//
// Result converter for functions without results (returning void).
//===----------------------------------------------------------------------===//

struct NoResultConverter : public ResultConverter {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult ReturnValue(unsigned, const Type*, const Type*,
                            void*) const final {
    assert(false && "no result converter must never be called");
    return failure();
  }

  void ReturnError(const absl::Status&) const final {}
};

//===----------------------------------------------------------------------===//
// Returns results using user-provided set of conversion functions.
//===----------------------------------------------------------------------===//

template <typename RetError, typename... RetValue>
class ResultConverterSet : public ResultConverter {
  static_assert(sizeof...(RetValue), "result converters must be non-empty");

  static_assert(std::is_invocable_v<RetError, const absl::Status&>);

  static_assert(std::conjunction_v<
                std::is_invocable_r<LogicalResult, RetValue, unsigned,
                                    const Type*, const Type*, void*>...>);

 public:
  explicit ResultConverterSet(RetError ret_error, RetValue... ret_value)
      : ret_error_(std::move(ret_error)),
        ret_value_(std::forward<RetValue>(ret_value)...) {}

  LogicalResult ReturnValue(unsigned result_index, const Type* type,
                            const Type* runtime_type, void* ret) const final {
    return ReturnValue<0>(result_index, type, runtime_type, ret);
  }

  void ReturnError(const absl::Status& error) const final { ret_error_(error); }

 private:
  template <size_t idx>
  LogicalResult ReturnValue(unsigned result_index, const Type* type,
                            const Type* runtime_type, void* ret) const {
    // Try to call the user-provided converter.
    auto& converter = std::get<idx>(ret_value_);
    if (succeeded(converter(result_index, type, runtime_type, ret)))
      return success();

    // If conversion failed try the next one if available.
    if constexpr (idx + 1 < sizeof...(RetValue))
      return ReturnValue<idx + 1>(result_index, type, runtime_type, ret);

    return failure();
  }

  RetError ret_error_;
  std::tuple<RetValue...> ret_value_;
};

template <typename RetError, typename... RetValue>
ResultConverterSet(RetError, RetValue...)
    -> ResultConverterSet<RetError, RetValue...>;

//===----------------------------------------------------------------------===//
// Helper functions for converting results of canonical types.
//===----------------------------------------------------------------------===//

namespace internal {

// This struct corresponds to the `llvm.struct` used by memref to llvm lowering
// pass to represent memref descriptors in the compilation pipeline. It is a
// type-erased version of `::mlir::StridedMemRefType<T>` template.
struct MemrefDescriptor {
  void* base_ptr;
  void* data_ptr;
  int64_t offset;
  int64_t sizes_and_strides[];
};

}  // namespace internal

// Converts returned memref using user-provided converter. Converter must
// satisfy this concept:
//
//   struct Converter {
//     ResultType operator()(PrimitiveType element_type, void* base_ptr,
//                           void* data_ptr, int64_t offset,
//                           absl::Span<const int64_t> dims,
//                           absl::Span<const int64_t> strides);
//   };
//
template <typename T, typename Converter>
FailureOr<T> ConvertReturnedMemref(const Converter& converter,
                                   const Type* memref_type, void* ret) {
  // Check if the runtime type is a valid memref.
  auto* memref = llvm::dyn_cast<MemrefType>(memref_type);
  if (!memref) return failure();

  PrimitiveType element_type = memref->element_type();
  size_t rank = memref->rank();

  auto* desc = reinterpret_cast<internal::MemrefDescriptor*>(ret);
  absl::Span<const int64_t> dims(desc->sizes_and_strides, rank);
  absl::Span<const int64_t> strides(desc->sizes_and_strides + rank, rank);

  return converter(element_type, desc->base_ptr, desc->data_ptr, desc->offset,
                   dims, strides);
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_RESULTS_H_
