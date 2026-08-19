/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_RTTI_H_
#define XLA_PYTHON_IFRT_RTTI_H_

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "xla/python/ifrt/ref_wrapper.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// TODO(hyeontaek): Rename member variable and methods to conform to Google C++
// Style Guide.

namespace internal {

// Maximum supported inheritance depth in IFRT RTTI hierarchies.
inline constexpr size_t kMaxRTTIDepth = 8;

// Static read-only type descriptor for O(1) ancestry display table lookups.
struct RTTITypeInfo {
  const void* const id;
  const std::array<const void*, kMaxRTTIDepth> ancestors;
};

}  // namespace internal

// Abstract root class for IFRT RTTI hierarchies.
class RTTIRoot {
 public:
  virtual ~RTTIRoot() = default;

  RTTIRoot& operator=(const RTTIRoot&) { return *this; }
  RTTIRoot& operator=(RTTIRoot&&) noexcept { return *this; }

  // Returns the static class ID of RTTIRoot.
  static const void* classID() { return &ID; }

  // Returns the dynamic runtime class ID of this instance.
  const void* dynamicClassID() const { return rtti_type_info_->id; }

  static char ID;  // NOLINT

 protected:
  static constexpr size_t kDepth = 0;
  static constexpr internal::RTTITypeInfo kTypeInfo = {&ID, {&ID}};

  const internal::RTTITypeInfo* rtti_type_info_;

  RTTIRoot() : rtti_type_info_(&kTypeInfo) {}
  RTTIRoot(const RTTIRoot&) : rtti_type_info_(&kTypeInfo) {}
  RTTIRoot(RTTIRoot&&) noexcept : rtti_type_info_(&kTypeInfo) {}

  template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
  friend bool isa(const T* absl_nonnull ptr);
};

// CRTP base class for IFRT RTTI in single-inheritance class hierarchies.
//
// `RTTIExtends<ThisT, ParentT>` constructs a static O(1) display table
// in `.rodata` padded with `nullptr` sentinels, enabling single-instruction
// ancestry checks.
template <typename ThisT, typename ParentT>
class RTTIExtends : public ParentT {
 public:
  static_assert(std::is_base_of_v<RTTIRoot, ParentT>,
                "ParentT must be part of the IFRT RTTI hierarchy.");
  static_assert(ParentT::kDepth + 1 < internal::kMaxRTTIDepth,
                "Exceeded maximum supported IFRT RTTI inheritance depth.");

  // TODO(hyeontaek): Add a `static_assert` that `&ThisT::ID is different from
  // `&ParentT::ID`. While not defining `ID` is handy when dynamic casting is
  // not expected (e.g., classes defined for testing), it is error-prone in
  // general. Thus, it would be safer to require `ID` to be defined. Note that
  // we cannot enforce this requirement yet because the user code must be first
  // migrated to define `ID`.

  RTTIExtends& operator=(const RTTIExtends&) = default;
  RTTIExtends& operator=(RTTIExtends&&) = default;

  // Returns the static class ID for `ThisT`.
  static const void* classID() { return &ThisT::ID; }

 private:
  template <size_t... Is>
  static constexpr internal::RTTITypeInfo MakeTypeInfo(
      std::index_sequence<Is...>) {
    return {&ThisT::ID,
            {(Is < kDepth ? ParentT::kTypeInfo.ancestors[Is]
                          : (Is == kDepth ? &ThisT::ID : nullptr))...}};
  }

 protected:
  static constexpr size_t kDepth = ParentT::kDepth + 1;
  static constexpr internal::RTTITypeInfo kTypeInfo =
      MakeTypeInfo(std::make_index_sequence<internal::kMaxRTTIDepth>{});

  RTTIExtends() { this->rtti_type_info_ = &kTypeInfo; }
  RTTIExtends(const RTTIExtends& other) : ParentT(other) {
    this->rtti_type_info_ = &kTypeInfo;
  }
  RTTIExtends(RTTIExtends&& other) : ParentT(std::move(other)) {
    this->rtti_type_info_ = &kTypeInfo;
  }

  template <
      typename FirstArg, typename... RestArgs,
      std::enable_if_t<!std::is_same_v<std::decay_t<FirstArg>, RTTIExtends> ||
                           (sizeof...(RestArgs) > 0),
                       int> = 0>
  explicit RTTIExtends(FirstArg&& first_arg, RestArgs&&... rest_args)
      : ParentT(std::forward<FirstArg>(first_arg),
                std::forward<RestArgs>(rest_args)...) {
    this->rtti_type_info_ = &kTypeInfo;
  }

  template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
  friend bool isa(const T* absl_nonnull ptr);
};

// -----------------------------------------------------------------------------
// isa<TargetT>
// -----------------------------------------------------------------------------

// Returns true if pointer `ptr` is of runtime type `TargetT` or a subclass.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa(const T* absl_nonnull ptr) {
  static_assert(std::is_base_of_v<RTTIRoot, TargetT>,
                "TargetT must be part of the IFRT RTTI hierarchy.");
  static_assert(std::is_base_of_v<RTTIRoot, T>,
                "Source type T must be part of the IFRT RTTI hierarchy.");
  static_assert(
      std::is_base_of_v<TargetT, T> || std::is_base_of_v<T, TargetT>,
      "Casting between disjoint/unrelated types in the hierarchy is illegal.");
  CHECK(ptr != nullptr) << "isa<> used on a null pointer";
  if constexpr (std::is_base_of_v<TargetT, T>) {
    return true;
  }
  return ptr->rtti_type_info_->ancestors[TargetT::kDepth] == TargetT::classID();
}

// Returns true if reference `ref` is of runtime type `TargetT` or a subclass.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::enable_if_t<std::is_base_of_v<RTTIRoot, T>, bool> isa(const T& ref) {
  return isa<TargetT>(&ref);
}

// Smart pointer overloads for `isa<TargetT>`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull std::unique_ptr<T>& val) {
  return isa<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull std::shared_ptr<T>& val) {
  return isa<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull tsl::RCReference<T>& val) {
  return isa<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa(const RCReferenceWrapper<T>& val) {
  return isa<TargetT>(val.get());
}

// Variadic overloads for checking if a value is of any of the target types.
template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
bool isa(const T* absl_nonnull ptr) {
  return isa<First>(ptr) || (isa<Second>(ptr) || ... || isa<Rest>(ptr));
}

template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
std::enable_if_t<std::is_base_of_v<RTTIRoot, T>, bool> isa(const T& ref) {
  return isa<First, Second, Rest...>(&ref);
}

template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull std::unique_ptr<T>& val) {
  return isa<First, Second, Rest...>(val.get());
}

template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull std::shared_ptr<T>& val) {
  return isa<First, Second, Rest...>(val.get());
}

template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
bool isa(const absl_nonnull tsl::RCReference<T>& val) {
  return isa<First, Second, Rest...>(val.get());
}

template <typename First, typename Second, typename... Rest,
          int&... ExplicitParameterBarrier, typename T>
bool isa(const RCReferenceWrapper<T>& val) {
  return isa<First, Second, Rest...>(val.get());
}

// -----------------------------------------------------------------------------
// isa_and_present<TargetT> / isa_and_nonnull<TargetT>
// -----------------------------------------------------------------------------

// Returns false if pointer `ptr` is nullptr; otherwise returns
// `isa<TargetT>(ptr)`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_present(const T* absl_nullable ptr) {
  return ptr != nullptr && isa<TargetT>(ptr);
}

// Smart pointer overloads for `isa_and_present<TargetT>`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_present(const absl_nullable std::unique_ptr<T>& val) {
  return isa_and_present<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_present(const absl_nullable std::shared_ptr<T>& val) {
  return isa_and_present<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_present(const absl_nullable tsl::RCReference<T>& val) {
  return isa_and_present<TargetT>(val.get());
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_present(const RCReferenceWrapper<T>& val) {
  return isa_and_present<TargetT>(val.get());
}

// Alias for `isa_and_present` matching LLVM naming conventions.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_nonnull(const T* absl_nullable ptr) {
  return isa_and_present<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_nonnull(const absl_nullable std::unique_ptr<T>& val) {
  return isa_and_present<TargetT>(val);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_nonnull(const absl_nullable std::shared_ptr<T>& val) {
  return isa_and_present<TargetT>(val);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_nonnull(const absl_nullable tsl::RCReference<T>& val) {
  return isa_and_present<TargetT>(val);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
bool isa_and_nonnull(const RCReferenceWrapper<T>& val) {
  return isa_and_present<TargetT>(val);
}

// -----------------------------------------------------------------------------
// cast<TargetT>
// -----------------------------------------------------------------------------

// Unconditionally casts non-null pointer `ptr` to `TargetT*`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* cast(T* absl_nonnull ptr) {
  CHECK(ptr != nullptr) << "cast<> used on a null pointer";
  CHECK(isa<TargetT>(ptr)) << "cast<> failed: type mismatch";
  return static_cast<TargetT*>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* cast(const T* absl_nonnull ptr) {
  CHECK(ptr != nullptr) << "cast<> used on a null pointer";
  CHECK(isa<TargetT>(ptr)) << "cast<> failed: type mismatch";
  return static_cast<const TargetT*>(ptr);
}

// Unconditionally casts reference `ref` to `TargetT&`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::enable_if_t<std::is_base_of_v<RTTIRoot, T>, TargetT&> cast(T& ref) {
  CHECK(isa<TargetT>(ref)) << "cast<> failed: type mismatch";
  return static_cast<TargetT&>(ref);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::enable_if_t<std::is_base_of_v<RTTIRoot, T>, const TargetT&> cast(
    const T& ref) {
  CHECK(isa<TargetT>(ref)) << "cast<> failed: type mismatch";
  return static_cast<const TargetT&>(ref);
}

// Smart pointer overloads for `cast<TargetT>`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> cast(absl_nonnull std::unique_ptr<T> val) {
  CHECK(val != nullptr) << "cast<> used on a null unique_ptr";
  CHECK(isa<TargetT>(val.get())) << "cast<> failed: type mismatch";
  return std::unique_ptr<TargetT>(static_cast<TargetT*>(val.release()));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> cast(absl_nonnull std::shared_ptr<T> val) {
  CHECK(val != nullptr) << "cast<> used on a null shared_ptr";
  CHECK(isa<TargetT>(val.get())) << "cast<> failed: type mismatch";
  return std::static_pointer_cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> cast(absl_nonnull tsl::RCReference<T> val) {
  CHECK(val != nullptr) << "cast<> used on a null RCReference";
  CHECK(isa<TargetT>(val.get())) << "cast<> failed: type mismatch";
  return tsl::TakeRef(static_cast<TargetT*>(val.release()));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> cast(RCReferenceWrapper<T> val) {
  CHECK(val != nullptr) << "cast<> used on a null RCReferenceWrapper";
  CHECK(isa<TargetT>(val.get())) << "cast<> failed: type mismatch";
  return RCReferenceWrapper<TargetT>(
      tsl::TakeRef(static_cast<TargetT*>(val.release())));
}

// -----------------------------------------------------------------------------
// cast_if_present<TargetT> / cast_or_null<TargetT>
// -----------------------------------------------------------------------------

// Returns nullptr if `ptr` is nullptr; otherwise returns `cast<TargetT>(ptr)`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* cast_if_present(T* absl_nullable ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }
  return cast<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* cast_if_present(const T* absl_nullable ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }
  return cast<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> cast_if_present(absl_nullable std::unique_ptr<T> val) {
  if (val == nullptr) {
    return {};
  }
  return cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> cast_if_present(absl_nullable std::shared_ptr<T> val) {
  if (val == nullptr) {
    return {};
  }
  return cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> cast_if_present(
    absl_nullable tsl::RCReference<T> val) {
  if (val == nullptr) {
    return {};
  }
  return cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> cast_if_present(RCReferenceWrapper<T> val) {
  if (val == nullptr) {
    return {};
  }
  return cast<TargetT>(std::move(val));
}

// Alias for `cast_if_present`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* cast_or_null(T* absl_nullable ptr) {
  return cast_if_present<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* cast_or_null(const T* absl_nullable ptr) {
  return cast_if_present<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> cast_or_null(absl_nullable std::unique_ptr<T> val) {
  return cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> cast_or_null(absl_nullable std::shared_ptr<T> val) {
  return cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> cast_or_null(absl_nullable tsl::RCReference<T> val) {
  return cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> cast_or_null(RCReferenceWrapper<T> val) {
  return cast_if_present<TargetT>(std::move(val));
}

// -----------------------------------------------------------------------------
// dyn_cast<TargetT>
// -----------------------------------------------------------------------------

// Dynamically casts non-null pointer `ptr` to `TargetT*` if `isa<TargetT>(ptr)`
// is true; otherwise returns nullptr.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* dyn_cast(T* absl_nonnull ptr) {
  CHECK(ptr != nullptr) << "dyn_cast<> used on a null pointer";
  if (isa<TargetT>(ptr)) {
    return static_cast<TargetT*>(ptr);
  }
  return nullptr;
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* dyn_cast(const T* absl_nonnull ptr) {
  CHECK(ptr != nullptr) << "dyn_cast<> used on a null pointer";
  if (isa<TargetT>(ptr)) {
    return static_cast<const TargetT*>(ptr);
  }
  return nullptr;
}

// Smart pointer overloads for `dyn_cast<TargetT>`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> dyn_cast(absl_nonnull std::unique_ptr<T> val) {
  CHECK(val != nullptr) << "dyn_cast<> used on a null unique_ptr";
  if (isa<TargetT>(val.get())) {
    return std::unique_ptr<TargetT>(static_cast<TargetT*>(val.release()));
  }
  return {};
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> dyn_cast(absl_nonnull std::shared_ptr<T> val) {
  CHECK(val != nullptr) << "dyn_cast<> used on a null shared_ptr";
  if (isa<TargetT>(val.get())) {
    return std::static_pointer_cast<TargetT>(std::move(val));
  }
  return {};
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> dyn_cast(absl_nonnull tsl::RCReference<T> val) {
  CHECK(val != nullptr) << "dyn_cast<> used on a null RCReference";
  if (isa<TargetT>(val.get())) {
    return tsl::TakeRef(static_cast<TargetT*>(val.release()));
  }
  return {};
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> dyn_cast(RCReferenceWrapper<T> val) {
  CHECK(val != nullptr) << "dyn_cast<> used on a null RCReferenceWrapper";
  if (isa<TargetT>(val.get())) {
    return RCReferenceWrapper<TargetT>(
        tsl::TakeRef(static_cast<TargetT*>(val.release())));
  }
  return {};
}

// -----------------------------------------------------------------------------
// dyn_cast_if_present<TargetT> / dyn_cast_or_null<TargetT>
// -----------------------------------------------------------------------------

// Returns nullptr if `ptr` is nullptr; otherwise returns
// `dyn_cast<TargetT>(ptr)`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* dyn_cast_if_present(T* absl_nullable ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }
  return dyn_cast<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* dyn_cast_if_present(const T* absl_nullable ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }
  return dyn_cast<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> dyn_cast_if_present(
    absl_nullable std::unique_ptr<T> val) {
  if (val == nullptr) {
    return {};
  }
  return dyn_cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> dyn_cast_if_present(
    absl_nullable std::shared_ptr<T> val) {
  if (val == nullptr) {
    return {};
  }
  return dyn_cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> dyn_cast_if_present(
    absl_nullable tsl::RCReference<T> val) {
  if (val == nullptr) {
    return {};
  }
  return dyn_cast<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> dyn_cast_if_present(RCReferenceWrapper<T> val) {
  if (val == nullptr) {
    return {};
  }
  return dyn_cast<TargetT>(std::move(val));
}

// Alias for `dyn_cast_if_present`.
template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
TargetT* dyn_cast_or_null(T* absl_nullable ptr) {
  return dyn_cast_if_present<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
const TargetT* dyn_cast_or_null(const T* absl_nullable ptr) {
  return dyn_cast_if_present<TargetT>(ptr);
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::unique_ptr<TargetT> dyn_cast_or_null(
    absl_nullable std::unique_ptr<T> val) {
  return dyn_cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
std::shared_ptr<TargetT> dyn_cast_or_null(
    absl_nullable std::shared_ptr<T> val) {
  return dyn_cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
tsl::RCReference<TargetT> dyn_cast_or_null(
    absl_nullable tsl::RCReference<T> val) {
  return dyn_cast_if_present<TargetT>(std::move(val));
}

template <typename TargetT, int&... ExplicitParameterBarrier, typename T>
RCReferenceWrapper<TargetT> dyn_cast_or_null(RCReferenceWrapper<T> val) {
  return dyn_cast_if_present<TargetT>(std::move(val));
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_RTTI_H_
