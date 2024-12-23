/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_COMPARISON_UTIL_H_
#define XLA_COMPARISON_UTIL_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

// A utility class for primitive comparisons. A comparison includes three
// components: the type of the elements being compared (F32, S16, etc), whether
// it is a partial or total order comparison, and the actual comparison operator
// (==, <=, >, etc).
//
// Note that integer comparisons are always total order. Float comparisons can
// be either total or partial order.
//
// Some examples:
//
//   Comparison a(
//     Comparison::Direction::kLt,
//     xla::PrimitiveType::BF16,
//     Comparison::Order::kTotal
//   );
//   a.ToString(); /* ".LT.BF16.TOTALORDER" */
//
//   Comparison b(Comparison::Direction::kEq, xla::PrimitiveType::U32);
//   b.IsTotalOrder(); /* true */
class Comparison {
 public:
  // Represents the ordering of the comparison.
  enum class Order : uint8_t {
    // https://en.wikipedia.org/wiki/Total_order
    kTotal,
    // https://en.wikipedia.org/wiki/Partially_ordered_set
    kPartial,
  };

  friend absl::string_view ComparisonOrderToString(Comparison::Order order);

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Order& p) {
    absl::Format(&sink, "%s", ComparisonOrderToString(p));
  }

  // Represents different comparison operations.
  enum class Direction : uint8_t {
    kEq,
    kNe,
    kGe,
    kGt,
    kLe,
    kLt,
  };

  // (DEPRECATED) Represents the type of comparison. Prefer xla::PrimitiveType
  // and Comparison::Order, since there are multiple floating point
  // representations that support total ordering.
  enum class [[deprecated("Use PrimitiveType and Order")]] Type : uint8_t{
      kFloat,
      kFloatTotalOrder,
      kSigned,
      kUnsigned,
  };

  Comparison() = delete;

  // This will default to the expected behavior for Comparison::Order: integers
  // will use total ordering, and floats will use partial ordering.
  explicit Comparison(Direction dir, PrimitiveType type);

  // Pass in a Comparison::Order to specify a non-default ordering, e.g., some
  // targets may support total order floating point type comparisons.
  explicit Comparison(Direction dir, PrimitiveType type, Order order);

  // Returns a comparison with a primitive type matching the Comparison::Type
  // and using a default bit width of 32. For example,
  // Comparison(Direction::kLt, Type::kFloat).PrimitiveType()  /* F32 */
  [[deprecated(
      "Use Comparison(Comparison::Direction, "
      "PrimitiveType)")]] explicit Comparison(Direction dir, Type type);

  inline Direction GetDirection() const { return dir_; }
  inline PrimitiveType GetPrimitiveType() const { return primitive_type_; }
  inline Order GetOrder() const { return order_; }

  [[deprecated("Use GetPrimitiveType() and GetOrder()")]] inline Type GetType()
      const {
    return type_;
  }

  inline bool IsEq() const { return dir_ == Direction::kEq; }
  inline bool IsNe() const { return dir_ == Direction::kNe; }
  inline bool IsGe() const { return dir_ == Direction::kGe; }
  inline bool IsGt() const { return dir_ == Direction::kGt; }
  inline bool IsLt() const { return dir_ == Direction::kLt; }
  inline bool IsTotalOrder() const { return order_ == Order::kTotal; }
  inline bool IsPartialOrder() const { return order_ == Order::kPartial; }

  // Returns whether this is a floating point total order comparison.
  inline bool IsF32TotalOrder() const {
    return primitive_type_ == PrimitiveType::F32 && IsTotalOrder();
  }

  // Returns whether this is a standard comparison, i.e., what you would expect
  // as the industry standard on most architectures.
  inline bool IsStandardF32() const {
    return primitive_type_ == PrimitiveType::F32 && IsPartialOrder();
  }
  inline bool IsStandardS32() const {
    return primitive_type_ == PrimitiveType::S32 && IsTotalOrder();
  }
  inline bool IsStandardU32() const {
    return primitive_type_ == PrimitiveType::U32 && IsTotalOrder();
  }

  inline bool IsIntegralPrimitiveType() const {
    return primitive_util::IsIntegralType(primitive_type_);
  }
  inline bool IsFloatingPointPrimitiveType() const {
    return primitive_util::IsFloatingPointType(primitive_type_);
  }

  // Returns whether (a dir a) is always true for this comparison.
  bool IsReflexive() const;

  // Returns whether (a dir a) is always false for this comparison.
  bool IsAntireflexive() const;

  // Gets the converse of the given comparison direction (e.g. >= turns to <=).
  // Useful when commuting operands to get constants into immediate-accepting
  // positions in the ISA.
  Comparison Converse() const;

  // Gets the inverse of the given comparison if it exists (e.g. >= turns to <).
  // Returns optional value because not all inversions may be supported.
  std::optional<Comparison> Inverse() const;

  // Returns a string version of this comparison, e.g., ".GT.F32.TOTALORDER"
  std::string ToString(std::string prefix1 = ".", std::string prefix2 = ".",
                       std::string prefix3 = ".") const;

  // Returns a comparison operator: (T, T) -> bool for this Comparison's
  // Direction.
  template <typename T>
  inline std::function<bool(T, T)> GetComparator() const {
    switch (GetDirection()) {
      case Direction::kEq:
        return std::equal_to<T>();
      case Direction::kNe:
        return std::not_equal_to<T>();
      case Direction::kGe:
        return std::greater_equal<T>();
      case Direction::kGt:
        return std::greater<T>();
      case Direction::kLe:
        return std::less_equal<T>();
      case Direction::kLt:
        return std::less<T>();
    }
  }

  template <typename T>
  inline bool Compare(const T a, const T b) const {
    DCHECK(primitive_util::IsCanonicalRepresentation<T>(primitive_type_));
    if constexpr (is_specialized_floating_point_v<T>) {
      if (IsTotalOrder()) {
        //  -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
        // Reference:
        // https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations
        using R = SignedIntegerTypeForSizeType<sizeof(T)>;
        return GetComparator<R>()(ToSignMagnitude(a), ToSignMagnitude(b));
      }
    }
    // Applies the comparison from this Comparison's direction and ordering.
    return GetComparator<T>()(a, b);
  }

  // Returns the Comparison::Type for the given primitive type. This assumes
  // that each numerical representation follows the standard behavior, e.g.,
  // integers are total order and floats are partial order.
  [[deprecated("Use PrimitiveType and Order")]] static Comparison::Type
  DefaultComparisonType(PrimitiveType type);

 private:
  // The direction of the Comparison, e.g., GT.
  const Direction dir_;
  // The primitive type of the Comparison operands, e.g., F32.
  const PrimitiveType primitive_type_;
  // The ordering of the Comparison, e.g., kPartial.
  const Order order_;
  // The Type of the Comparison. This tries to mesh together the ordering and
  // the numerical data classification.
  [[deprecated]] const Type type_;
};

using ComparisonDirection = Comparison::Direction;
using ComparisonOrder = Comparison::Order;

inline std::ostream& operator<<(std::ostream& os, const Comparison& cmp) {
  return os << cmp.ToString();
}

std::string ComparisonDirectionToString(Comparison::Direction direction);
std::string ComparisonTypeToString(Comparison::Type type);
absl::string_view ComparisonPrimitiveTypeToString(PrimitiveType type);

absl::StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction);
absl::StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view comparison);

// Returns a comparison function using the provided key function on each value,
// i.e. `key_fn(a) < key_fn(b)`.
template <typename KeyFn>
auto LessThanByKey(KeyFn&& key_fn) {
  return [=](const auto& a, const auto& b) { return key_fn(a) < key_fn(b); };
}

// Two comparisons are equivalent iff they have the same direction, precision,
// and ordering.
inline bool operator==(const Comparison& a, const Comparison& b) {
  return a.GetDirection() == b.GetDirection() &&
         a.GetPrimitiveType() == b.GetPrimitiveType() &&
         a.GetOrder() == b.GetOrder();
}

inline bool operator!=(const Comparison& a, const Comparison& b) {
  return !(a == b);
}

}  // namespace xla

#endif  // XLA_COMPARISON_UTIL_H_
