#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

class MathUtil {
 public:
  template <typename IntegralType>
  static IntegralType CeilOfRatio(IntegralType numerator,
                                  IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
  }
  template <typename IntegralType>
  static IntegralType FloorOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
  }
  template <typename IntegralType, bool ceil>
  static IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                         IntegralType denominator);
};

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64 into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template<typename IntegralType, bool ceil>
IntegralType MathUtil::CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  static_assert(std::is_integral<IntegralType>::value,
                 "CeilOfRatio_is_only_defined_for_integral_types");
  assert(denominator != 0);
  // Dividing the smallest signed integer by -1 is not supported: it would
  // SIGFPE
  assert(!std::is_signed<IntegralType>::value ||
         numerator != std::numeric_limits<IntegralType>::min() ||
         denominator != -1);

  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if (ceil) {  // Compile-time condition: not an actual branching
    // When rounded_toward_zero is negative, then an adjustment is never needed:
    // the real ratio is negative, and so rounded toward zero is the ceil.
    // When rounded_toward_zero is non-negative, an adjustment is needed if the
    // sign of the difference numerator - intermediate_product is the same as
    // the sign of the denominator.
    //
    // Using a bool and then a static_cast to IntegralType is not strictly
    // necessary, but it makes the code clear, and anyway the compiler should
    // get rid of it.
    const bool needs_adjustment = (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
            (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    // Floor case: symmetrical to the previous one
    const bool needs_adjustment = (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_
