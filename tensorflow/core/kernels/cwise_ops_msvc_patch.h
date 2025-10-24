// Platform-specific fix for MSVC 64-bit integer comparison bug in minimum/maximum functors.
// This patch ensures that int64 comparisons are always performed as true 64-bit operations on Windows.

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_MSVC_PATCH_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_MSVC_PATCH_H_

#include <cstdint>
#include <type_traits>

namespace tensorflow {
namespace functor {

#if defined(_MSC_VER)
// Specialize maximum for int64_t on MSVC to avoid 32-bit truncation.
template <>
struct maximum<int64_t> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64_t operator()(const int64_t& x, const int64_t& y) const {
    // Use explicit cast to ensure 64-bit comparison
    return (static_cast<uint64_t>(x) > static_cast<uint64_t>(y)) ? x : y;
  }
};

template <>
struct minimum<int64_t> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64_t operator()(const int64_t& x, const int64_t& y) const {
    // Use explicit cast to ensure 64-bit comparison
    return (static_cast<uint64_t>(x) < static_cast<uint64_t>(y)) ? x : y;
  }
};
#endif

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_MSVC_PATCH_H_
