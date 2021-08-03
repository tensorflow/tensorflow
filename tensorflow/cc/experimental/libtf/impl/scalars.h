/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_SCALARS_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_SCALARS_H_

#include <stdint.h>

#include <iosfwd>
#include <utility>

namespace tf {
namespace libtf {
namespace impl {

/** A thin wrapper around a C++ scalar value.
 * This wrapper makes the scalar immutable.
 */
template <typename T>
class Scalar final {
 public:
  explicit Scalar(T x) : value_(x) {}
  Scalar(const Scalar<T>& o) : value_(o.value_) {}

  bool operator==(const Scalar<T>& o) const { return o.value_ == value_; }

  T get() const { return value_; }

  /** Absl hash function. */
  template <typename H>
  friend H AbslHashValue(H h, const Scalar<T>& x) {
    return H::combine(std::move(h), x.value_);
  }

 private:
  const T value_;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& o, const Scalar<T>& x) {
  return o << x.get();
}

/** The overloaded addition operator. */
template <typename T1, typename T2>
inline auto operator+(const Scalar<T1>& x1, const Scalar<T2>& x2)
    -> Scalar<decltype(x1.get() + x2.get())> {
  using Ret = decltype(x1 + x2);  // Return type of this function.
  return Ret(x1.get() + x2.get());
}

using Int64 = Scalar<int64_t>;
using Float32 = Scalar<float>;

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_SCALARS_H_
