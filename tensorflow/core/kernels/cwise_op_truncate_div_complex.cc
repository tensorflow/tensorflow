
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename T>
struct truncate_div<std::complex<T>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator()(
      const std::complex<T>& a, const std::complex<T>& b) const {
    auto result = a / b;
    return std::complex<T>(std::trunc(result.real()), std::trunc(result.imag()));
  }
};

}  // namespace functor
}  // namespace tensorflow

REGISTER2(BinaryOp, CPU, "TruncateDiv", tensorflow::functor::truncate_div,
          std::complex<float>, std::complex<double>);
