#include <cmath>
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace test {

template <typename T>
bool IsClose(const T& x, const T& y, double atol, double rtol) {
  return fabs(x - y) < atol + rtol * fabs(x);
}

template <typename T>
void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (!IsClose(Tx(i), Ty(i), atol, rtol)) {
      LOG(ERROR) << "x = " << x.DebugString();
      LOG(ERROR) << "y = " << y.DebugString();
      LOG(ERROR) << "atol = " << atol << " rtol = " << rtol
                 << " tol = " << atol + rtol * std::fabs(Tx(i));
      EXPECT_TRUE(false) << i << "-th element is not close " << Tx(i) << " vs. "
                         << Ty(i);
    }
  }
}

void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  internal::AssertSameTypeDims(x, y);
  switch (x.dtype()) {
    case DT_FLOAT:
      ExpectClose<float>(x, y, atol, rtol);
      break;
    case DT_DOUBLE:
      ExpectClose<double>(x, y, atol, rtol);
      break;
    default:
      LOG(FATAL) << "Unexpected type : " << DataTypeString(x.dtype());
  }
}

}  // end namespace test
}  // end namespace tensorflow
