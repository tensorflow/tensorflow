#ifndef TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
