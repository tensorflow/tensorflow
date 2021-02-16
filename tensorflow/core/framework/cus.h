#ifndef TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_
#define TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

// This type only supports conversion back and forth with float.

namespace tensorflow {

// Conversion routines between an array of float and cus of
// "size".
void FloatToCus(const float* src, cus* dst, int64 size);
void CusToFloat(const cus* src, float* dst, int64 size);

}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_FRAMEWORK_CUS_TYPE_H_
