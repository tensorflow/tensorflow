#include "tensorflow/core/framework/bfloat16.h"

namespace tensorflow {

void FloatToBFloat16(const float* src, bfloat16* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  for (; size; p += 2, q++, size--) {
    *q = p[1];
  }
}

void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  for (; size; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
}

}  // end namespace tensorflow
