#include "tensorflow/core/framework/cus.h"

namespace tensorflow {

void FloatToCus(const float* src, cus* dst, int64 size) {
  //   for (; size != 0; src++, dst++, size--) {
  // #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  //     memcpy(dst, src, sizeof(cus));
  // #else
  //     memcpy(
  //         dst,
  //         reinterpret_cast<const char*>(src) + sizeof(float) -
  //         sizeof(cus), sizeof(cus));
  // #endif
  //   }
  dst->value = *src;
}

void CusToFloat(const cus* src, float* dst, int64 size) {
  //   const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  //   uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  // #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  //   for (; size != 0; p += 2, q++, size--) {
  //     *q = p[0];
  //   }
  // #else
  //   for (; size != 0; p += 2, q++, size--) {
  //     *q = p[1];
  //   }
  // #endif
  *dst = src->value;
}

}  // end namespace tensorflow
