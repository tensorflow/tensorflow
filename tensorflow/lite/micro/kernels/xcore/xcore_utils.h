#ifndef XCORE_UTILS_H_
#define XCORE_UTILS_H_

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

/* Unpack an integer data type from a byte array
 *  T  data type to unpack
 *
 * Example usage:
 *      int32_t t0 = unpack<int32_t>(&my_buffer[23]);
 *      int32_t t1 = unpack<int32_t>(&my_buffer[27]);
 */
template <class T>
T unpack(const uint8_t* buffer) {
  T retval = 0;
  for (int i = 0; i < sizeof(T); ++i) retval |= buffer[i] << (8 * i);
  return retval;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_UTILS_H_