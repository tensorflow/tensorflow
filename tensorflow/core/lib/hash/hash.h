// Simple hash functions used for internal data structures

#ifndef TENSORFLOW_LIB_HASH_HASH_H_
#define TENSORFLOW_LIB_HASH_HASH_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "tensorflow/core/platform/port.h"

namespace tensorflow {

extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const string& str) {
  return Hash64(str.data(), str.size());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_HASH_HASH_H_
