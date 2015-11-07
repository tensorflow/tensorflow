#ifndef TENSORFLOW_LIB_RANDOM_RANDOM_H_
#define TENSORFLOW_LIB_RANDOM_RANDOM_H_

#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace random {

// Return a 64-bit random value.  Different sequences are generated
// in different processes.
uint64 New64();

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_RANDOM_H_
