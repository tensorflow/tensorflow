#include "tensorflow/core/util/port.h"

namespace tensorflow {

bool IsGoogleCudaEnabled() {
#if GOOGLE_CUDA
  return true;
#else
  return false;
#endif
}

}  // end namespace tensorflow
