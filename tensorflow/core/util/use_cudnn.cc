#include "tensorflow/core/util/use_cudnn.h"

#include <stdlib.h>

#include "tensorflow/core/platform/port.h"

namespace tensorflow {

bool CanUseCudnn() {
  const char* tf_use_cudnn = getenv("TF_USE_CUDNN");
  if (tf_use_cudnn != nullptr) {
    string tf_use_cudnn_str = tf_use_cudnn;
    if (tf_use_cudnn_str == "0") {
      return false;
    }
  }
  return true;
}

}  // namespace tensorflow
