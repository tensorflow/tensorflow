#ifndef TENSORFLOW_PUBLIC_TENSORFLOW_SERVER_H_
#define TENSORFLOW_PUBLIC_TENSORFLOW_SERVER_H_

#include "tensorflow/core/public/status.h"

namespace tensorflow {

// Initialize the TensorFlow service for this address space.
// This is a blocking call that never returns.
// See BUILD file for details on linkage guidelines.
::tensorflow::Status InitTensorFlow();

// Like InitTensorFlow() but returns after the Tensorflow
// services have been launched.
::tensorflow::Status LaunchTensorFlow();

}  // namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_TENSORFLOW_SERVER_H_
