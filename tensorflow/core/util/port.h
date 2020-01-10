#ifndef TENSORFLOW_UTIL_PORT_H_
#define TENSORFLOW_UTIL_PORT_H_

namespace tensorflow {

// Returns true if GOOGLE_CUDA is defined.
bool IsGoogleCudaEnabled();

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_PORT_H_
