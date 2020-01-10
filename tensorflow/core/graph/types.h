#ifndef TENSORFLOW_GRAPH_TYPES_H_
#define TENSORFLOW_GRAPH_TYPES_H_

#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// We model running time in microseconds.
TF_LIB_GTL_DEFINE_INT_TYPE(Microseconds, int64);

// We model size in bytes.
TF_LIB_GTL_DEFINE_INT_TYPE(Bytes, int64);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_TYPES_H_
