#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_SELECT(CPU, "Select", "", float);
REGISTER_SELECT(CPU, "Select", "", double);
REGISTER_SELECT(CPU, "Select", "", int32);
REGISTER_SELECT(CPU, "Select", "", int64);
REGISTER_SELECT(CPU, "Select", "", complex64);
REGISTER_SELECT(CPU, "Select", "", string);
#if GOOGLE_CUDA
REGISTER_SELECT(GPU, "Select", "", float);
REGISTER_SELECT(GPU, "Select", "", double);
REGISTER_SELECT(GPU, "Select", "", int32);
REGISTER_SELECT(GPU, "Select", "", int64);
REGISTER_SELECT(GPU, "Select", "", complex64);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
