#ifndef TENSORFLOW_C_DLPACK_H_
#define TENSORFLOW_C_DLPACK_H_

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace tensorflow {

const char* const kDlTensorCapsuleName = "dltensor";

void* TFE_HandleToDLPack(TFE_TensorHandle* h, TF_Status* status);

TFE_TensorHandle* TFE_HandleFromDLPack(void* dlm, TF_Status* status);

void TFE_CallDLManagedTensorDeleter(void* dlm_ptr);
}  // namespace tensorflow

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_DLPACK_H_
