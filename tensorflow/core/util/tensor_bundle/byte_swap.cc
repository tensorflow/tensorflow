/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

Status ByteSwapArray(char* array, size_t bytes_per_elem, int array_len) {
  if (bytes_per_elem == 1) {
    // No-op
    return Status::OK();
  } else if (bytes_per_elem == 2) {
    auto array_16 = reinterpret_cast<uint16_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_16[i] = BYTE_SWAP_16(array_16[i]);
    }
    return Status::OK();
  } else if (bytes_per_elem == 4) {
    auto array_32 = reinterpret_cast<uint32_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_32[i] = BYTE_SWAP_32(array_32[i]);
    }
    return Status::OK();
  } else if (bytes_per_elem == 8) {
    auto array_64 = reinterpret_cast<uint64_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_64[i] = BYTE_SWAP_64(array_64[i]);
    }
    return Status::OK();
  } else {
    return errors::Unimplemented("Byte-swapping of ", bytes_per_elem,
                                 "-byte values not supported.");
  }
}

Status ByteSwapTensor(Tensor* t) {
  size_t bytes_per_elem = 0;
  int array_len = t->NumElements();

  switch (t->dtype()) {
    // Types that don't need byte-swapping
    case DT_STRING:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_BOOL:
    case DT_UINT8:
    case DT_INT8:
      return Status::OK();

    // 16-bit types
    case DT_BFLOAT16:
    case DT_HALF:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:
      bytes_per_elem = 2;
      break;

    // 32-bit types
    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:
      bytes_per_elem = 4;
      break;

    // 64-bit types
    case DT_INT64:
    case DT_DOUBLE:
    case DT_UINT64:
      bytes_per_elem = 8;
      break;

    // Complex types need special handling
    case DT_COMPLEX64:
      bytes_per_elem = 4;
      array_len *= 2;
      break;

    case DT_COMPLEX128:
      bytes_per_elem = 8;
      array_len *= 2;
      break;

    // Types that ought to be supported in the future
    case DT_RESOURCE:
    case DT_VARIANT:
      return errors::Unimplemented(
          "Byte-swapping not yet implemented for tensors with dtype ",
          t->dtype());

    // Byte-swapping shouldn't make sense for other dtypes.
    default:
      return errors::Unimplemented(
          "Byte-swapping not supported for tensors with dtype ", t->dtype());
  }

  char* backing_buffer = const_cast<char*>((t->tensor_data().data()));
  TF_RETURN_IF_ERROR(ByteSwapArray(backing_buffer, bytes_per_elem, array_len));
  return Status::OK();
}

}  // namespace tensorflow
