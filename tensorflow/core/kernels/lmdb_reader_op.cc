/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

#include <sys/stat.h>

namespace tensorflow {

#define MDB_CHECK(val) CHECK_EQ(val, MDB_SUCCESS) << mdb_strerror(val)

class LMDBReaderOp : public ReaderOpKernel {
 public:
  explicit LMDBReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    OP_REQUIRES(
        context, false,
        errors::Unimplemented(
            "LMDB support is removed from TensorFlow. This API will be deleted "
            "in the next TensorFlow release. If you need LMDB support, please "
            "file a GitHub issue."));
  }
};

REGISTER_KERNEL_BUILDER(Name("LMDBReader").Device(DEVICE_CPU), LMDBReaderOp);

}  // namespace tensorflow
