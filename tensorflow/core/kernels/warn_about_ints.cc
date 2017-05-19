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

#include "tensorflow/core/kernels/warn_about_ints.h"

namespace tensorflow {

void WarnAboutInts(OpKernelConstruction* context) {
  DataType dtype;
  OP_REQUIRES_OK(context, context->GetAttr("T", &dtype));
  if (DataTypeIsInteger(dtype)) {
    LOG(WARNING) << "Op " << context->def().name() << " of type "
                 << context->def().op() << " used with integer dtype "
                 << DataTypeString(dtype)
                 << ".  This op was registered with integer support "
                 << "accidentally, and you won't like the result.";
  }
}

}  // namespace tensorflow
