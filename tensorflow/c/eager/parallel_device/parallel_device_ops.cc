/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

// TODO(allenl): Figure out if we need this op, and if so whether we should move
// it to core TF. Right now the eager C API does some checking of op
// registrations before calling into custom devices, but we may be able to avoid
// that.
REGISTER_OP("DeviceID")
    .Output("device_id: int64")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);
