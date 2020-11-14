/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {

// The magic number in the template parameter is the maximum number of ops that
// can be added to AllOpsResolver. It can be increased if needed. And most
// applications that care about the memory footprint will want to directly use
// MicroMutableOpResolver and have an application specific template parameter.
// The examples directory has sample code for this.
class AllOpsResolver : public MicroMutableOpResolver<128> {
 public:
  AllOpsResolver();

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_
