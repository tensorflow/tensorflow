/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_OPS_CC_OP_GEN_H_
#define TENSORFLOW_CC_OPS_CC_OP_GEN_H_

#include "tensorflow/core/framework/op_def.pb.h"

namespace tensorflow {

// Result is written to files dot_h and dot_cc.
void WriteCCOps(const OpList& ops, const std::string& dot_h_fname,
                const std::string& dot_cc_fname);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CC_OP_GEN_H_
