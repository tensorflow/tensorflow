/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/defs.h"

namespace tensorflow {

const char* const kXlaMustCompileAttr = "_XlaMustCompile";

const char* const kXlaCompileAttr = "_XlaCompile";

// User-provided through jit_scope APIs. Effective only when auto_jit is OFF.
const char* const kXlaScopeAttr = "_XlaScope";

// Automatically inserted by auto_jit to guide clustering results.  Effective
// only when auto_jit is ON.
const char* const kXlaInternalScopeAttr = "_XlaInternalScope";

const char* const kXlaClusterIdAttr = "_xla_compile_id";

}  // namespace tensorflow
