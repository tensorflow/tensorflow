/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef DTENSOR_OPS
#else
#define DTENSOR_OPS

include "dtensor_dialect.td"

//===----------------------------------------------------------------------===//
// DTensor op definitions
//===----------------------------------------------------------------------===//

// Base class for the operations in this dialect.
class DTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<DTensorDialect, mnemonic, traits>;

#endif // DTENSOR_OPS
