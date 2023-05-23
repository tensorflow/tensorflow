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

// Registers casts between custom types.  This is to help reduce dependencies
// between the types themselves.

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_CUSTOM_CASTS_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_CUSTOM_CASTS_H_

namespace tsl {

bool RegisterCustomCasts();

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_CUSTOM_CASTS_H_
