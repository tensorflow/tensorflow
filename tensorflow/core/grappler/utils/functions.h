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

#ifndef TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_
#define TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {

namespace grappler {

// Factory method for creating a GrapplerItem from a FunctionDef.
// Returns nullptr if the given function def cannot be converted.
std::unique_ptr<GrapplerItem> GrapplerItemFromFunctionDef(
    const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr,
    const FunctionDefLibrary& library);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_
