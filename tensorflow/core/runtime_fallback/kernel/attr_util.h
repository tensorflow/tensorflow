/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_

#include <map>
#include <string>
#include <typeinfo>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/util/padding.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime

namespace tensorflow {

// Map from attribute name to a string value representation.
typedef llvm::StringMap<std::string> AttrMap;

// Parse value from the given string input.
Status ParseValue(StringPiece input, bool* value);
Status ParseValue(StringPiece input, int32* value);
Status ParseValue(StringPiece input, DataType* value);
Status ParseValue(StringPiece input, std::string* value);
Status ParseValue(StringPiece input, std::vector<int32>* value);
Status ParseValue(StringPiece input, Padding* value);

Status AddOpAttr(const std::string& name, const std::string& attr_value,
                 tfrt::OpAttrs* opattrs);

Status FillOpAttrs(tfrt::RemainingAttributes attrs, tfrt::OpAttrs* opattrs);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_ATTR_UTIL_H_
