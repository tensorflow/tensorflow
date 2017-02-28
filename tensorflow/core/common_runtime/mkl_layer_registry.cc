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

#ifdef INTEL_MKL

#include <set>
#include <utility>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/common_runtime/mkl_layer_registry.h"

namespace tensorflow {

MklLayerRegistry *MklLayerRegistry::mkl_layer_registry_ = nullptr;

void MklLayerRegistry::Register(const string& opname, DataType T) {
  mkl_layer_ops_[opname].insert(T);
  VLOG(1) << "Added Op: " << opname << " to MKL layer registry";
}

bool MklLayerRegistry::Find(const string& opname, DataType T) const {
  // The op name has to match, and also the data type has to match.
  auto mit = mkl_layer_ops_.end();
  return (((mit = mkl_layer_ops_.find(opname)) != mkl_layer_ops_.end()) &&
           (mit->second.find(T) != mit->second.end()));
}

void MklLayerRegistry::Clear() {
  mkl_layer_ops_.clear();
}

MklLayerRegistry* MklLayerRegistry::Instance() {
  if (mkl_layer_registry_ == nullptr) {
    mkl_layer_registry_ = new MklLayerRegistry();
  }
  return mkl_layer_registry_;
}

}  // namespace tensorflow

#endif  // INTEL_MKL
