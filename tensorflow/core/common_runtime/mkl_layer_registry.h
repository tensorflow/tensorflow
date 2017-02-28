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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYER_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYER_REGISTRY_H_

#ifdef INTEL_MKL

#include <set>
#include <map>
#include <utility>
#include <string>

#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// A global MKLLayerRegistry is used to hold information about all MKL-compliant
// layers. Registry maintains information about <opname,T> pair supported
// by MKL API.
class MklLayerRegistry {
 public:
  // Add an optimization pass to the registry.
  // Registration is done based on opname supported by a layer.
  //
  // @input: name of the MKL-compliant op
  // @input: Data type of the MKL-compliant op
  // @return: none.
  void Register(const std::string& opname, DataType T);

  // Check whether opname is registered as MKL-compliant in the registry.
  //
  // @input: name of the op
  // @input: datatype of the op
  // @return: true if opname along with datatype is registered
  // as MKL-compliant; false otherwise.
  bool Find(const std::string& opname, DataType T) const;

  // Clear whole registry. Mostly used for unit test.
  // @input: None
  // @return: None
  void Clear(void);

  // Returns the global registry of MKL layers
  static MklLayerRegistry* Instance();

 private:
  /// map to hold op names and the supported types for compliant layers
  std::map<string, std::set<DataType>> mkl_layer_ops_;
  /// Only single registry for MKL registration
  static MklLayerRegistry* mkl_layer_registry_;

 private:
  // Private constructor - for singleton pattern
  MklLayerRegistry() {}

  TF_DISALLOW_COPY_AND_ASSIGN(MklLayerRegistry);
};

// Helper class to allow us to register MKL layers from ops
class MklLayerRegistrar {
 public:
  MklLayerRegistrar(const std::string& opname, DataType T) {
    MklLayerRegistry::Instance()->Register(opname, T);
  }
};

///////////////////////////////////////////////////////////
//           Public interface for registration
///////////////////////////////////////////////////////////

// We do not need any types other than half, float, and double for MKL for now.
#define REGISTER_MKL_LAYER_half(opname)                                 \
  REGISTER_MKL_LAYER_UNIQ_HELPER(__COUNTER__, std::string(opname), DT_DOUBLE)

#define REGISTER_MKL_LAYER_float(opname)                                 \
  REGISTER_MKL_LAYER_UNIQ_HELPER(__COUNTER__, std::string(opname), DT_FLOAT)

#define REGISTER_MKL_LAYER_double(opname)                                 \
  REGISTER_MKL_LAYER_UNIQ_HELPER(__COUNTER__, std::string(opname), DT_DOUBLE)

#define REGISTER_MKL_LAYER_UNIQ_HELPER(ctr, opname, T)          \
  REGISTER_MKL_LAYER_UNIQ(ctr, opname, T)

#define REGISTER_MKL_LAYER_UNIQ(ctr, opname, T)                       \
  static tensorflow::MklLayerRegistrar __internal_mkllayer##ctr##_obj(opname, T)

#define IS_MKL_LAYER(opname, T)                                     \
  tensorflow::MklLayerRegistry::Instance()->Find(opname, T)
}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYER_REGISTRY_H_

