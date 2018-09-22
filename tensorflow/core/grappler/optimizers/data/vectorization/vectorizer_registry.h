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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_REGISTRY_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_REGISTRY_H_

#include <functional>
#include <map>

#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

// A global VectorizerRegistry is used to hold all the vectorizers.
class VectorizerRegistry {
 public:
  // Returns a pointer to a global VectorizerRegistry object.
  static VectorizerRegistry* Global();

  // Returns a pointer to a vectorizer that can vectorize an op for the op type.
  Vectorizer* Get(const string& op_type);

  // Registers a vectorizer that can vectorize an op for the given op type.
  void Register(const string& op_type, std::unique_ptr<Vectorizer> vectorizer);

 private:
  std::map<string, std::unique_ptr<Vectorizer>> vectorizers_;
};

namespace vectorizer_registration {

class VectorizerRegistration {
 public:
  VectorizerRegistration(const string& op_type,
                         std::unique_ptr<Vectorizer> vectorizer) {
    VectorizerRegistry::Global()->Register(op_type, std::move(vectorizer));
  }
};

}  // namespace vectorizer_registration

#define REGISTER_VECTORIZER(op_type, vectorizer) \
  REGISTER_VECTORIZER_UNIQ_HELPER(__COUNTER__, op_type, vectorizer)

#define REGISTER_VECTORIZER_UNIQ_HELPER(ctr, op_type, vectorizer) \
  REGISTER_VECTORIZER_UNIQ(ctr, op_type, vectorizer)

#define REGISTER_VECTORIZER_UNIQ(ctr, op_type, vectorizer)                  \
  static ::tensorflow::grappler::vectorization_utils::                      \
      vectorizer_registration::VectorizerRegistration                       \
          vectorizer_registration_##ctr(                                    \
              op_type,                                                      \
              ::std::unique_ptr<                                            \
                  ::tensorflow::grappler::vectorization_utils::Vectorizer>( \
                  new vectorizer()))

}  // namespace vectorization_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_VECTORIZER_REGISTRY_H_
