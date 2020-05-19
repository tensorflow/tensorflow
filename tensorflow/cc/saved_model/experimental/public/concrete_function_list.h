/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_LIST_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_LIST_H_

#include <vector>

#include "tensorflow/c/experimental/saved_model/public/concrete_function_list.h"
#include "tensorflow/cc/saved_model/experimental/public/concrete_function.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// ConcreteFunctionList helps convert an opaque pointer to an array of
// ConcreteFunction pointers to a std::vector.
class ConcreteFunctionList {
 public:
  // Converts this object to a std::vector<ConcreteFunction*>
  std::vector<ConcreteFunction*> ToVector();

 private:
  friend class SavedModelAPI;
  // Wraps a TF_ConcreteFunctionList. Takes ownership of list.
  explicit ConcreteFunctionList(TF_ConcreteFunctionList* list) : list_(list) {}

  struct TFConcreteFunctionListDeleter {
    void operator()(TF_ConcreteFunctionList* p) const {
      TF_DeleteConcreteFunctionList(p);
    }
  };
  std::unique_ptr<TF_ConcreteFunctionList, TFConcreteFunctionListDeleter> list_;
};

inline std::vector<ConcreteFunction*> ConcreteFunctionList::ToVector() {
  int size = TF_ConcreteFunctionListSize(list_.get());
  std::vector<ConcreteFunction*> result;
  result.reserve(size);
  for (int i = 0; i < size; ++i) {
    result.push_back(
        ConcreteFunction::wrap(TF_ConcreteFunctionListGet(list_.get(), i)));
  }
  return result;
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_LIST_H_
