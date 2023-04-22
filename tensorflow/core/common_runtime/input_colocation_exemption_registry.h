/* Copyright 2019 The TensorFlow Authors. Al Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_INPUT_COLOCATION_EXEMPTION_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_INPUT_COLOCATION_EXEMPTION_REGISTRY_H_

#include <string>

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TensorFlow runtime (both eager and graph) will aim to colocate ops with
// their resource inputs so that the ops can access the resource state. In some
// cases, such as tf.data ops, this is not desirable as the ops themselves might
// not have a kernel registered for the device on which the resource is placed
// and instead use a mechanism, such as a multi-device function, to access the
// resource state.
//
// This registry can be used to register and list ops that should be exempt from
// the input colocation described above.
//
// Example usage:
//   REGISTER_INPUT_COLOCATION_EXEMPTION("MapDataset");
class InputColocationExemptionRegistry {
 public:
  // Returns a pointer to a global InputColocationExemptionRegistry object.
  static InputColocationExemptionRegistry* Global();

  // Returns the set of ops exempt from the input colocation constraints.
  const gtl::FlatSet<string>& Get() { return ops_; }

  // Registers an op to be excluded from the input colocation constraints.
  void Register(const string& op);

 private:
  gtl::FlatSet<string> ops_;
};

namespace input_colocation_exemption_registration {

class InputColocationExemptionRegistration {
 public:
  explicit InputColocationExemptionRegistration(const string& op) {
    InputColocationExemptionRegistry::Global()->Register(op);
  }
};

}  // namespace input_colocation_exemption_registration

#define REGISTER_INPUT_COLOCATION_EXEMPTION(op) \
  REGISTER_INPUT_COLOCATION_EXEMPTION_UNIQ_HELPER(__COUNTER__, op)

#define REGISTER_INPUT_COLOCATION_EXEMPTION_UNIQ_HELPER(ctr, op) \
  REGISTER_INPUT_COLOCATION_EXEMPTION_UNIQ(ctr, op)

#define REGISTER_INPUT_COLOCATION_EXEMPTION_UNIQ(ctr, op) \
  static input_colocation_exemption_registration::        \
      InputColocationExemptionRegistration                \
          input_colocation_exemption_registration_fn_##ctr(op)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_INPUT_COLOCATION_EXEMPTION_REGISTRY_H_
