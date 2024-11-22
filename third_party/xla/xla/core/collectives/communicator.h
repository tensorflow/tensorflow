/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_CORE_COLLECTIVES_COMMUNICATOR_H_
#define XLA_CORE_COLLECTIVES_COMMUNICATOR_H_

#include <ostream>
#include <string>

namespace xla {

// Collective communicator defines the set of communicating XLA processes.
class Communicator {
 public:
  virtual ~Communicator() = default;

  virtual std::string ToString() const = 0;
};

inline std::ostream& operator<<(std::ostream& os, const Communicator& comm) {
  return os << comm.ToString();
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COMMUNICATOR_H_
