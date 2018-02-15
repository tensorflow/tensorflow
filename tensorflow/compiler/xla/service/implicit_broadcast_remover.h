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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_IMPLICIT_BROADCAST_REMOVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_IMPLICIT_BROADCAST_REMOVER_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Pass which replaces all implicit broadcasts with their equivalent sequence of
// explicit broadcast and reshape instructions.
class ImplicitBroadcastRemover : public HloPassInterface {
 public:
  ImplicitBroadcastRemover() {}
  ~ImplicitBroadcastRemover() override {}

  tensorflow::StringPiece name() const override {
    return "implicit-broadcast-remover";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_IMPLICIT_BROADCAST_REMOVER_H_
