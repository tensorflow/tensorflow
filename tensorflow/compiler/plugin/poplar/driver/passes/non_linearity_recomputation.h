/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_NON_LINEARITY_RECOMPUTATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_NON_LINEARITY_RECOMPUTATION_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

// HLO pass which attempts to find non linearity (NL) operations in the graph
// (ReLU, sigmoid) which follow a training normalization operation (such as
// group norm). For every user of the NL, we clone the NL operation (NLClone)
// and replace all the uses of NL in the user with NLClone and we add control
// dependencies such that NLClone is executed as late as possible. This allows
// us not to store the output of NL for the backwards pass, saving us memory
// since we have to store the output of the normalization for the normalization
// gradient op anyway.
//
// For example:
// clang-format off
//  ---------------                              ---------------
// |     NORM      |                            |   NORM-GRAD   |
//  ---------------                              ---------------
//        ||                                       /\     /\
//        ||=======================================||     ||
//        \/                                              ||
//  ---------------                              ---------------
// |     RELU      |                            |   RELU-GRAD   |
//  ---------------                              ---------------
//        ||                                       /\     /\
//        ||=======================================||     ||
//        \/                        ||                    ||
//  ---------------                 ||           ---------------
// |   SOME-OP     |                ||          |  SOME-OP-GRAD |
//  ---------------                 ||           ---------------
//        ||                        ||             /\     /\
//        ||                        ||             ||     ||
//        \/                        ||=============||     ||
//       ....                                            ....
// clang-format on
// Here the ReLU has three users and follows the norm, so we can turn it into:
// clang-format off
//  ---------------                              ---------------
// |     NORM      |                            |   NORM-GRAD   |
//  ---------------                              ---------------
//        ||                                       /\     /\
//        ||=============||========||==============||     ||
//        \/             ||        \/                     ||
//  ---------------      ||  ---------------     ---------------
// |     RELU      |     || |     RELU      |   |   RELU-GRAD   |
//  ---------------      ||  ---------------     ---------------
//        ||             ||        ||              /\     /\
//        ||             ||        ||==============||     ||
//        \/             \/                               ||
//  ---------------     ---------------          ---------------
// |   SOME-OP     |   |     RELU      |        |  SOME-OP-GRAD |
//  ---------------     ---------------          ---------------
//        ||                  ||                   /\     /\
//        ||                  ||                   ||     ||
//        \/                  ||===================||     ||
//       ....                                            ....
// clang-format on
// This (memory)size-(computation) speed optimisation means that we don't need
// to hold onto the ReLU output, reducing the peak memory usage at the expense
// of recomputing the NL in the backwards pass.
class NonLinearityRecomputaion : public HloModulePass {
 public:
  NonLinearityRecomputaion(bool recompute_non_linearities);

  ~NonLinearityRecomputaion() override = default;

  absl::string_view name() const override {
    return "non-linearity-recomputation";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool recompute_non_linearities_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
