// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_DUMP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_DUMP_H_

#include <istream>
#include <ostream>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"

namespace lrt::internal {

//
// Lrt IR
//

// Dump details about the given LrtOpT to the given stream.
void Dump(const LrtOpT& op, std::ostream& out = std::cerr);

// Dump details about the given LrtSubgraphT to the given stream.
void Dump(const LrtSubgraphT& subgraph, std::ostream& out = std::cerr);

// Dump details about the given LrtTensorT to the given stream.
void Dump(const LrtTensorT& tensor, std::ostream& out = std::cerr);

// Dump details about the given LrtOpCode to the given stream.
void Dump(LrtOpCode code, std::ostream& out = std::cerr);

// Dump details about the given LrtElementType to the given stream.
void Dump(LrtElementType type, std::ostream& out = std::cerr);

// Dump details about the given LrtRankedTensorType to the given stream.
void Dump(const LrtRankedTensorType& type, std::ostream& out = std::cerr);

// Dump details about the given LrtModel to the given stream.
void Dump(const LrtModelT& model, std::ostream& out = std::cerr);

// Dump details about options
void DumpOptions(const LrtOpT& op, std::ostream& out = std::cerr);

//
// Library Utilities
//

// Dumps details about the loaded LrtCompilerPlugin library.
void Dump(const CompilerPlugin& plugin, std::ostream& out = std::cerr);

// Dumps details about the dynamic library (see "dlinfo").
void Dump(void* lib_handle, std::ostream& out = std::cerr);

}  // namespace lrt::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_DUMP_H_
