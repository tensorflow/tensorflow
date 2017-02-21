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

#ifndef TENSORFLOW_COMPILER_AOT_COMPILE_H_
#define TENSORFLOW_COMPILER_AOT_COMPILE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/aot/tfcompile.pb.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace tfcompile {

// Constants for op types and attribute names.
extern const char* const kArgOp;
extern const char* const kRetvalOp;
extern const char* const kFeedIdAttr;
extern const char* const kFetchIdAttr;
extern const char* const kShapeAttr;
extern const char* const kDebugNameAttr;

// InitGraph creates a graph based on the graph_def, that may then be compiled
// by CompileGraph.
//
// The graph is rewritten with _Arg and _Retval nodes, representing the inputs
// and outputs of the function that will be compiled.  Each feed id causes a new
// _Arg node to be created, where we first collect all existing edges pointing
// from the named node's output index, and then rewrite them to point from that
// _Arg node instead.  Each fetch id causes a new _Retval node to be created,
// with a new edge pointing from the named node's output index to that _Retval
// node.  All _Retval nodes also point to a special CompileExpressions node,
// used internally to finish the compilation.
//
// The rewritten graph is then pruned to only contain the portion necessary to
// compute the outputs.  If dump_graphs is true, graph rewrites will be dumped
// for debugging.
Status InitGraph(const GraphDef& graph_def, const Config& config,
                 const MainFlags& flags, const FunctionLibraryDefinition* flib,
                 std::unique_ptr<Graph>* graph);

// CompileResult describes the output of CompileGraph, where the object file
// data and meta-information is available in aot.
struct CompileResult {
  // Contains object file and meta-info.
  std::unique_ptr<xla::cpu::CpuAotCompilationResult> aot;
  xla::ProgramShape program_shape;  // Static shape of args and results.
  bool has_context_arg = false;     // Is last arg XlaLocalRuntimeContext?
  string entry_point;               // Name of generated function.
  int pointer_size = 0;             // Size of a pointer in bytes.
};

// CompileGraph compiles the graph into an object file containing a function
// that performs the graph operations.
//
// The graph must have _Arg and _Retval nodes representing the function inputs
// and outputs.  Every _Arg node must have a shape attribute (key=kShapeAttr,
// value=TensorShape) representing the static shape of that input, and every
// _Retval node must point to a CompileExpressions node.
//
// Typically InitGraph is called to perform this initialization, followed by
// full specification of the shape attributes.
//
// The XLA compilation options are specified in the flags.
Status CompileGraph(std::unique_ptr<Graph> graph, const MainFlags& flags,
                    const FunctionLibraryDefinition* flib,
                    CompileResult* result);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_COMPILE_H_
