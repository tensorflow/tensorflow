/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.h"

#include <string>
#include <typeinfo>

#include "absl/strings/string_view.h"
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"

// Generated dialect defs.
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.cc.inc"

namespace mlir {
namespace dtensor {

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void DTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc.inc"
      >();
  registerAttributes();
}

// Parses a #dtensor.mesh attribute of the following format:
//
//   #dtensor.mesh<serializedMesh>
//
// where the first element is a SymbolRefAttr and the second element is the
// location.
static MeshAttr ParseMeshAttr(MLIRContext *context, StringRef spec,
                              Location loc) {
  // Define error function.
  auto emit_error = [&](std::string text) {
    emitError(loc, "invalid TensorFlow Mesh attribute ") << text;
    return nullptr;
  };
  // Check correct format and consume prefix, otherwise throw error.
  if (!spec.consume_front("mesh<"))
    return emit_error("Unexpected start to mesh specification");

  // Consume back from ">".
  if (!spec.consume_back(">"))
    return emit_error("Unexpected closing of mesh specification");

  // Cast from StringRef to string.
  std::string mesh_str = spec.str();

  // Check if serializedMesh is correct.
  using Mesh = tensorflow::dtensor::Mesh;
  using MeshOr = tensorflow::dtensor::StatusOr<Mesh>;
  MeshOr mesh_or = Mesh::FromString(mesh_str);
  if (!mesh_or.ok()) {
    std::string status_msg = mesh_or.status().ToString();
    return emit_error("parsing serialized string. More details: " + status_msg);
  }
  return MeshAttr::get(context, mesh_or.ValueOrDie());
}

// Parses a #dtensor.layout attribute of the following format:
//
//   #dtensor.layout<serializedLayout>
static LayoutAttr ParseLayoutAttr(MLIRContext *context, StringRef spec,
                                  Location loc) {
  // Define error function.
  auto emit_error = [&](std::string text) {
    emitError(loc, "invalid TensorFlow Mesh attribute ") << text;
    return nullptr;
  };
  // Check correct format and consume prefix, otherwise throw error.
  if (!spec.consume_front("layout<"))
    return emit_error("Unexpected start to layout specification");

  // Consume back from "\">".
  if (!spec.consume_back(">"))
    return emit_error("Unexpected closing of layout specification");

  // Cast into string
  std::string layout_str = spec.str();

  // Check if serializedMesh is correct, else error from line 37.
  using Layout = tensorflow::dtensor::Layout;
  using LayoutOr = tensorflow::dtensor::StatusOr<Layout>;
  LayoutOr layout_or = Layout::FromString(layout_str);
  if (!layout_or.ok()) {
    std::string status_msg = layout_or.status().ToString();
    return emit_error("parsing serialized string. More details: " + status_msg);
  }
  // Extract layout.
  Layout layout = layout_or.ValueOrDie();

  return LayoutAttr::get(context, layout);
}

Attribute DTensorDialect::parseAttribute(DialectAsmParser &parser,
                                         Type type) const {
  StringRef spec = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  if (spec.startswith("mesh")) return ParseMeshAttr(getContext(), spec, loc);

  if (spec.startswith("layout"))
    return ParseLayoutAttr(getContext(), spec, loc);

  return (emitError(loc, "unknown DTensor attribute: " + spec), nullptr);
}

// Print a type registered to this dialect.
// Prints a #dtensor.dtensor attribute of the following format:
//
//   #dtensor.mesh<mesh>
static void printMeshAttr(MeshAttr attr, DialectAsmPrinter &os) {
  os << "mesh<" << attr.getValue().ToString() << ">";
}

// Prints a #dtensor.dtensor attribute of the following format:
//
//   #dtensor.layout<layout>
static void printLayoutAttr(LayoutAttr attr, DialectAsmPrinter &os) {
  os << "layout<" << attr.getValue().ToString() << ">";
}

// Override general virtual function
void DTensorDialect::printAttribute(Attribute attr,
                                    DialectAsmPrinter &os) const {
  // Cast into correct attribute and print
  if (auto mesh_attr = attr.dyn_cast<MeshAttr>()) printMeshAttr(mesh_attr, os);

  if (auto layout_attr = attr.dyn_cast<LayoutAttr>())
    printLayoutAttr(layout_attr, os);
}
}  // namespace dtensor
}  // namespace mlir

// Ops definition from ODS

#define GET_OP_CLASSES
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc.inc"
