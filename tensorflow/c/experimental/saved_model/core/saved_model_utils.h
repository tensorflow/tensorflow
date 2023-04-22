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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_

// Some internal utility functions for the SavedModelAPI, factored out into a
// separately unit-testable header.

#include <memory>
#include <unordered_map>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/asset.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {
namespace internal {

// Load a TensorProto into a tensorflow::Constant. This is similar to the
// constant loading logic in python:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L437
Status TensorProtoToConstant(ImmediateExecutionContext* ctx,
                             const TensorProto& proto,
                             std::unique_ptr<Constant>* output);

// Creates a tensorflow::Variable from a SavedVariable. This is similar to the
// logic in:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L407
// Note that the caller **must assign a value** to the loaded variable.
Status LoadSavedVariable(ImmediateExecutionContext* ctx,
                         const SavedVariable& variable,
                         std::unique_ptr<Variable>* output);

Status LoadSavedAsset(ImmediateExecutionContext* ctx, const SavedAsset& asset,
                      const std::string& saved_model_dir,
                      absl::Span<const AssetFileDef> assets,
                      std::unique_ptr<Asset>* output);

// Creates a TFConcreteFunction from a SavedConcreteFunction.
Status LoadTFConcreteFunction(
    const SavedConcreteFunction& saved_concrete_function,
    const FunctionDef* function_def,
    const std::unordered_map<int, std::unique_ptr<TensorHandleConvertible>>&
        captured_objects,
    ImmediateExecutionContext* ctx, std::unique_ptr<TFConcreteFunction>* out);

// Flattens `signature` into a vector of TensorSpecProto pointers back into
// `signature`. `signature` must outlive flattened_specs. `signature` must also
// be the input or output signature of a SavedConcreteFunction (i.e. "nested
// structures of tensorspecs").
Status FlattenSignature(const StructuredValue& signature,
                        std::vector<const TensorSpecProto*>* flattened_specs);

// Find the node id in `object_graph` at location `path`. `path` must be
// a dot-delimited string of object names relative to the root object. If no
// object is found, returns absl::nullopt.
absl::optional<int> FindNodeAtPath(StringPiece path,
                                   const SavedObjectGraph& object_graph);

// Maps each node in `graphdef` to its corresponding Attribute Map.
// Callers must ensure that `graphdef` outlives the returned map.
gtl::FlatMap<StringPiece, const AttrValueMap*, StringPieceHasher> NodeToAttrMap(
    const tensorflow::GraphDef& graphdef);

// Maps the name of each FunctionDef in `library` to its corresponding
// FunctionDef. Callers must ensure `library` outlives the returned map.
gtl::FlatMap<StringPiece, const tensorflow::FunctionDef*, StringPieceHasher>
FunctionNameToFunctionDefMap(const FunctionDefLibrary& library);

// Walks through the SavedObjectGraph in metagraph, and restores all nodes
// (except "UserDefinedObjects") with their corresponding type in
// "PartiallyRevivedObjects".
Status PartiallyReviveSavedModelObjects(const MetaGraphDef& metagraph,
                                        ImmediateExecutionContext* context,
                                        const std::string& directory,
                                        PartiallyRevivedObjects* objects);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_
