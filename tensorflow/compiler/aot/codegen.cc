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

#include "tensorflow/compiler/aot/codegen.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/aot/embedded_constant_buffers.h"
#include "tensorflow/compiler/aot/embedded_protocol_buffers.h"
#include "tensorflow/compiler/aot/thunk_proto_execution_deserializer.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/cpu/buffer_info_util.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tsl/platform/casts.h"

namespace tensorflow {
namespace tfcompile {

namespace {

using BufferInfo = xla::cpu_function_runtime::BufferInfo;

bool IsAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAlphaNum(char c) { return IsAlpha(c) || (c >= '0' && c <= '9'); }

// Convert an XLA type into a C++ type.
absl::Status XLATypeToCpp(xla::PrimitiveType type, string* str) {
  switch (type) {
    case xla::PRED:
      *str = "bool";
      break;
    case xla::S8:
      *str = "tensorflow::int8";
      break;
    case xla::S16:
      *str = "tensorflow::int16";
      break;
    case xla::S32:
      *str = "tensorflow::int32";
      break;
    case xla::S64:
      *str = "int64_t";
      break;
    case xla::U8:
      *str = "tensorflow::uint8";
      break;
    case xla::U16:
      *str = "tensorflow::uint16";
      break;
    case xla::U32:
      *str = "tensorflow::uint32";
      break;
    case xla::U64:
      *str = "tensorflow::uint64";
      break;
    case xla::F32:
      *str = "float";
      break;
    case xla::F64:
      *str = "double";
      break;
    default:
      return errors::Unimplemented("XLA type ", xla::PrimitiveType_Name(type),
                                   " has no equivalent in C++");
  }
  return absl::OkStatus();
}

// Returns the sum of the size of each buffer in `buffer_infos`.
size_t TotalBufferBytes(const std::vector<BufferInfo>& buffer_infos) {
  return std::accumulate(buffer_infos.begin(), buffer_infos.end(), size_t{0},
                         [](size_t size, const BufferInfo& buffer_info) {
                           return size + buffer_info.size();
                         });
}

// Returns a vector of BufferInfo instances in `buffer_infos` that are entry
// parameter buffers.
std::vector<BufferInfo> ExtractEntryParamBufferInfos(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<BufferInfo> result;
  std::copy_if(buffer_infos.begin(), buffer_infos.end(),
               std::back_inserter(result), [](const BufferInfo& buffer_info) {
                 return buffer_info.is_entry_parameter();
               });
  return result;
}

// Returns a vector of BufferInfo instances in `buffer_infos` that are temp
// buffers.
std::vector<BufferInfo> ExtractTempBufferInfos(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<BufferInfo> result;
  std::copy_if(buffer_infos.begin(), buffer_infos.end(),
               std::back_inserter(result), [](const BufferInfo& buffer_info) {
                 return buffer_info.is_temp_buffer();
               });
  return result;
}

// Add (from,to) rewrite pairs based on the given shape.  These rewrite pairs
// are used to generate methods for args and results.
absl::Status AddRewritesForShape(
    int i, const xla::Shape& shape,
    std::vector<std::pair<string, string>>* rewrites) {
  string type;
  TF_RETURN_IF_ERROR(XLATypeToCpp(shape.element_type(), &type));
  std::vector<string> dim_vars;
  string dim_sizes, indices;
  int count = 1;
  if (shape.dimensions().size() == 0 ||
      (shape.dimensions().size() == 1 && shape.dimensions(0) == 1)) {
    dim_sizes = "[1]";
    indices = "[0]";
  } else {
    for (int dim = 0; dim < shape.dimensions().size(); ++dim) {
      dim_vars.push_back(absl::StrCat("size_t dim", dim));
      dim_sizes += absl::StrCat("[", shape.dimensions(dim), "]");
      indices += absl::StrCat("[dim", dim, "]");
      count *= shape.dimensions(dim);
    }
  }
  rewrites->push_back({"{{I}}", absl::StrCat(i)});
  rewrites->push_back({"{{TYPE}}", type});
  rewrites->push_back({"{{DIM_VARS}}", absl::StrJoin(dim_vars, ", ")});
  rewrites->push_back({"{{DIM_SIZES}}", dim_sizes});
  rewrites->push_back({"{{INDICES}}", indices});
  rewrites->push_back({"{{COUNT}}", absl::StrCat(count)});
  return absl::OkStatus();
}

// Returns code rewritten by replacing all rewrite pairs, with an extra rewrite
// for the name.  Note that the rewriting strategy is roughly O(N*M), where N is
// the size of the code and M is the number of rewrites.  It's fine for now
// since N and M are pretty small.
//
// TODO(toddw): If this becomes a problem, we should be able to change the
// algorithm to O(N) by using a state machine, e.g. regexps or a real
// text-templating mechanism.
string RewriteWithName(const string& name, string code,
                       const std::vector<std::pair<string, string>>& rewrites) {
  absl::StrReplaceAll(rewrites, &code);
  absl::StrReplaceAll({{"{{NAME}}", name}}, &code);
  return code;
}

// Generate methods for args (inputs).
absl::Status GenArgMethods(const tf2xla::Config& config,
                           const xla::ProgramShapeProto& ps,
                           const CompileResult& compile_result,
                           string* methods) {
  const int num_args = ps.parameters_size();
  // feed_size() + variable_size() is the maximum number of args as an
  // implementation may not create an argument for an unused variable.
  if (config.feed_size() + config.variable_size() < num_args) {
    return errors::InvalidArgument(
        "mismatch between feed_size(", config.feed_size(), ")+variable_size(",
        config.variable_size(), ") and num_args(", num_args, ")");
  }
  for (int i = 0; i < config.feed_size(); ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        xla::Shape::FromProto(ps.parameters(i)));
    TF_RETURN_IF_ERROR(AddRewritesForShape(i, shape, &rewrites));
    const string code = R"(
  void set_arg{{NAME}}_data(const void* data) {
    set_arg_data({{I}}, data);
  }
  {{TYPE}}* arg{{NAME}}_data() {
    return static_cast<{{TYPE}}*>(arg_data({{I}}));
  }
  {{TYPE}}& arg{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* arg{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(arg_data({{I}}));
  }
  const {{TYPE}}& arg{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  int arg{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int arg{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    *methods += RewriteWithName(absl::StrCat(i), code, rewrites);
    if (!config.feed(i).name().empty()) {
      *methods += RewriteWithName("_" + config.feed(i).name(), code, rewrites);
    }
  }
  return absl::OkStatus();
}

// Generate methods for results (outputs).
absl::Status GenResultMethods(const tf2xla::Config& config,
                              const xla::ProgramShapeProto& ps,
                              string* methods) {
  if (ps.result().element_type() != xla::TUPLE) {
    // The XlaCompiler we use to build the xla computation always generates a
    // tuple result, and we rely on this to simplify code generation.
    return xla::Internal("codegen requires the XLA result to be a tuple");
  }
  size_t num_results = ps.result().tuple_shapes_size();
  int readonly_variables = absl::c_count_if(
      config.variable(),
      [](const tf2xla::Variable& var) { return var.readonly(); });
  const int actual_num_results =
      config.fetch_size() + config.variable_size() - readonly_variables;
  if (actual_num_results != num_results) {
    return errors::InvalidArgument("mismatch between fetch_size(",
                                   config.fetch_size(), ")+variable_size(",
                                   config.variable_size(), ") and tuple_size(",
                                   ps.result().tuple_shapes_size(), ")");
  }
  for (int i = 0; i < config.fetch_size(); ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        xla::Shape::FromProto(ps.result().tuple_shapes(i)));
    TF_RETURN_IF_ERROR(AddRewritesForShape(i, shape, &rewrites));
    string code = R"(
  {{TYPE}}* result{{NAME}}_data() {
    return static_cast<{{TYPE}}*>(result_data({{I}}));
  }
  {{TYPE}}& result{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{TYPE}}(*){{DIM_SIZES}}>(
        result_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* result{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(result_data({{I}}));
  }
  const {{TYPE}}& result{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        result_data({{I}}))){{INDICES}};
  }
  int result{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int result{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    *methods += RewriteWithName(absl::StrCat(i), code, rewrites);
    if (!config.fetch(i).name().empty()) {
      *methods += RewriteWithName("_" + config.fetch(i).name(), code, rewrites);
    }
  }
  return absl::OkStatus();
}

// Generate methods for variables.
absl::Status GenVariableMethods(const tf2xla::Config& config,
                                const xla::ProgramShapeProto& ps,
                                string* methods) {
  const int num_args = ps.parameters_size();
  for (int i = config.feed_size(); i < num_args; ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        xla::Shape::FromProto(ps.parameters(i)));
    TF_RETURN_IF_ERROR(AddRewritesForShape(i, shape, &rewrites));
    const string code = R"(
  void set_var_{{NAME}}_data({{MAYBE_CONST}}{{TYPE}}* data) {
    set_arg_data({{I}}, data);
  }
  {{MAYBE_CONST}}{{TYPE}}* var_{{NAME}}_data() {
    return static_cast<{{MAYBE_CONST}}{{TYPE}}*>(arg_data({{I}}));
  }
  {{MAYBE_CONST}}{{TYPE}}& var_{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{MAYBE_CONST}}{{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* var_{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(arg_data({{I}}));
  }
  const {{TYPE}}& var_{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  int var_{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int var_{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    const tf2xla::Variable& var = config.variable(i - config.feed_size());
    rewrites.emplace_back("{{MAYBE_CONST}}", var.readonly() ? "const " : "");
    *methods += RewriteWithName(
        var.name().empty() ? var.node_name() : var.name(), code, rewrites);
  }
  return absl::OkStatus();
}

// Generate shape infos for args (inputs).
absl::Status GenArgShapeInfos(const xla::ProgramShapeProto& ps, string* infos) {
  for (int i = 0; i < ps.parameters_size(); ++i) {
    const xla::ShapeProto& shape = ps.parameters(i);
    if (shape.element_type() == xla::TUPLE) {
      // ShapeInfo cannot represent tuple args.
      return absl::InternalError(
          absl::StrCat("parameter ", i,
                       ": codegen requires XLA parameters to "
                       "be non-tuples."));
    }
    // Please some compilers (e.g. MSVC) by avoiding the initialization of an
    // array of unknown size an empty initializer. Use "-1" for this; note that
    // this value is never used (the size attribute is set to 0 in ShapeInfo).
    *infos += absl::Substitute(R"(  static constexpr int32_t kArg$0Shapes[] = {
$1
  };
)",
                               i,
                               shape.dimensions_size() > 0
                                   ? absl::StrJoin(shape.dimensions(), ", ")
                                   : "-1");
  }
  *infos += R"(  static const ShapeInfo* ArgShapeInfos() {
    static constexpr ShapeInfo kArgShapeInfoTable[kNumArgs] = {
)";
  for (int i = 0; i < ps.parameters_size(); ++i) {
    const xla::ShapeProto& shape = ps.parameters(i);
    *infos +=
        absl::Substitute("{ kArg$0Shapes, $1 },\n", i, shape.dimensions_size());
  }
  *infos += R"(    };
    return kArgShapeInfoTable;
  })";
  return absl::OkStatus();
}

// Generate shape infos for results.
absl::Status GenResultShapeInfos(const xla::ProgramShapeProto& ps,
                                 string* infos) {
  if (ps.result().element_type() != xla::TUPLE) {
    return absl::InternalError("codegen requires the XLA result to be a tuple");
  }
  for (int i = 0; i < ps.result().tuple_shapes_size(); ++i) {
    const xla::ShapeProto& shape = ps.result().tuple_shapes(i);
    // See above comment about the use here of "-1".
    *infos += absl::Substitute(
        R"(  static constexpr int32_t kResult$0Shapes[] = {
$1
  };
)",
        i,
        shape.dimensions_size() > 0 ? absl::StrJoin(shape.dimensions(), ", ")
                                    : "-1");
  }
  *infos += R"(  static const ShapeInfo* ResultShapeInfos() {
    static constexpr ShapeInfo kResultShapeInfoTable[kNumResults] = {
)";
  for (int i = 0; i < ps.result().tuple_shapes_size(); ++i) {
    const xla::ShapeProto& shape = ps.result().tuple_shapes(i);
    *infos += absl::Substitute("{ kResult$0Shapes, $1 },\n", i,
                               shape.dimensions_size());
  }
  *infos += R"(    };
    return kResultShapeInfoTable;
  })";
  return absl::OkStatus();
}

// Generates code implementing {Arg,Result}Names(), where T is one of
// tf2xla::{Feed,Fetch,Variable}. Each feed or fetch name results in a C-style
// string literal in the array, with nullptr terminating the array.
template <typename T>
string GenNameToIndexCode(const T& entries, bool generate) {
  // No need for a static array if we're not supposed to generate the data.
  if (!generate) {
    return "{\n    return nullptr;\n  }";
  }
  // Determine when to stop. We stop emitting string literals after the last
  // non-empty name.
  int end = entries.size();
  for (int i = entries.size() - 1; i >= 0; --i) {
    if (!entries[i].name().empty()) {
      break;
    }
    end = i;
  }
  // Emit string literals up to the last non-empty name.
  string code = "{\n    static const char* kNames[] = {";
  for (int i = 0; i < end; ++i) {
    if (i > 0) {
      code += ", ";
    }
    code += "\"";
    code += entries[i].name();
    code += "\"";
  }
  if (end > 0) {
    code += ", ";
  }
  code += "nullptr};\n    return kNames;\n  }";
  return code;
}

absl::Status ValidateFeedFetchCppNames(const tf2xla::Config& config) {
  for (const tf2xla::Feed& feed : config.feed()) {
    if (!feed.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(feed.name(), "feed name"));
    }
  }
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    if (!fetch.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(fetch.name(), "fetch name"));
    }
  }
  for (const tf2xla::Variable& variable : config.variable()) {
    if (!variable.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(variable.name(), "variable name"));
    } else {
      TF_RETURN_IF_ERROR(
          ValidateCppIdent(variable.node_name(), "variable name"));
    }
  }
  return absl::OkStatus();
}

// Returns a list of C++ expressions that, when executed, will construct the
// BufferInfo instances in `buffer_infos`.
std::vector<string> BufferInfosToCppExpression(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<string> buffer_infos_as_strings;
  std::transform(buffer_infos.begin(), buffer_infos.end(),
                 std::back_inserter(buffer_infos_as_strings),
                 [](const BufferInfo& buffer_info) {
                   xla::cpu_function_runtime::EncodedBufferInfo encoded =
                       buffer_info.Encode();
                   auto param_to_str = [](uint32_t param) -> std::string {
                     return param == ~0U ? "~0U" : absl::StrCat(param, "U");
                   };
                   return absl::StrCat(
                       "::xla::cpu_function_runtime::BufferInfo("
                       "::xla::cpu_function_runtime::EncodedBufferInfo{",
                       encoded.packed_kind_and_size, "ULL, ",
                       param_to_str(encoded.entry_param_number), ", ",
                       param_to_str(encoded.result_param_number), "})");
                 });
  return buffer_infos_as_strings;
}

absl::Status CheckEqual(size_t a, size_t b, absl::string_view error_msg) {
  if (a != b) {
    return absl::InternalError(
        absl::StrCat(error_msg, ". Expected ", a, ", got ", b, "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GenFunctionDeclaration(
    const xla::cpu::SymbolProto& symbol) {
  std::string function_declaration;

  switch (symbol.function_type_id()) {
    case xla::cpu::SymbolProto::COMPARATOR:
      absl::StrAppend(
          &function_declaration, " void ", symbol.name(),
          "(bool* result, const void* run_options, const void** params, "
          "const void* buffer_table, const void* status, "
          "const void* prof_counters);");
      break;
    case xla::cpu::SymbolProto::KERNEL:
      absl::StrAppend(&function_declaration, " XLA_CPU_KernelError* ",
                      symbol.name(),
                      "(const XLA_CPU_KernelCallFrame* call_frame);");
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported symbol kind: ", symbol.function_type_id()));
  }
  return function_declaration;
}

absl::StatusOr<std::string> GenFunctionDeclarations(
    absl::Span<xla::cpu::SymbolProto> compiled_symbols) {
  std::string function_declarations =
      !compiled_symbols.empty() ? "extern \"C\" {\n" : "";
  for (const auto& symbol : compiled_symbols) {
    TF_ASSIGN_OR_RETURN(std::string function_declaration,
                        GenFunctionDeclaration(symbol));
    absl::StrAppend(&function_declarations, function_declaration, "\n");
  }
  if (!compiled_symbols.empty()) {
    function_declarations += "}\n";
  }
  return function_declarations;
}

absl::StatusOr<std::string> GenSymbolMapInitializerString(
    absl::Span<xla::cpu::SymbolProto> entry_point_symbols) {
  std::string symbol_map_initializer = "{";
  for (const auto& symbol : entry_point_symbols) {
    absl::StrAppend(&symbol_map_initializer,
                    "  std::pair<std::string, void*>{\"", symbol.name(),
                    "\", reinterpret_cast<void*>(", symbol.name(), ")},");
  }
  symbol_map_initializer += "}";
  return symbol_map_initializer;
}

std::vector<xla::cpu::SymbolProto> ExtractEntryPointSymbols(
    const xla::cpu::CompilationResultProto& proto) {
  std::vector<xla::cpu::SymbolProto> compiled_symbols;
  for (const auto& symbol : proto.compiled_symbols()) {
    compiled_symbols.push_back(symbol);
  }

  std::vector<xla::cpu::SymbolProto> entry_point_symbols;

  auto check_if_compiled_symbol_and_add_entry_point =
      [](const std::string& symbol_name,
         const std::vector<xla::cpu::SymbolProto>& compiled_symbols,
         std::vector<xla::cpu::SymbolProto>& entry_point_symbols) {
        auto it = std::find_if(compiled_symbols.begin(), compiled_symbols.end(),
                               [&symbol_name](const auto& symbol) {
                                 return symbol.name() == symbol_name;
                               });

        if (it != compiled_symbols.end()) {
          entry_point_symbols.push_back(*it);
        }
      };

  xla::cpu::ForEachThunkProto(
      proto.thunk_sequence(), [&entry_point_symbols, &compiled_symbols,
                               &check_if_compiled_symbol_and_add_entry_point](
                                  const xla::cpu::ThunkProto& thunk) {
        if (thunk.has_kernel_thunk()) {
          auto symbol_name = thunk.kernel_thunk().kernel_name();
          check_if_compiled_symbol_and_add_entry_point(
              symbol_name, compiled_symbols, entry_point_symbols);
        } else if (thunk.has_sort_thunk()) {
          auto symbol_name = thunk.sort_thunk().comparator_name();
          check_if_compiled_symbol_and_add_entry_point(
              symbol_name, compiled_symbols, entry_point_symbols);
        }
      });

  return entry_point_symbols;
}

bool HasThunkKind(const xla::cpu::ThunkSequenceProto& thunk_sequence,
                  xla::cpu::ThunkProto::ImplCase kind) {
  bool has_kind = false;
  xla::cpu::ForEachThunkProto(
      thunk_sequence, [&kind, &has_kind](const xla::cpu::ThunkProto& thunk) {
        if (thunk.impl_case() == kind) {
          has_kind = true;
        }
      });

  return has_kind;
}

absl::StatusOr<std::vector<xla::cpu::ThunkProto>> ExtractThunksOfKind(
    const xla::cpu::ThunkSequenceProto& thunk_sequence,
    xla::cpu::ThunkProto::ImplCase kind) {
  std::vector<xla::cpu::ThunkProto> matching_thunks;

  xla::cpu::ForEachThunkProto(
      thunk_sequence,
      [&kind, &matching_thunks](const xla::cpu::ThunkProto& thunk) {
        if (thunk.impl_case() == kind) {
          matching_thunks.push_back(thunk);
        }
      });

  return matching_thunks;
}

absl::StatusOr<std::string> GetThunkSpecificConstantAllocationsInitializers(
    const xla::cpu::CompilationResultProto& proto,
    const EmbeddedConstantBuffers& embedded_constant_buffers) {
  std::vector<absl::string_view> constant_buffer_accesses;
  constant_buffer_accesses.reserve(
      embedded_constant_buffers.variable_decls.size());
  for (const auto& variable_decl : embedded_constant_buffers.variable_decls) {
    constant_buffer_accesses.push_back(variable_decl.cpp_access_shim);
  }
  std::string constant_allocations_initializer = absl::StrCat(
      R"(
{
)",
      absl::StrJoin(constant_buffer_accesses, ", "),
      R"(
};
)");

  return constant_allocations_initializer;
}

absl::Status ExtendRewrites(
    std::vector<std::pair<std::string, std::string>>& rewrites,
    const xla::cpu::CpuAotCompilationResultThunks* aot_thunks,
    const MetadataResult& metadata_result, const CodegenOpts& codegen_opts,
    const EmbeddedConstantBuffers& embedded_constant_buffers) {
  std::vector<xla::cpu::SymbolProto> entry_point_symbols =
      ExtractEntryPointSymbols(aot_thunks->proto());

  TF_ASSIGN_OR_RETURN(
      const std::string symbol_map_initializer,
      GenSymbolMapInitializerString(absl::MakeSpan(entry_point_symbols)));

  TF_ASSIGN_OR_RETURN(
      const std::string function_declarations_from_obj_files,
      GenFunctionDeclarations(absl::MakeSpan(entry_point_symbols)));

  const int64_t buffer_infos_size = aot_thunks->buffer_infos().size();
  const std::optional<size_t> temp_allocation_index =
      aot_thunks->temp_allocation_index();
  if (temp_allocation_index.has_value() &&
      (*temp_allocation_index < 0 ||
       *temp_allocation_index >= buffer_infos_size)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "temp allocation index: ", *temp_allocation_index,
        " is outside the range of temp sizes: [0,", buffer_infos_size, ")"));
  }

  std::vector<std::string> runtime_specific_includes = {R"(
#include "absl/log/check.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/types.h")"};

  if (HasThunkKind(aot_thunks->proto().thunk_sequence(),
                   xla::cpu::ThunkProto::kDotThunk)) {
    runtime_specific_includes.push_back(
        R"(#include "xla/service/cpu/runtime_matmul.h")");
    runtime_specific_includes.push_back(
        R"(#include "xla/service/cpu/runtime_single_threaded_matmul.h")");
  }

  if (HasThunkKind(aot_thunks->proto().thunk_sequence(),
                   xla::cpu::ThunkProto::kConvolutionThunk)) {
    runtime_specific_includes.push_back(
        R"(#include "xla/service/cpu/runtime_conv2d.h")");

    runtime_specific_includes.push_back(
        R"(#include "xla/service/cpu/runtime_single_threaded_conv2d.h")");
  }

  if (HasThunkKind(aot_thunks->proto().thunk_sequence(),
                   xla::cpu::ThunkProto::kSortThunk)) {
    runtime_specific_includes.push_back(
        R"(#include "xla/service/cpu/runtime_key_value_sort.h")");
  }

  TF_ASSIGN_OR_RETURN(
      const auto rng_thunks,
      ExtractThunksOfKind(aot_thunks->proto().thunk_sequence(),
                          xla::cpu::ThunkProto::kRngGetAndUpdateStateThunk));

  std::string executable_proto_getter = "";
  std::string thunk_specific_run_impl = "";
  std::string nanort_specific_static_data_setters = "";
  std::string thunk_specific_static_data_setters = "";
  std::string thunk_run_impl_getter = "";
  std::string embedded_constant_buffers_initializer_getter = "";
  std::string rng_deltas_initializer_getter = "";
  std::string computation_class_base = "";

  if (codegen_opts.use_xla_nanort_runtime) {
    executable_proto_getter = absl::StrReplaceAll(
        R"(static const ::xla::cpu::CompilationResultProto* StaticCompilationResultProto() {
    static const ::xla::cpu::CompilationResultProto* kCompilationResultProto = {{EXECUTABLE_PROTO_SHIM_EXPRESSION}};
    return kCompilationResultProto;
  })",
        {{"{{EXECUTABLE_PROTO_SHIM_EXPRESSION}}",
          metadata_result.cpu_executable_access_shim}});

    runtime_specific_includes.push_back(
        R"(#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function_thunks.h")");

    nanort_specific_static_data_setters =
        "set_static_data_compilation_result_proto(data, "
        "StaticCompilationResultProto());";

    computation_class_base = "XlaCompiledCpuFunctionThunks";
  } else {
    ThunkProtoExecutionDeserializer thunk_proto_execution_deserializer;
    TF_ASSIGN_OR_RETURN(const std::string run_impl,
                        std::move(thunk_proto_execution_deserializer)
                            .GetThunkSpecificRunImpl(aot_thunks->proto()));

    nanort_specific_static_data_setters =
        "set_static_data_compilation_result_proto(data, nullptr);";

    computation_class_base = "XlaCompiledCpuFunction";

    thunk_specific_static_data_setters =
        R"(set_static_data_thunk_run_impl(data, ThunkRunImplFunction());
        set_static_data_embedded_constant_buffers(data, EmbeddedConstantBuffers());
        set_static_data_rng_state_deltas(data, RngStateDeltas());)";

    std::vector<int64_t> rng_deltas;
    rng_deltas.reserve(rng_thunks.size());
    for (const auto& rng_thunk : rng_thunks) {
      rng_deltas.push_back(rng_thunk.rng_get_and_update_state_thunk().delta());
    }
    rng_deltas_initializer_getter = absl::StrReplaceAll(
        R"(
  static std::vector<int64_t> RngStateDeltas() {
    return {
      {{RNG_DELTAS}}
    };
  }
  )",
        {{"{{RNG_DELTAS}}", absl::StrJoin(rng_deltas, ", ")}});

    thunk_run_impl_getter = absl::StrReplaceAll(
        R"(
  static std::function<bool(
        void** buffer_table, xla::ExecutableRunOptions*,
        std::vector<std::unique_ptr<xla::cpu::RngState>>&)>
  ThunkRunImplFunction() {
    return [](void** buffer_table, xla::ExecutableRunOptions* run_options, 
    std::vector<std::unique_ptr<xla::cpu::RngState>>& rng_states) {
      {{THUNK_RUN_IMPL}}
      return true;
    };
  }
  )",
        {{"{{THUNK_RUN_IMPL}}", run_impl}});
  }

  auto embedded_constant_buffers_initializer_getter_format = R"(
  static std::vector<std::pair<uint64_t, char*>> EmbeddedConstantBuffers() {
    return {{EMBEDDED_CONSTANT_BUFFERS}};
  }
  )";

  TF_ASSIGN_OR_RETURN(auto const_buffs_initializer,
                      GetThunkSpecificConstantAllocationsInitializers(
                          aot_thunks->proto(), embedded_constant_buffers));
  embedded_constant_buffers_initializer_getter = absl::StrReplaceAll(
      embedded_constant_buffers_initializer_getter_format,
      {{"{{EMBEDDED_CONSTANT_BUFFERS}}", const_buffs_initializer}});

  std::vector<std::pair<std::string, std::string>> rewrites_thunks = {
      {"{{SYMBOL_MAP_INITIALIZER}}", symbol_map_initializer},
      {"{{FUNCTION_DECLARATIONS_FROM_OBJ_FILES}}",
       function_declarations_from_obj_files},
      {"{{TEMP_ALLOCATION_INDEX}}", temp_allocation_index.has_value()
                                        ? absl::StrCat(*temp_allocation_index)
                                        : "std::nullopt"},
      {"{{RUNTIME_SPECIFIC_INCLUDES}}",
       absl::StrJoin(runtime_specific_includes, "\n")},
      {"{{EXECUTABLE_PROTO_GETTER}}", executable_proto_getter},
      {"{{THUNK_SPECIFIC_STATIC_DATA_SETTERS}}",
       thunk_specific_static_data_setters},
      {"{{THUNK_RUN_IMPL_GETTER}}", thunk_run_impl_getter},
      {"{{EMBEDDED_CONSTANT_BUFFERS_INITIALIZER_GETTER}}",
       embedded_constant_buffers_initializer_getter},
      {"{{RNG_DELTAS_INITIALIZER_GETTER}}", rng_deltas_initializer_getter},
      {"{{NANORT_SPECIFIC_STATIC_DATA_SETTERS}}",
       nanort_specific_static_data_setters},
      {"{{IS_THUNK_MODE}}", "true"},
      {"{{COMPUTATION_CLASS_BASE}}", computation_class_base},
      // TODO(basioli): Remove these once legacy runtime is removed.
      {"{{LEGACY_SPECIFIC_STATIC_DATA_SETTERS}}", ""},
      {"{{LEGACY_SPECIFIC_STATIC_DATA_GENERATORS}}", ""}};

  rewrites.insert(rewrites.end(), rewrites_thunks.begin(),
                  rewrites_thunks.end());

  return absl::OkStatus();
}

}  // namespace

absl::Status ExtendRewrites(
    std::vector<std::pair<std::string, std::string>>& rewrites,
    const xla::cpu::CpuAotCompilationResultLegacy* aot_legacy,
    const MetadataResult& metadata_result, const CodegenOpts& opts,
    const std::string& raw_function_entry_identifier) {
  const int64_t result_index = aot_legacy->result_buffer_index();
  const int64_t buffer_infos_size = aot_legacy->buffer_infos().size();
  if (result_index < 0 || result_index >= buffer_infos_size) {
    return errors::InvalidArgument("result index: ", result_index,
                                   " is outside the range of temp sizes: [0,",
                                   buffer_infos_size, ")");
  }

  const string include_hlo_profile_printer_data_proto =
      opts.gen_hlo_profile_printer_data
          ? R"(#include "xla/service/hlo_profile_printer_data.pb.h")"
          : "";

  // When HLO profiling is disabled we only forward declare the
  // HloProfilePrinter protobuf.  So we can only conditionally emit this code
  // calling HloProfilePrinter::profile_counters_size.
  const string assign_profile_counters_size =
      opts.gen_hlo_profile_printer_data
          ? "set_static_data_profile_counters_size(data, "
            "get_static_data_hlo_profile_printer_data(data)->"
            "profile_counters_size());"
          : "";

  const std::string function_declarations_from_obj_files = absl::StrReplaceAll(
      R"(// (Implementation detail) Entry point to the function in the object file.
extern "C" void {{ENTRY}}(
   void* result, const ::xla::ExecutableRunOptions* run_options,
   const void** args, void** temps, XlaCustomCallStatus* status,
   int64_t* profile_counters);
)",
      {{"{{ENTRY}}", raw_function_entry_identifier}});

  const std::string legacy_specific_static_data_setters = absl::StrReplaceAll(
      R"(
set_static_data_raw_function(data, {{RAW_FUNCTION_ENTRY_IDENTIFIER}});
set_static_data_hlo_profile_printer_data(data, StaticHloProfilePrinterData());
{{ASSIGN_PROFILE_COUNTERS_SIZE}}
)",
      {
          {"{{RAW_FUNCTION_ENTRY_IDENTIFIER}}", raw_function_entry_identifier},
          {"{{ASSIGN_PROFILE_COUNTERS_SIZE}}", assign_profile_counters_size},
      });

  const std::string legacy_specific_static_data_generators =
      absl::StrReplaceAll(
          R"(
// Metadata that can be used to pretty-print profile counters.
 static const ::xla::HloProfilePrinterData* StaticHloProfilePrinterData() {
   static const ::xla::HloProfilePrinterData* kHloProfilePrinterData =
     {{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}};
   return kHloProfilePrinterData;
}
)",
          {
              {"{{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}}",
               metadata_result.hlo_profile_printer_data_access_shim},
          });

  std::vector<std::pair<std::string, std::string>> rewrites_legacy = {
      {"{{SYMBOL_MAP_INITIALIZER}}", "{}"},
      {"{{FUNCTION_DECLARATIONS_FROM_OBJ_FILES}}",
       function_declarations_from_obj_files},
      {"{{TEMP_ALLOCATION_INDEX}}", "std::nullopt"},
      {"{{RUNTIME_SPECIFIC_INCLUDES}}", include_hlo_profile_printer_data_proto},
      {"{{EXECUTABLE_PROTO_GETTER}}", ""},
      {"{{THUNK_SPECIFIC_STATIC_DATA_SETTERS}}",
       "set_static_data_thunk_run_impl(data, nullptr);"},
      {"{{THUNK_RUN_IMPL_GETTER}}", ""},
      {"{{EMBEDDED_CONSTANT_BUFFERS_INITIALIZER_GETTER}}", ""},
      {"{{RNG_DELTAS_INITIALIZER_GETTER}}", ""},
      {"{{NANORT_SPECIFIC_STATIC_DATA_SETTERS}}",
       "set_static_data_compilation_result_proto(data, nullptr);"},
      {"{{COMPUTATION_CLASS_BASE}}", "XlaCompiledCpuFunction"},
      {"{{IS_THUNK_MODE}}", "false"},
      {"{{LEGACY_SPECIFIC_STATIC_DATA_SETTERS}}",
       legacy_specific_static_data_setters},
      {"{{LEGACY_SPECIFIC_STATIC_DATA_GENERATORS}}",
       legacy_specific_static_data_generators},
  };

  rewrites.insert(rewrites.end(), rewrites_legacy.begin(),
                  rewrites_legacy.end());

  return absl::OkStatus();
}

absl::Status GenerateHeader(
    const CodegenOpts& opts, const tf2xla::Config& config,
    const CompileResult& compile_result, const MetadataResult& metadata_result,
    const EmbeddedConstantBuffers& embedded_constant_buffers, string* header) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  TF_RETURN_IF_ERROR(ValidateFeedFetchCppNames(config));

  const bool is_aot_thunks = compile_result.is_aot_thunks();

  const std::vector<BufferInfo>& buffer_infos =
      compile_result.get_aot()->buffer_infos();

  const std::vector<int32> arg_index_table =
      ::xla::cpu::CreateArgIndexTableFromBufferInfos(buffer_infos);
  const std::vector<int32> result_index_table =
      ::xla::cpu::CreateResultIndexTableFromBufferInfos(buffer_infos);
  std::vector<string> buffer_infos_as_strings =
      BufferInfosToCppExpression(buffer_infos);

  // Compute sizes and generate methods.
  std::vector<BufferInfo> buffer_infos_for_args =
      ExtractEntryParamBufferInfos(buffer_infos);
  std::vector<BufferInfo> buffer_infos_for_temps =
      ExtractTempBufferInfos(buffer_infos);
  const xla::ProgramShapeProto& ps = compile_result.program_shape;
  string methods_arg, methods_result, methods_variable;
  TF_RETURN_IF_ERROR(GenArgMethods(config, ps, compile_result, &methods_arg));
  TF_RETURN_IF_ERROR(GenResultMethods(config, ps, &methods_result));
  TF_RETURN_IF_ERROR(GenVariableMethods(config, ps, &methods_variable));
  string arg_shape_infos, result_shape_infos;
  TF_RETURN_IF_ERROR(GenArgShapeInfos(ps, &arg_shape_infos));
  TF_RETURN_IF_ERROR(
      CheckEqual(ps.parameters_size(), arg_index_table.size(),
                 "Arg number mismatch, proto vs. arg_index_table"));
  TF_RETURN_IF_ERROR(GenResultShapeInfos(ps, &result_shape_infos));
  TF_RETURN_IF_ERROR(
      CheckEqual(ps.result().tuple_shapes_size(), result_index_table.size(),
                 "Result number mismatch, proto vs. result_index_table"));
  TF_ASSIGN_OR_RETURN(auto program_shape, xla::ProgramShape::FromProto(ps));
  const size_t arg_bytes_aligned =
      xla::cpu_function_runtime::AlignedBufferBytes(
          buffer_infos_for_args.data(), buffer_infos_for_args.size(),
          /*allocate_entry_params=*/true);
  const size_t arg_bytes_total = TotalBufferBytes(buffer_infos_for_args);
  const size_t temp_bytes_aligned =
      xla::cpu_function_runtime::AlignedBufferBytes(
          buffer_infos_for_temps.data(), buffer_infos_for_temps.size(),
          /*allocate_entry_params=*/true);
  const size_t temp_bytes_total = TotalBufferBytes(buffer_infos_for_temps);

  // Create rewrite strings for namespace start and end.
  string ns_start;
  for (const string& n : opts.namespaces) {
    ns_start += absl::StrCat("namespace ", n, " {\n");
  }
  ns_start += "\n";
  string ns_end("\n");
  for (int i = opts.namespaces.size() - 1; i >= 0; --i) {
    const string& n = opts.namespaces[i];
    ns_end += absl::StrCat("}  // end namespace ", n, "\n");
  }

  // Generate metadata.
  const string arg_names_code =
      GenNameToIndexCode(config.feed(), opts.gen_name_to_index);

  auto variable_copy = config.variable();
  for (auto& var : variable_copy) {
    if (var.name().empty()) {
      var.set_name(var.node_name());
    }
  }
  const string variable_names_code =
      GenNameToIndexCode(variable_copy, opts.gen_name_to_index);

  const string result_names_code =
      GenNameToIndexCode(config.fetch(), opts.gen_name_to_index);
  const string include_xla_data_proto =
      opts.gen_program_shape
          ? R"(#include "xla/xla_data.pb.h")"
          : "";

  // Use a poor-man's text templating mechanism; first populate the full
  // header with placeholder tokens, and then rewrite the tokens with real
  // values.
  *header =
      R"(// Generated by tfcompile, the TensorFlow graph compiler.  DO NOT EDIT!
//
// This header was generated via ahead-of-time compilation of a TensorFlow
// graph.  An object file corresponding to this header was also generated.
// This header gives access to the functionality in that object file.
//
// clang-format off

#ifndef TFCOMPILE_GENERATED_{{ENTRY}}_H_  // NOLINT(build/header_guard)
#define TFCOMPILE_GENERATED_{{ENTRY}}_H_  // NOLINT(build/header_guard)

{{INCLUDE_XLA_DATA_PROTO}}
{{RUNTIME_SPECIFIC_INCLUDES}}
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }

{{FUNCTION_DECLARATIONS_FROM_OBJ_FILES}}

{{DECLS_FROM_OBJ_FILE}}

{{NS_START}}
// {{CLASS}} represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code. This extends the generic
// XlaCompiledCpuFunction class with statically type-safe arg and result
// methods. Usage example:
//
//   {{CLASS}} computation;
//   // ...set args using computation.argN methods
//   CHECK(computation.Run());
//   // ...inspect results using computation.resultN methods
//
// The Run method invokes the actual computation, with inputs read from arg
// buffers, and outputs written to result buffers. Each Run call may also use
// a set of temporary buffers for the computation.
//
// By default each instance of this class manages its own arg, result and temp
// buffers. The AllocMode constructor parameter may be used to modify the
// buffer allocation strategy.
//
// Under the default allocation strategy, this class is thread-compatible:
// o Calls to non-const methods require exclusive access to the object.
// o Concurrent calls to const methods are OK, if those calls are made while it
//   is guaranteed that no thread may call a non-const method.
//
// The logical function signature is:
//   {{PROGRAM_SHAPE}}
//
// Memory stats:
//   arg bytes total:    {{ARG_BYTES_TOTAL}}
//   arg bytes aligned:  {{ARG_BYTES_ALIGNED}}
//   temp bytes total:   {{TEMP_BYTES_TOTAL}}
//   temp bytes aligned: {{TEMP_BYTES_ALIGNED}}
class {{CLASS}} final : public tensorflow::{{COMPUTATION_CLASS_BASE}} {
 public:
  // Number of input arguments for the compiled computation.
  static constexpr size_t kNumArgs = {{ARG_NUM}};

  static constexpr size_t kNumResults = {{RESULT_NUM}};

  // Number of variables for the compiled computation.
  static constexpr size_t kNumVariables = {{VARIABLE_NUM}};

  // Byte size of each argument buffer. There are kNumArgs entries.
  static const ::int64_t ArgSize(::tensorflow::int32 index) {
    return BufferInfos()[ArgIndexToBufferIndex()[index]].size();
  }

  // Returns static data used to create an XlaCompiledCpuFunction.
  static const tensorflow::XlaCompiledCpuFunction::StaticData& StaticData() {
    static XlaCompiledCpuFunction::StaticData* kStaticData = [](){
      XlaCompiledCpuFunction::StaticData* data =
        new XlaCompiledCpuFunction::StaticData;
      set_static_data_function_library_symbol_map(data, FunctionLibrarySymbolMap());
      set_static_data_buffer_infos(data, BufferInfos());
      set_static_data_num_buffers(data, kNumBuffers);
      set_static_data_result_index_table(data, ResultIndexToBufferIndex());
      set_static_data_num_results(data, kNumResults);
      set_static_data_arg_index_table(data, ArgIndexToBufferIndex());
      set_static_data_num_args(data, kNumArgs);
      set_static_data_num_variables(data, kNumVariables);
      set_static_data_temp_allocation_index(data, kTempAllocationIndex);
      set_static_data_arg_shape_infos(data, ArgShapeInfos());
      set_static_data_result_shape_infos(data, ResultShapeInfos());
      set_static_data_arg_names(data, StaticArgNames());
      set_static_data_variable_names(data, StaticVariableNames());
      set_static_data_result_names(data, StaticResultNames());
      set_static_data_program_shape(data, StaticProgramShape());
      {{THUNK_SPECIFIC_STATIC_DATA_SETTERS}}
      {{NANORT_SPECIFIC_STATIC_DATA_SETTERS}}
      {{LEGACY_SPECIFIC_STATIC_DATA_SETTERS}}
      return data;
    }();
    return *kStaticData;
  }

  {{CLASS}}(AllocMode alloc_mode =
            AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS)
      : {{COMPUTATION_CLASS_BASE}}(StaticData(), alloc_mode) {}

  {{CLASS}}(const {{CLASS}}&) = delete;
  {{CLASS}}& operator=(const {{CLASS}}&) = delete;

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument, with the following
  // general form:
  //
  // void set_argN_data(void* data)
  //   Sets the buffer of type T for positional argument N. May be called in
  //   any AllocMode. Must be called before Run to have an effect. Must be
  //   called in AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY for each positional
  //   argument, to set the argument buffers.
  //
  // T* argN_data()
  //   Returns the buffer of type T for positional argument N.
  //
  // T& argN(...dim indices...)
  //   Returns a reference to the value of type T for positional argument N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
{{METHODS_ARG}}

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result, with the following general form:
  //
  // T* resultN_data()
  //   Returns the buffer of type T for positional result N.
  //
  // T& resultN(...dim indices...)
  //   Returns a reference to the value of type T for positional result N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
  //
  // Unlike the arg methods, there is no set_resultN_data method. The result
  // buffers are managed internally, and may change after each call to Run.
{{METHODS_RESULT}}

  // Methods for managing variable buffers. Buffers are in row-major order.
  //
  // For read-write variables we generate the following methods:
  //
  // void set_var_X_data(T* data)
  //   Sets the buffer for variable X.  Must be called before Run if the
  //   allocation mode is RESULTS_PROFILES_AND_TEMPS_ONLY.
  //
  // T* var_X_data()
  //   Returns the buffer of type T for variable X.  If the allocation mode is
  //   RESULTS_PROFILES_AND_TEMPS_ONLY then this buffer is the same as the
  //   buffer passed to set_var_X_data.
  //
  // T& var_X(...dim indices...)
  //   Returns a reference to the value of type T for variable X,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
  //
  // For readonly variables we generate the same set of methods, except that we
  // use `const T` instead of `T`.  We use `const T` to avoid erasing the
  // constness of the buffer passed to `set_var_X_data` but the underlying
  // buffer is not const (and thus the const can be safely const-cast'ed away)
  // unless `set_var_X_data` is called with a pointer to constant storage.
{{METHODS_VARIABLE}}

 private:
  // Number of buffers for the compiled computation.
  static constexpr size_t kNumBuffers = {{NUM_BUFFERS}};

  static const ::xla::cpu_function_runtime::BufferInfo* BufferInfos() {
    static const ::xla::cpu_function_runtime::BufferInfo
      kBufferInfos[kNumBuffers] = {
{{BUFFER_INFOS_AS_STRING}}
      };
    return kBufferInfos;
  }

  static const ::tensorflow::int32* ResultIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kResultIndexToBufferIndex[kNumResults] = {
{{RESULT_INDEX_TABLE}}
    };
    return kResultIndexToBufferIndex;
  }

  static const ::tensorflow::int32* ArgIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kArgIndexToBufferIndex[kNumArgs] = {
{{ARG_INDEX_TABLE}}
    };
    return kArgIndexToBufferIndex;
  }

  // Temp allocation index..
  static constexpr std::optional<size_t> kTempAllocationIndex = {{TEMP_ALLOCATION_INDEX}};

  // Shapes of the input arguments.
{{ARG_SHAPE_INFOS}};

  // Shapes of the results.
{{RESULT_SHAPE_INFOS}};

  // Array of names of each positional argument, terminated by nullptr.
  static const char** StaticArgNames() {{ARG_NAMES_CODE}}

  // Array of names of each positional variable, terminated by nullptr.
  static const char** StaticVariableNames() {{VARIABLE_NAMES_CODE}}

  // Array of names of each positional result, terminated by nullptr.
  static const char** StaticResultNames() {{RESULT_NAMES_CODE}}

  // Shape of the args and results.
  static const ::xla::ProgramShapeProto* StaticProgramShape() {
    static const ::xla::ProgramShapeProto* kShape = {{PROGRAM_SHAPE_SHIM_EXPRESSION}};
    return kShape;
  }

  {{EXECUTABLE_PROTO_GETTER}}
  {{THUNK_RUN_IMPL_GETTER}}
  {{EMBEDDED_CONSTANT_BUFFERS_INITIALIZER_GETTER}}
  {{RNG_DELTAS_INITIALIZER_GETTER}}
  {{LEGACY_SPECIFIC_STATIC_DATA_GENERATORS}}

  static absl::flat_hash_map<std::string, void*> FunctionLibrarySymbolMap() {
    return {{SYMBOL_MAP_INITIALIZER}};
  }
 protected:
  bool is_thunk_mode() const override { return {{IS_THUNK_MODE}}; }
};
{{NS_END}}

#endif  // TFCOMPILE_GENERATED_{{ENTRY}}_H_

// clang-format on
)";

  std::vector<std::string> decls_from_obj_file =
      metadata_result.header_variable_decls;
  for (const auto& constant_buffer_var_info :
       embedded_constant_buffers.variable_decls) {
    decls_from_obj_file.push_back(constant_buffer_var_info.variable_decl);
  }

  // The replacement strategy is naive, but good enough for our purposes.
  std::vector<std::pair<string, string>> rewrites = {
      {"{{ARG_BYTES_ALIGNED}}", absl::StrCat(arg_bytes_aligned)},
      {"{{ARG_BYTES_TOTAL}}", absl::StrCat(arg_bytes_total)},
      {"{{ARG_NAMES_CODE}}", arg_names_code},
      {"{{ARG_NUM}}", absl::StrCat(arg_index_table.size())},
      {"{{ARG_SHAPE_INFOS}}", arg_shape_infos},
      {"{{VARIABLE_NUM}}", absl::StrCat(config.variable_size())},
      {"{{ARG_INDEX_TABLE}}", absl::StrJoin(arg_index_table, ", ")},
      {"{{RESULT_NUM}}", absl::StrCat(result_index_table.size())},
      {"{{RESULT_INDEX_TABLE}}", absl::StrJoin(result_index_table, ", ")},
      {"{{CLASS}}", opts.class_name},
      {"{{DECLS_FROM_OBJ_FILE}}", absl::StrJoin(decls_from_obj_file, "\n")},
      {"{{ENTRY}}", compile_result.entry_point},
      {"{{INCLUDE_XLA_DATA_PROTO}}", include_xla_data_proto},
      {"{{METHODS_ARG}}\n", methods_arg},
      {"{{METHODS_RESULT}}\n", methods_result},
      {"{{METHODS_VARIABLE}}\n", methods_variable},
      {"{{NS_END}}\n", ns_end},
      {"{{NS_START}}\n", ns_start},
      {"{{PROGRAM_SHAPE}}", xla::ShapeUtil::HumanString(program_shape)},
      {"{{PROGRAM_SHAPE_SHIM_EXPRESSION}}",
       metadata_result.program_shape_access_shim},
      {"{{VARIABLE_NAMES_CODE}}", variable_names_code},
      {"{{RESULT_NAMES_CODE}}", result_names_code},
      {"{{RESULT_SHAPE_INFOS}}", result_shape_infos},
      {"{{TEMP_BYTES_ALIGNED}}", absl::StrCat(temp_bytes_aligned)},
      {"{{TEMP_BYTES_TOTAL}}", absl::StrCat(temp_bytes_total)},
      {"{{NUM_BUFFERS}}", absl::StrCat(buffer_infos.size())},
      {"{{BUFFER_INFOS_AS_STRING}}",
       absl::StrJoin(buffer_infos_as_strings, ",\n")},
  };

  if (is_aot_thunks) {
    TF_RETURN_IF_ERROR(
        ExtendRewrites(rewrites, compile_result.get_aot_thunks().value(),
                       metadata_result, opts, embedded_constant_buffers));
  } else {
    TF_RETURN_IF_ERROR(
        ExtendRewrites(rewrites, compile_result.get_aot_legacy().value(),
                       metadata_result, opts, compile_result.entry_point));
  }
  absl::StrReplaceAll(rewrites, header);
  return absl::OkStatus();
}

static string CreateUniqueIdentifier(const CodegenOpts& opts,
                                     absl::string_view suffix) {
  string result = "__tfcompile";
  for (const string& n : opts.namespaces) {
    absl::StrAppend(&result, "_", n);
  }

  absl::StrAppend(&result, "_", opts.class_name, "_", suffix);
  return result;
}

absl::StatusOr<EmbeddedConstantBuffers> GenerateConstantBuffersData(
    const CodegenOpts& opts, const CompileResult& compile_result) {
  TF_ASSIGN_OR_RETURN(auto aot_thunk_result, compile_result.get_aot_thunks());

  // Create a temporary object for aot_thunk_result to be able to call
  // LoadExecutable without moving the original object.
  TF_ASSIGN_OR_RETURN(auto serialized, aot_thunk_result->SerializeAsString());
  TF_ASSIGN_OR_RETURN(
      auto aot_thunk_result_temp,
      xla::cpu::CpuAotCompilationResultThunks::FromString(serialized, nullptr));

  TF_ASSIGN_OR_RETURN(
      auto executable,
      std::move(*aot_thunk_result_temp).LoadExecutable(nullptr, nullptr));

  xla::cpu::CpuExecutable* cpu_executable =
      tsl::down_cast<xla::cpu::CpuExecutable*>(executable.get());

  std::vector<ConstantToEmbed> constants_to_embed;

  int constant_identifier = 0;
  for (const auto& constant : cpu_executable->constants()) {
    const uint8_t* constant_data_bytes_ptr = reinterpret_cast<const uint8_t*>(
        constant.AsDeviceMemoryBase().opaque());
    const size_t constant_size = constant.AsDeviceMemoryBase().size();

    // NOTE(basioli): Some constants are empty, we don't need to embed them
    if (constant_size == 0) {
      continue;
    }

    ConstantToEmbed constant_to_embed;
    constant_to_embed.SerializeIntoBuffer(
        absl::MakeSpan(constant_data_bytes_ptr, constant_size));

    constant_to_embed.symbol_prefix = CreateUniqueIdentifier(
        opts, absl::StrCat("Constant_", std::to_string(constant_identifier++)));

    constants_to_embed.push_back(std::move(constant_to_embed));
  }

  return CreateEmbeddedConstantBuffers(opts.target_triple,
                                       absl::MakeSpan(constants_to_embed));
}

absl::Status GenerateMetadata(const CodegenOpts& opts,
                              const CompileResult& compile_result,
                              MetadataResult* metadata_result) {
  std::unique_ptr<xla::ProgramShapeProto> program_shape;

  if (opts.gen_program_shape) {
    program_shape =
        std::make_unique<xla::ProgramShapeProto>(compile_result.program_shape);

    // The parameter names are currently meaningless, and redundant with the
    // rest of our metadata, so clear them out to avoid confusion and save
    // space.
    program_shape->clear_parameter_names();
  }

  const bool is_aot_thunks = compile_result.is_aot_thunks();

  // When asked to serialize a null protobuf, CreateEmbeddedProtocolBuffer
  // gives a shim that evaluates to nullptr, which is what we want.
  std::vector<ProtobufToEmbed> protobufs_to_embed;
  protobufs_to_embed.push_back(
      ProtobufToEmbed{CreateUniqueIdentifier(opts, "ProgramShapeProto"),
                      "::xla::ProgramShapeProto", program_shape.get()});

  if (is_aot_thunks) {
    protobufs_to_embed.push_back(
        ProtobufToEmbed{CreateUniqueIdentifier(opts, "HloProfilePrinterData"),
                        "::xla::HloProfilePrinterData", nullptr});
    protobufs_to_embed.push_back(
        ProtobufToEmbed{CreateUniqueIdentifier(opts, "CompilationResultProto"),
                        "::xla::cpu::CompilationResultProto",
                        &compile_result.get_aot_thunks().value()->proto()});
  } else {
    protobufs_to_embed.push_back(ProtobufToEmbed{
        CreateUniqueIdentifier(opts, "HloProfilePrinterData"),
        "::xla::HloProfilePrinterData",
        compile_result.get_aot_legacy().value()->hlo_profile_printer_data()});

    protobufs_to_embed.push_back(
        ProtobufToEmbed{CreateUniqueIdentifier(opts, "CompilationResultProto"),
                        "::xla::cpu::CompilationResultProto", nullptr});
  }

  TF_ASSIGN_OR_RETURN(
      EmbeddedProtocolBuffers embedded_protobufs,
      CreateEmbeddedProtocolBuffers(opts.target_triple, protobufs_to_embed));

  metadata_result->program_shape_access_shim =
      std::move(embedded_protobufs.cpp_shims[0].expression);
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[0].variable_decl));

  metadata_result->hlo_profile_printer_data_access_shim =
      std::move(embedded_protobufs.cpp_shims[1].expression);
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[1].variable_decl));

  metadata_result->cpu_executable_access_shim =
      std::move(embedded_protobufs.cpp_shims[2].expression);
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[2].variable_decl));

  metadata_result->object_file_data =
      std::move(embedded_protobufs.object_file_data);
  return absl::OkStatus();
}

absl::Status ParseCppClass(const string& cpp_class, string* class_name,
                           std::vector<string>* namespaces) {
  class_name->clear();
  namespaces->clear();
  if (cpp_class.empty()) {
    return errors::InvalidArgument("empty cpp_class: " + cpp_class);
  }
  std::vector<string> parts = absl::StrSplit(cpp_class, "::");
  if (parts.front().empty()) {
    // Allow a fully qualified name that starts with "::".
    parts.erase(parts.begin());
  }
  for (int i = 0, end = parts.size(); i < end; ++i) {
    if (i < end - 1) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(
          parts[i], "in namespace component of cpp_class: " + cpp_class));
      namespaces->push_back(parts[i]);
    } else {
      TF_RETURN_IF_ERROR(ValidateCppIdent(
          parts[i], "in class name of cpp_class: " + cpp_class));
      *class_name = parts[i];
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateCppIdent(absl::string_view ident, absl::string_view msg) {
  if (ident.empty()) {
    return errors::InvalidArgument("empty identifier: ", msg);
  }
  // Require that the identifier starts with a nondigit, and is composed of
  // nondigits and digits, as specified in section [2.11 Identifiers] of the
  // C++11 Standard.  Note that nondigit is defined as [_a-zA-Z] and digit is
  // defined as [0-9].
  //
  // Technically the standard also allows for `universal-character-name`, with
  // a table of allowed unicode ranges, as well as `other
  // implementation-defined characters`.  We disallow those here to give
  // better error messages, at the expensive of being more restrictive than
  // the standard.
  if (ident[0] != '_' && !IsAlpha(ident[0])) {
    return errors::InvalidArgument("illegal leading char: ", msg);
  }
  for (size_t pos = 1; pos < ident.size(); ++pos) {
    if (ident[pos] != '_' && !IsAlphaNum(ident[pos])) {
      return errors::InvalidArgument("illegal char: ", msg);
    }
  }
  return absl::OkStatus();
}

}  // namespace tfcompile
}  // namespace tensorflow
