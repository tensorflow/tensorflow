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

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/aot/embedded_protocol_buffers.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace tfcompile {

namespace {

using BufferInfo = xla::cpu_function_runtime::BufferInfo;

bool IsAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAlphaNum(char c) { return IsAlpha(c) || (c >= '0' && c <= '9'); }

// Convert an XLA type into a C++ type.
Status XLATypeToCpp(xla::PrimitiveType type, string* str) {
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
  return OkStatus();
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
Status AddRewritesForShape(int i, const xla::Shape& shape,
                           std::vector<std::pair<string, string>>* rewrites) {
  string type;
  TF_RETURN_IF_ERROR(XLATypeToCpp(shape.element_type(), &type));
  std::vector<string> dim_vars;
  string dim_sizes, indices;
  int count = 1;
  if (shape.rank() == 0 ||
      (shape.dimensions_size() == 1 && shape.dimensions(0) == 1)) {
    dim_sizes = "[1]";
    indices = "[0]";
  } else {
    for (int dim = 0; dim < shape.dimensions_size(); ++dim) {
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
  return OkStatus();
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
Status GenArgMethods(const tf2xla::Config& config,
                     const xla::ProgramShapeProto& ps,
                     const CompileResult& compile_result, string* methods) {
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
    TF_RETURN_IF_ERROR(
        AddRewritesForShape(i, xla::Shape(ps.parameters(i)), &rewrites));
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
  return OkStatus();
}

// Generate methods for results (outputs).
Status GenResultMethods(const tf2xla::Config& config,
                        const xla::ProgramShapeProto& ps, string* methods) {
  if (ps.result().element_type() != xla::TUPLE) {
    // The XlaCompiler we use to build the xla computation always generates a
    // tuple result, and we rely on this to simplify code generation.
    return errors::Internal("codegen requires the XLA result to be a tuple");
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
    TF_RETURN_IF_ERROR(AddRewritesForShape(
        i, xla::Shape(ps.result().tuple_shapes(i)), &rewrites));
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
  return OkStatus();
}

// Generate methods for variables.
Status GenVariableMethods(const tf2xla::Config& config,
                          const xla::ProgramShapeProto& ps, string* methods) {
  const int num_args = ps.parameters_size();
  for (int i = config.feed_size(); i < num_args; ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(
        AddRewritesForShape(i, xla::Shape(ps.parameters(i)), &rewrites));
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
  return OkStatus();
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

Status ValidateFeedFetchCppNames(const tf2xla::Config& config) {
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
  return OkStatus();
}

// Returns a list of C++ expressions that, when executed, will construct the
// BufferInfo instances in `buffer_infos`.
std::vector<string> BufferInfosToCppExpression(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<string> buffer_infos_as_strings;
  std::transform(buffer_infos.begin(), buffer_infos.end(),
                 std::back_inserter(buffer_infos_as_strings),
                 [](const BufferInfo& buffer_info) {
                   std::pair<uint64, uint64> encoded = buffer_info.Encode();
                   string encoded_second_as_str =
                       encoded.second == ~0ULL
                           ? "~0ULL"
                           : absl::StrCat(encoded.second, "ULL");
                   return absl::StrCat(
                       "::xla::cpu_function_runtime::BufferInfo({",
                       encoded.first, "ULL, ", encoded_second_as_str, "})");
                 });
  return buffer_infos_as_strings;
}
}  // namespace

Status GenerateHeader(const CodegenOpts& opts, const tf2xla::Config& config,
                      const CompileResult& compile_result,
                      const MetadataResult& metadata_result, string* header) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  TF_RETURN_IF_ERROR(ValidateFeedFetchCppNames(config));
  const int64_t result_index = compile_result.aot->result_buffer_index();
  const std::vector<BufferInfo>& buffer_infos =
      compile_result.aot->buffer_infos();
  const std::vector<int32> arg_index_table =
      ::xla::cpu::CreateArgIndexTableFromBufferInfos(buffer_infos);
  std::vector<string> buffer_infos_as_strings =
      BufferInfosToCppExpression(buffer_infos);
  const int64_t buffer_infos_size = buffer_infos.size();
  if (result_index < 0 || result_index >= buffer_infos_size) {
    return errors::InvalidArgument("result index: ", result_index,
                                   " is outside the range of temp sizes: [0,",
                                   buffer_infos.size(), ")");
  }

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
          ? R"(#include "tensorflow/compiler/xla/xla_data.pb.h")"
          : "";

  const string include_hlo_profile_printer_data_proto =
      opts.gen_hlo_profile_printer_data
          ? R"(#include "tensorflow/compiler/xla/service/hlo_profile_printer_data.pb.h")"
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

  // Use a poor-man's text templating mechanism; first populate the full header
  // with placeholder tokens, and then rewrite the tokens with real values.
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
{{INCLUDE_HLO_PROFILE_PRINTER_DATA_PROTO}}
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }

// (Implementation detail) Entry point to the function in the object file.
extern "C" void {{ENTRY}}(
    void* result, const ::xla::ExecutableRunOptions* run_options,
    const void** args, void** temps, XlaCustomCallStatus* status,
    int64_t* profile_counters);

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
class {{CLASS}} final : public tensorflow::XlaCompiledCpuFunction {
 public:
  // Number of input arguments for the compiled computation.
  static constexpr size_t kNumArgs = {{ARG_NUM}};

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
      set_static_data_raw_function(data, {{ENTRY}});
      set_static_data_buffer_infos(data, BufferInfos());
      set_static_data_num_buffers(data, kNumBuffers);
      set_static_data_arg_index_table(data, ArgIndexToBufferIndex());
      set_static_data_num_args(data, kNumArgs);
      set_static_data_num_variables(data, kNumVariables);
      set_static_data_result_index(data, kResultIndex);
      set_static_data_arg_names(data, StaticArgNames());
      set_static_data_variable_names(data, StaticVariableNames());
      set_static_data_result_names(data, StaticResultNames());
      set_static_data_program_shape(data, StaticProgramShape());
      set_static_data_hlo_profile_printer_data(
          data, StaticHloProfilePrinterData());
{{ASSIGN_PROFILE_COUNTERS_SIZE}}
      return data;
    }();
    return *kStaticData;
  }

  {{CLASS}}(AllocMode alloc_mode =
            AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS)
      : XlaCompiledCpuFunction(StaticData(), alloc_mode) {}

  {{CLASS}}(const {{CLASS}}&) = delete;
  {{CLASS}}& operator=(const {{CLASS}}&) = delete;

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument, with the following
  // general form:
  //
  // void set_argN_data(void* data)
  //   Sets the buffer of type T for positional argument N. May be called in
  //   any AllocMode. Must be called before Run to have an affect. Must be
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

  static const ::tensorflow::int32* ArgIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kArgIndexToBufferIndex[kNumArgs] = {
{{ARG_INDEX_TABLE}}
    };
    return kArgIndexToBufferIndex;
  }

  // The 0-based index of the result tuple in the temporary buffers.
  static constexpr size_t kResultIndex = {{RESULT_INDEX}};

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

  // Metadata that can be used to pretty-print profile counters.
  static const ::xla::HloProfilePrinterData* StaticHloProfilePrinterData() {
    static const ::xla::HloProfilePrinterData* kHloProfilePrinterData =
      {{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}};
    return kHloProfilePrinterData;
  }
};
{{NS_END}}

#endif  // TFCOMPILE_GENERATED_{{ENTRY}}_H_

// clang-format on
)";
  // The replacement strategy is naive, but good enough for our purposes.
  const std::vector<std::pair<string, string>> rewrites = {
      {"{{ARG_BYTES_ALIGNED}}", absl::StrCat(arg_bytes_aligned)},
      {"{{ARG_BYTES_TOTAL}}", absl::StrCat(arg_bytes_total)},
      {"{{ARG_NAMES_CODE}}", arg_names_code},
      {"{{ARG_NUM}}", absl::StrCat(arg_index_table.size())},
      {"{{VARIABLE_NUM}}", absl::StrCat(config.variable_size())},
      {"{{ARG_INDEX_TABLE}}", absl::StrJoin(arg_index_table, ", ")},
      {"{{ASSIGN_PROFILE_COUNTERS_SIZE}}", assign_profile_counters_size},
      {"{{CLASS}}", opts.class_name},
      {"{{DECLS_FROM_OBJ_FILE}}",
       absl::StrJoin(metadata_result.header_variable_decls, "\n")},
      {"{{ENTRY}}", compile_result.entry_point},
      {"{{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}}",
       metadata_result.hlo_profile_printer_data_access_shim},
      {"{{INCLUDE_XLA_DATA_PROTO}}", include_xla_data_proto},
      {"{{INCLUDE_HLO_PROFILE_PRINTER_DATA_PROTO}}",
       include_hlo_profile_printer_data_proto},
      {"{{METHODS_ARG}}\n", methods_arg},
      {"{{METHODS_RESULT}}\n", methods_result},
      {"{{METHODS_VARIABLE}}\n", methods_variable},
      {"{{NS_END}}\n", ns_end},
      {"{{NS_START}}\n", ns_start},
      {"{{PROGRAM_SHAPE}}", xla::ShapeUtil::HumanString(xla::ProgramShape(ps))},
      {"{{PROGRAM_SHAPE_SHIM_EXPRESSION}}",
       metadata_result.program_shape_access_shim},
      {"{{VARIABLE_NAMES_CODE}}", variable_names_code},
      {"{{RESULT_INDEX}}", absl::StrCat(result_index)},
      {"{{RESULT_NAMES_CODE}}", result_names_code},
      {"{{TEMP_BYTES_ALIGNED}}", absl::StrCat(temp_bytes_aligned)},
      {"{{TEMP_BYTES_TOTAL}}", absl::StrCat(temp_bytes_total)},
      {"{{NUM_BUFFERS}}", absl::StrCat(buffer_infos.size())},
      {"{{BUFFER_INFOS_AS_STRING}}",
       absl::StrJoin(buffer_infos_as_strings, ",\n")}};
  absl::StrReplaceAll(rewrites, header);
  return OkStatus();
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

Status GenerateMetadata(const CodegenOpts& opts,
                        const CompileResult& compile_result,
                        MetadataResult* metadata_result) {
  std::unique_ptr<xla::ProgramShapeProto> program_shape;

  if (opts.gen_program_shape) {
    program_shape =
        absl::make_unique<xla::ProgramShapeProto>(compile_result.program_shape);

    // The parameter names are currently meaningless, and redundant with the
    // rest of our metadata, so clear them out to avoid confusion and save
    // space.
    program_shape->clear_parameter_names();
  }

  // When asked to serialize a null protobuf, CreateEmbeddedProtocolBuffer gives
  // a shim that evaluates to nullptr, which is what we want.

  ProtobufToEmbed program_shape_protobuf{
      CreateUniqueIdentifier(opts, "ProgramShapeProto"),
      "::xla::ProgramShapeProto", program_shape.get()};

  ProtobufToEmbed hlo_profile_printer_data_protobuf{
      CreateUniqueIdentifier(opts, "HloProfilePrinterData"),
      "::xla::HloProfilePrinterData",
      compile_result.aot->hlo_profile_printer_data()};

  TF_ASSIGN_OR_RETURN(
      EmbeddedProtocolBuffers embedded_protobufs,
      CreateEmbeddedProtocolBuffers(
          opts.target_triple,
          {program_shape_protobuf, hlo_profile_printer_data_protobuf}));

  metadata_result->program_shape_access_shim =
      std::move(embedded_protobufs.cpp_shims[0].expression);
  metadata_result->hlo_profile_printer_data_access_shim =
      std::move(embedded_protobufs.cpp_shims[1].expression);
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[0].variable_decl));
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[1].variable_decl));
  metadata_result->object_file_data =
      std::move(embedded_protobufs.object_file_data);
  return OkStatus();
}

Status ParseCppClass(const string& cpp_class, string* class_name,
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
  return OkStatus();
}

Status ValidateCppIdent(absl::string_view ident, absl::string_view msg) {
  if (ident.empty()) {
    return errors::InvalidArgument("empty identifier: ", msg);
  }
  // Require that the identifier starts with a nondigit, and is composed of
  // nondigits and digits, as specified in section [2.11 Identifiers] of the
  // C++11 Standard.  Note that nondigit is defined as [_a-zA-Z] and digit is
  // defined as [0-9].
  //
  // Technically the standard also allows for `universal-character-name`, with a
  // table of allowed unicode ranges, as well as `other implementation-defined
  // characters`.  We disallow those here to give better error messages, at the
  // expensive of being more restrictive than the standard.
  if (ident[0] != '_' && !IsAlpha(ident[0])) {
    return errors::InvalidArgument("illegal leading char: ", msg);
  }
  for (size_t pos = 1; pos < ident.size(); ++pos) {
    if (ident[pos] != '_' && !IsAlphaNum(ident[pos])) {
      return errors::InvalidArgument("illegal char: ", msg);
    }
  }
  return OkStatus();
}

}  // namespace tfcompile
}  // namespace tensorflow
