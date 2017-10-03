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

#include "tensorflow/compiler/aot/runtime.h"
#include "tensorflow/compiler/tf2xla/str_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace tfcompile {

namespace {

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
      *str = "tensorflow::int64";
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
  return Status::OK();
}

// total_buffer_bytes returns the sum of each size in `sizes`, skipping -1
// values.  There are `n` entries in `sizes`.
size_t total_buffer_bytes(const intptr_t* sizes, size_t n) {
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    if (sizes[i] != -1) {
      total += sizes[i];
    }
  }
  return total;
}

// Fills in arg_sizes with the byte size of each positional arg.
Status ComputeArgSizes(const CompileResult& compile_result,
                       std::vector<int64>* arg_sizes) {
  const xla::ProgramShape& ps = compile_result.program_shape;
  for (int i = 0; i < ps.parameters_size(); ++i) {
    if (i == ps.parameters_size() - 1 && compile_result.has_context_arg) {
      // If the compiled function needs a XlaLocalRuntimeContext* arg, it's
      // always last, and must be represented as an opaque type.
      const xla::PrimitiveType type = ps.parameters(i).element_type();
      if (type != xla::OPAQUE) {
        return errors::InvalidArgument(
            "expected final context arg to be opaque, but got type: ",
            xla::PrimitiveType_Name(type), ", from program shape: ",
            xla::ShapeUtil::HumanString(ps));
      }
      arg_sizes->push_back(-1);
    } else {
      arg_sizes->push_back(xla::ShapeUtil::ByteSizeOf(
          ps.parameters(i), compile_result.pointer_size));
    }
  }
  return Status::OK();
}

// Add (from,to) rewrite pairs based on the given shape.  These rewrite pairs
// are used to generate methods for args and results.
Status AddRewritesForShape(int i, const xla::Shape& shape,
                           std::vector<std::pair<string, string>>* rewrites) {
  string type;
  TF_RETURN_IF_ERROR(XLATypeToCpp(shape.element_type(), &type));
  std::vector<string> dim_vars;
  string dim_sizes, indices;
  if (xla::ShapeUtil::Rank(shape) == 0 ||
      (shape.dimensions_size() == 1 && shape.dimensions(0) == 1)) {
    dim_sizes = "[1]";
    indices = "[0]";
  } else {
    for (int dim = 0; dim < shape.dimensions_size(); ++dim) {
      dim_vars.push_back(strings::StrCat("size_t dim", dim));
      dim_sizes += strings::StrCat("[", shape.dimensions(dim), "]");
      indices += strings::StrCat("[dim", dim, "]");
    }
  }
  rewrites->push_back({"{{I}}", strings::StrCat(i)});
  rewrites->push_back({"{{TYPE}}", type});
  rewrites->push_back({"{{DIM_VARS}}", str_util::Join(dim_vars, ", ")});
  rewrites->push_back({"{{DIM_SIZES}}", dim_sizes});
  rewrites->push_back({"{{INDICES}}", indices});
  return Status::OK();
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
  str_util::ReplaceAllPairs(&code, rewrites);
  return str_util::StringReplace(code, "{{NAME}}", name, /*replace_all=*/true);
}

// Generate methods for args (inputs).
Status GenArgMethods(const tf2xla::Config& config, const xla::ProgramShape& ps,
                     const CompileResult& compile_result, string* methods) {
  size_t num_args = ps.parameters_size();
  if (compile_result.has_context_arg) {
    // If the compiled function needs a XlaLocalRuntimeContext* arg, it's
    // always last, and is set in the class constructor.
    num_args--;
  }
  if (config.feed_size() != num_args) {
    return errors::InvalidArgument("mismatch between feed_size(",
                                   config.feed_size(), ") and num_args(",
                                   num_args, ")");
  }
  for (int i = 0; i < num_args; ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(AddRewritesForShape(i, ps.parameters(i), &rewrites));
    const string code = R"(
  void set_arg{{NAME}}_data(void* data) {
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
)";
    *methods += RewriteWithName(strings::StrCat(i), code, rewrites);
    if (!config.feed(i).name().empty()) {
      *methods += RewriteWithName("_" + config.feed(i).name(), code, rewrites);
    }
  }
  return Status::OK();
}

// Generate methods for results (outputs).
Status GenResultMethods(const tf2xla::Config& config,
                        const xla::ProgramShape& ps, string* methods) {
  if (ps.result().element_type() != xla::TUPLE) {
    // The XlaCompiler we use to build the xla computation always generates a
    // tuple result, and we rely on this to simplify code generation.
    return errors::Internal("codegen requires the XLA result to be a tuple");
  }
  if (config.fetch_size() != ps.result().tuple_shapes_size()) {
    return errors::InvalidArgument("mismatch between fetch_size(",
                                   config.feed_size(), ") and tuple_size(",
                                   ps.result().tuple_shapes_size(), ")");
  }
  for (int i = 0; i < ps.result().tuple_shapes_size(); ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(
        AddRewritesForShape(i, ps.result().tuple_shapes(i), &rewrites));
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
)";
    *methods += RewriteWithName(strings::StrCat(i), code, rewrites);
    if (!config.fetch(i).name().empty()) {
      *methods += RewriteWithName("_" + config.fetch(i).name(), code, rewrites);
    }
  }
  return Status::OK();
}

// Generates code implementing {Arg,Result}Names(), where T is one of
// tf2xla::{Feed,Fetch}. Each feed or fetch name results in a C-style string
// literal in the array, with nullptr terminating the array.
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

// Converts the given `str` into a comma-separated list of per-character values.
string StringToCharList(const string& str) {
  string list;
  for (const char c : str) {
    if (!list.empty()) {
      list += ",";
    }
    list += strings::StrCat(static_cast<int>(c));
  }
  return list;
}

string GenProgramShapeCode(xla::ProgramShape program_shape, bool generate) {
  // No need for any static magic if we're not supposed to generate the data.
  if (!generate) {
    return "{\n    return nullptr;\n  }";
  }
  // The parameter names are currently meaningless, and redundant with the rest
  // of our metadata, so clear them out to avoid confusion and save space.
  program_shape.clear_parameter_names();
  const string proto_str = program_shape.SerializeAsString();
  // Embed the program shape as a serialized protobuf in the header file.
  //
  // TODO(toddw): This strategy will likely fail for larger protobufs, depending
  // on the C++ compiler that is used. Figure out another solution if necessary.
  string code = R"({
    static const xla::ProgramShape* kShape = []() {
      static const char kProto[] = {{{PROTO_LIST}}};
      static constexpr int kProtoSize = {{PROTO_SIZE}};
      xla::ProgramShape* shape = new xla::ProgramShape;
      shape->ParseFromArray(kProto, kProtoSize);
      return shape;
    }();
    return kShape;
  })";
  str_util::ReplaceAllPairs(
      &code, {
                 {"{{PROTO_LIST}}", StringToCharList(proto_str)},
                 {"{{PROTO_SIZE}}", strings::StrCat(proto_str.size())},
             });
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
  return Status::OK();
}

}  // namespace

Status GenerateHeader(const HeaderOpts& opts, const tf2xla::Config& config,
                      const CompileResult& compile_result, string* header) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  TF_RETURN_IF_ERROR(ValidateFeedFetchCppNames(config));
  const int64 result_index = compile_result.aot->result_buffer_index();
  const xla::BufferSizes& temp_sizes = compile_result.aot->buffer_sizes();
  if (result_index < 0 || result_index > temp_sizes.size()) {
    return errors::InvalidArgument("result index: ", result_index,
                                   " is outside the range of temp sizes: [0,",
                                   temp_sizes.size(), ")");
  }

  // Compute sizes and generate methods.
  std::vector<int64> arg_sizes;
  TF_RETURN_IF_ERROR(ComputeArgSizes(compile_result, &arg_sizes));
  const xla::ProgramShape& ps = compile_result.program_shape;
  string methods_arg, methods_result;
  TF_RETURN_IF_ERROR(GenArgMethods(config, ps, compile_result, &methods_arg));
  TF_RETURN_IF_ERROR(GenResultMethods(config, ps, &methods_result));
  const std::vector<intptr_t> iarg(arg_sizes.begin(), arg_sizes.end());
  const std::vector<intptr_t> itemp(temp_sizes.begin(), temp_sizes.end());
  const size_t arg_bytes_aligned =
      runtime::aligned_buffer_bytes(iarg.data(), iarg.size());
  const size_t arg_bytes_total = total_buffer_bytes(iarg.data(), iarg.size());
  const size_t temp_bytes_aligned =
      runtime::aligned_buffer_bytes(itemp.data(), itemp.size());
  const size_t temp_bytes_total =
      total_buffer_bytes(itemp.data(), itemp.size());

  // Create rewrite strings for namespace start and end.
  string ns_start;
  for (const string& n : opts.namespaces) {
    ns_start += strings::StrCat("namespace ", n, " {\n");
  }
  ns_start += "\n";
  string ns_end("\n");
  for (int i = opts.namespaces.size() - 1; i >= 0; --i) {
    const string& n = opts.namespaces[i];
    ns_end += strings::StrCat("}  // end namespace ", n, "\n");
  }

  // Generate metadata.
  const string arg_names_code =
      GenNameToIndexCode(config.feed(), opts.gen_name_to_index);
  const string result_names_code =
      GenNameToIndexCode(config.fetch(), opts.gen_name_to_index);
  const string include_xla_data_proto =
      opts.gen_program_shape
          ?
          R"(#include "tensorflow/compiler/xla/xla_data.pb.h")"
          : "";
  const string program_shape_code =
      GenProgramShapeCode(ps, opts.gen_program_shape);

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
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }

// (Implementation detail) Entry point to the function in the object file.
extern "C" void {{ENTRY}}(
    void* result, const xla::ExecutableRunOptions* run_options,
    const void** args, void** temps);

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
class {{CLASS}} : public tensorflow::XlaCompiledCpuFunction {
 public:
  // Number of input arguments for the compiled computation.
  static constexpr size_t kNumArgs = {{ARG_NUM}};

  // Byte size of each argument buffer. There are kNumArgs entries.
  static const intptr_t* ArgSizes() {
    static constexpr intptr_t kArgSizes[kNumArgs] = {{{ARG_SIZES}}};
    return kArgSizes;
  }

  // Returns static data used to create an XlaCompiledCpuFunction.
  static const tensorflow::XlaCompiledCpuFunction::StaticData& StaticData() {
    static XlaCompiledCpuFunction::StaticData* kStaticData = [](){
      XlaCompiledCpuFunction::StaticData* data =
        new XlaCompiledCpuFunction::StaticData;
      data->raw_function = {{ENTRY}};
      data->arg_sizes = ArgSizes();
      data->num_args = kNumArgs;
      data->temp_sizes = TempSizes();
      data->num_temps = kNumTemps;
      data->result_index = kResultIndex;
      data->requires_runtime_context = {{HAS_CONTEXT_ARG}};
      data->arg_names = StaticArgNames();
      data->result_names = StaticResultNames();
      data->program_shape = StaticProgramShape();
      return data;
    }();
    return *kStaticData;
  }

  {{CLASS}}(AllocMode alloc_mode = AllocMode::ARGS_RESULTS_AND_TEMPS)
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
  //   called in AllocMode::RESULTS_AND_TEMPS_ONLY for each positional argument,
  //   to set the argument buffers.
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

 private:
  // Number of result and temporary buffers for the compiled computation.
  static constexpr size_t kNumTemps = {{TEMP_NUM}};
  // The 0-based index of the result tuple in the temporary buffers.
  static constexpr size_t kResultIndex = {{RESULT_INDEX}};

  // Byte size of each result / temporary buffer. There are kNumTemps entries.
  static const intptr_t* TempSizes() {
    static constexpr intptr_t kTempSizes[kNumTemps] = {{{TEMP_SIZES}}};
    return kTempSizes;
  }

  // Array of names of each positional argument, terminated by nullptr.
  static const char** StaticArgNames() {{ARG_NAMES_CODE}}

  // Array of names of each positional result, terminated by nullptr.
  static const char** StaticResultNames() {{RESULT_NAMES_CODE}}

  // Shape of the args and results.
  static const xla::ProgramShape* StaticProgramShape() {{PROGRAM_SHAPE_CODE}}
};
{{NS_END}}

#endif  // TFCOMPILE_GENERATED_{{ENTRY}}_H_

// clang-format on
)";
  // The replacement strategy is naive, but good enough for our purposes.
  const std::vector<std::pair<string, string>> rewrites = {
      {"{{ARG_BYTES_ALIGNED}}", strings::StrCat(arg_bytes_aligned)},
      {"{{ARG_BYTES_TOTAL}}", strings::StrCat(arg_bytes_total)},
      {"{{ARG_NAMES_CODE}}", arg_names_code},
      {"{{ARG_NUM}}", strings::StrCat(arg_sizes.size())},
      {"{{ARG_SIZES}}", str_util::Join(arg_sizes, ", ")},
      {"{{CLASS}}", opts.class_name},
      {"{{ENTRY}}", compile_result.entry_point},
      {"{{HAS_CONTEXT_ARG}}",
       compile_result.has_context_arg ? "true" : "false"},
      {"{{INCLUDE_XLA_DATA_PROTO}}", include_xla_data_proto},
      {"{{METHODS_ARG}}\n", methods_arg},
      {"{{METHODS_RESULT}}\n", methods_result},
      {"{{NS_END}}\n", ns_end},
      {"{{NS_START}}\n", ns_start},
      {"{{PROGRAM_SHAPE}}", xla::ShapeUtil::HumanString(ps)},
      {"{{PROGRAM_SHAPE_CODE}}", program_shape_code},
      {"{{RESULT_INDEX}}", strings::StrCat(result_index)},
      {"{{RESULT_NAMES_CODE}}", result_names_code},
      {"{{TEMP_BYTES_ALIGNED}}", strings::StrCat(temp_bytes_aligned)},
      {"{{TEMP_BYTES_TOTAL}}", strings::StrCat(temp_bytes_total)},
      {"{{TEMP_NUM}}", strings::StrCat(temp_sizes.size())},
      {"{{TEMP_SIZES}}", str_util::Join(temp_sizes, ", ")},
  };
  str_util::ReplaceAllPairs(header, rewrites);
  return Status::OK();
}

Status ParseCppClass(const string& cpp_class, string* class_name,
                     std::vector<string>* namespaces) {
  class_name->clear();
  namespaces->clear();
  size_t begin = 0;
  size_t end = 0;
  while ((end = cpp_class.find("::", begin)) != string::npos) {
    const string ns = cpp_class.substr(begin, end - begin);
    TF_RETURN_IF_ERROR(ValidateCppIdent(
        ns, "in namespace component of cpp_class: " + cpp_class));
    namespaces->push_back(ns);
    begin = end + 2;  // +2 to skip the two colons
  }
  const string name = cpp_class.substr(begin);
  TF_RETURN_IF_ERROR(
      ValidateCppIdent(name, "in class name of cpp_class: " + cpp_class));
  *class_name = name;
  return Status::OK();
}

Status ValidateCppIdent(StringPiece ident, StringPiece msg) {
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
  return Status::OK();
}

}  // namespace tfcompile
}  // namespace tensorflow
