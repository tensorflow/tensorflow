/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flatbuffers/flatc.h"

#include <list>

namespace flatbuffers {

const char *FLATC_VERSION() { return FLATBUFFERS_VERSION(); }

void FlatCompiler::ParseFile(
    flatbuffers::Parser &parser, const std::string &filename,
    const std::string &contents,
    std::vector<const char *> &include_directories) const {
  auto local_include_directory = flatbuffers::StripFileName(filename);
  include_directories.push_back(local_include_directory.c_str());
  include_directories.push_back(nullptr);
  if (!parser.Parse(contents.c_str(), &include_directories[0],
                    filename.c_str())) {
    Error(parser.error_, false, false);
  }
  if (!parser.error_.empty()) { Warn(parser.error_, false); }
  include_directories.pop_back();
  include_directories.pop_back();
}

void FlatCompiler::LoadBinarySchema(flatbuffers::Parser &parser,
                                    const std::string &filename,
                                    const std::string &contents) {
  if (!parser.Deserialize(reinterpret_cast<const uint8_t *>(contents.c_str()),
      contents.size())) {
    Error("failed to load binary schema: " + filename, false, false);
  }
}

void FlatCompiler::Warn(const std::string &warn, bool show_exe_name) const {
  params_.warn_fn(this, warn, show_exe_name);
}

void FlatCompiler::Error(const std::string &err, bool usage,
                         bool show_exe_name) const {
  params_.error_fn(this, err, usage, show_exe_name);
}

std::string FlatCompiler::GetUsageString(const char *program_name) const {
  std::stringstream ss;
  ss << "Usage: " << program_name << " [OPTION]... FILE... [-- FILE...]\n";
  for (size_t i = 0; i < params_.num_generators; ++i) {
    const Generator &g = params_.generators[i];

    std::stringstream full_name;
    full_name << std::setw(12) << std::left << g.generator_opt_long;
    const char *name = g.generator_opt_short ? g.generator_opt_short : "  ";
    const char *help = g.generator_help;

    ss << "  " << full_name.str() << " " << name << "    " << help << ".\n";
  }
  // clang-format off
  ss <<
    "  -o PATH            Prefix PATH to all generated files.\n"
    "  -I PATH            Search for includes in the specified path.\n"
    "  -M                 Print make rules for generated files.\n"
    "  --version          Print the version number of flatc and exit.\n"
    "  --strict-json      Strict JSON: field names must be / will be quoted,\n"
    "                     no trailing commas in tables/vectors.\n"
    "  --allow-non-utf8   Pass non-UTF-8 input through parser and emit nonstandard\n"
    "                     \\x escapes in JSON. (Default is to raise parse error on\n"
    "                     non-UTF-8 input.)\n"
    "  --natural-utf8     Output strings with UTF-8 as human-readable strings.\n"
    "                     By default, UTF-8 characters are printed as \\uXXXX escapes.\n"
    "  --defaults-json    Output fields whose value is the default when\n"
    "                     writing JSON\n"
    "  --unknown-json     Allow fields in JSON that are not defined in the\n"
    "                     schema. These fields will be discared when generating\n"
    "                     binaries.\n"
    "  --no-prefix        Don\'t prefix enum values with the enum type in C++.\n"
    "  --scoped-enums     Use C++11 style scoped and strongly typed enums.\n"
    "                     also implies --no-prefix.\n"
    "  --gen-includes     (deprecated), this is the default behavior.\n"
    "                     If the original behavior is required (no include\n"
    "                     statements) use --no-includes.\n"
    "  --no-includes      Don\'t generate include statements for included\n"
    "                     schemas the generated file depends on (C++).\n"
    "  --gen-mutable      Generate accessors that can mutate buffers in-place.\n"
    "  --gen-onefile      Generate single output file for C# and Go.\n"
    "  --gen-name-strings Generate type name functions for C++.\n"
    "  --gen-object-api   Generate an additional object-based API.\n"
    "  --gen-compare      Generate operator== for object-based API types.\n"
    "  --gen-nullable     Add Clang _Nullable for C++ pointer. or @Nullable for Java\n"
    "  --java-checkerframework Add @Pure for Java.\n"
    "  --gen-generated    Add @Generated annotation for Java\n"
    "  --gen-all          Generate not just code for the current schema files,\n"
    "                     but for all files it includes as well.\n"
    "                     If the language uses a single file for output (by default\n"
    "                     the case for C++ and JS), all code will end up in this one\n"
    "                     file.\n"
    "  --cpp-include      Adds an #include in generated file.\n"
    "  --cpp-ptr-type T   Set object API pointer type (default std::unique_ptr).\n"
    "  --cpp-str-type T   Set object API string type (default std::string).\n"
    "                     T::c_str(), T::length() and T::empty() must be supported.\n"
    "                     The custom type also needs to be constructible from std::string\n"
    "                     (see the --cpp-str-flex-ctor option to change this behavior).\n"
    "  --cpp-str-flex-ctor Don't construct custom string types by passing std::string\n"
    "                     from Flatbuffers, but (char* + length).\n"
    "  --object-prefix    Customise class prefix for C++ object-based API.\n"
    "  --object-suffix    Customise class suffix for C++ object-based API.\n"
    "                     Default value is \"T\".\n"
    "  --no-js-exports    Removes Node.js style export lines in JS.\n"
    "  --goog-js-export   Uses goog.exports* for closure compiler exporting in JS.\n"
    "  --es6-js-export    Uses ECMAScript 6 export style lines in JS.\n"
    "  --go-namespace     Generate the overrided namespace in Golang.\n"
    "  --go-import        Generate the overrided import for flatbuffers in Golang\n"
    "                     (default is \"github.com/google/flatbuffers/go\").\n"
    "  --raw-binary       Allow binaries without file_indentifier to be read.\n"
    "                     This may crash flatc given a mismatched schema.\n"
    "  --size-prefixed    Input binaries are size prefixed buffers.\n"
    "  --proto            Input is a .proto, translate to .fbs.\n"
    "  --oneof-union      Translate .proto oneofs to flatbuffer unions.\n"
    "  --grpc             Generate GRPC interfaces for the specified languages.\n"
    "  --schema           Serialize schemas instead of JSON (use with -b).\n"
    "  --bfbs-comments    Add doc comments to the binary schema files.\n"
    "  --bfbs-builtins    Add builtin attributes to the binary schema files.\n"
    "  --conform FILE     Specify a schema the following schemas should be\n"
    "                     an evolution of. Gives errors if not.\n"
    "  --conform-includes Include path for the schema given with --conform PATH\n"
    "  --include-prefix   Prefix this path to any generated include statements.\n"
    "    PATH\n"
    "  --keep-prefix      Keep original prefix of schema include statement.\n"
    "  --no-fb-import     Don't include flatbuffers import statement for TypeScript.\n"
    "  --no-ts-reexport   Don't re-export imported dependencies for TypeScript.\n"
    "  --short-names      Use short function names for JS and TypeScript.\n"
    "  --reflect-types    Add minimal type reflection to code generation.\n"
    "  --reflect-names    Add minimal type/name reflection.\n"
    "  --root-type T      Select or override the default root_type\n"
    "  --force-defaults   Emit default values in binary output from JSON\n"
    "  --force-empty      When serializing from object API representation,\n"
    "                     force strings and vectors to empty rather than null.\n"
    "FILEs may be schemas (must end in .fbs), binary schemas (must end in .bfbs),\n"
    "or JSON files (conforming to preceding schema). FILEs after the -- must be\n"
    "binary flatbuffer format files.\n"
    "Output files are named using the base file name of the input,\n"
    "and written to the current directory or the path given by -o.\n"
    "example: " << program_name << " -c -b schema1.fbs schema2.fbs data.json\n";
  // clang-format on
  return ss.str();
}

int FlatCompiler::Compile(int argc, const char **argv) {
  if (params_.generators == nullptr || params_.num_generators == 0) {
    return 0;
  }

  flatbuffers::IDLOptions opts;
  std::string output_path;

  bool any_generator = false;
  bool print_make_rules = false;
  bool raw_binary = false;
  bool schema_binary = false;
  bool grpc_enabled = false;
  std::vector<std::string> filenames;
  std::list<std::string> include_directories_storage;
  std::vector<const char *> include_directories;
  std::vector<const char *> conform_include_directories;
  std::vector<bool> generator_enabled(params_.num_generators, false);
  size_t binary_files_from = std::numeric_limits<size_t>::max();
  std::string conform_to_schema;

  for (int argi = 0; argi < argc; argi++) {
    std::string arg = argv[argi];
    if (arg[0] == '-') {
      if (filenames.size() && arg[1] != '-')
        Error("invalid option location: " + arg, true);
      if (arg == "-o") {
        if (++argi >= argc) Error("missing path following: " + arg, true);
        output_path = flatbuffers::ConCatPathFileName(
            flatbuffers::PosixPath(argv[argi]), "");
      } else if (arg == "-I") {
        if (++argi >= argc) Error("missing path following" + arg, true);
        include_directories_storage.push_back(
            flatbuffers::PosixPath(argv[argi]));
        include_directories.push_back(
            include_directories_storage.back().c_str());
      } else if (arg == "--conform") {
        if (++argi >= argc) Error("missing path following" + arg, true);
        conform_to_schema = flatbuffers::PosixPath(argv[argi]);
      } else if (arg == "--conform-includes") {
        if (++argi >= argc) Error("missing path following" + arg, true);
        include_directories_storage.push_back(
            flatbuffers::PosixPath(argv[argi]));
        conform_include_directories.push_back(
            include_directories_storage.back().c_str());
      } else if (arg == "--include-prefix") {
        if (++argi >= argc) Error("missing path following" + arg, true);
        opts.include_prefix = flatbuffers::ConCatPathFileName(
            flatbuffers::PosixPath(argv[argi]), "");
      } else if (arg == "--keep-prefix") {
        opts.keep_include_path = true;
      } else if (arg == "--strict-json") {
        opts.strict_json = true;
      } else if (arg == "--allow-non-utf8") {
        opts.allow_non_utf8 = true;
      } else if (arg == "--natural-utf8") {
        opts.natural_utf8 = true;
      } else if (arg == "--no-js-exports") {
        opts.skip_js_exports = true;
      } else if (arg == "--goog-js-export") {
        opts.use_goog_js_export_format = true;
        opts.use_ES6_js_export_format = false;
      } else if (arg == "--es6-js-export") {
        opts.use_goog_js_export_format = false;
        opts.use_ES6_js_export_format = true;
      } else if (arg == "--go-namespace") {
        if (++argi >= argc) Error("missing golang namespace" + arg, true);
        opts.go_namespace = argv[argi];
      } else if (arg == "--go-import") {
        if (++argi >= argc) Error("missing golang import" + arg, true);
        opts.go_import = argv[argi];
      } else if (arg == "--defaults-json") {
        opts.output_default_scalars_in_json = true;
      } else if (arg == "--unknown-json") {
        opts.skip_unexpected_fields_in_json = true;
      } else if (arg == "--no-prefix") {
        opts.prefixed_enums = false;
      } else if (arg == "--scoped-enums") {
        opts.prefixed_enums = false;
        opts.scoped_enums = true;
      } else if (arg == "--no-union-value-namespacing") {
        opts.union_value_namespacing = false;
      } else if (arg == "--gen-mutable") {
        opts.mutable_buffer = true;
      } else if (arg == "--gen-name-strings") {
        opts.generate_name_strings = true;
      } else if (arg == "--gen-object-api") {
        opts.generate_object_based_api = true;
      } else if (arg == "--gen-compare") {
        opts.gen_compare = true;
      } else if (arg == "--cpp-include") {
        if (++argi >= argc) Error("missing include following" + arg, true);
        opts.cpp_includes.push_back(argv[argi]);
      } else if (arg == "--cpp-ptr-type") {
        if (++argi >= argc) Error("missing type following" + arg, true);
        opts.cpp_object_api_pointer_type = argv[argi];
      } else if (arg == "--cpp-str-type") {
        if (++argi >= argc) Error("missing type following" + arg, true);
        opts.cpp_object_api_string_type = argv[argi];
      } else if (arg == "--cpp-str-flex-ctor") {
        opts.cpp_object_api_string_flexible_constructor = true;
      } else if (arg == "--gen-nullable") {
        opts.gen_nullable = true;
      } else if (arg == "--java-checkerframework") {
        opts.java_checkerframework = true;
      } else if (arg == "--gen-generated") {
        opts.gen_generated = true;
      } else if (arg == "--object-prefix") {
        if (++argi >= argc) Error("missing prefix following" + arg, true);
        opts.object_prefix = argv[argi];
      } else if (arg == "--object-suffix") {
        if (++argi >= argc) Error("missing suffix following" + arg, true);
        opts.object_suffix = argv[argi];
      } else if (arg == "--gen-all") {
        opts.generate_all = true;
        opts.include_dependence_headers = false;
      } else if (arg == "--gen-includes") {
        // Deprecated, remove this option some time in the future.
        printf("warning: --gen-includes is deprecated (it is now default)\n");
      } else if (arg == "--no-includes") {
        opts.include_dependence_headers = false;
      } else if (arg == "--gen-onefile") {
        opts.one_file = true;
      } else if (arg == "--raw-binary") {
        raw_binary = true;
      } else if (arg == "--size-prefixed") {
        opts.size_prefixed = true;
      } else if (arg == "--") {  // Separator between text and binary inputs.
        binary_files_from = filenames.size();
      } else if (arg == "--proto") {
        opts.proto_mode = true;
      } else if (arg == "--oneof-union") {
        opts.proto_oneof_union = true;
      } else if (arg == "--schema") {
        schema_binary = true;
      } else if (arg == "-M") {
        print_make_rules = true;
      } else if (arg == "--version") {
        printf("flatc version %s\n", FLATC_VERSION());
        exit(0);
      } else if (arg == "--grpc") {
        grpc_enabled = true;
      } else if (arg == "--bfbs-comments") {
        opts.binary_schema_comments = true;
      } else if (arg == "--bfbs-builtins") {
        opts.binary_schema_builtins = true;
      } else if (arg == "--no-fb-import") {
        opts.skip_flatbuffers_import = true;
      } else if (arg == "--no-ts-reexport") {
        opts.reexport_ts_modules = false;
      } else if (arg == "--short-names") {
        opts.js_ts_short_names = true;
      } else if (arg == "--reflect-types") {
        opts.mini_reflect = IDLOptions::kTypes;
      } else if (arg == "--reflect-names") {
        opts.mini_reflect = IDLOptions::kTypesAndNames;
      } else if (arg == "--root-type") {
        if (++argi >= argc) Error("missing type following" + arg, true);
        opts.root_type = argv[argi];
      } else if (arg == "--force-defaults") {
        opts.force_defaults = true;
      } else if (arg == "--force-empty") {
        opts.set_empty_to_null = false;
      } else if (arg == "--java-primitive-has-method") {
        opts.java_primitive_has_method = true;
      } else {
        for (size_t i = 0; i < params_.num_generators; ++i) {
          if (arg == params_.generators[i].generator_opt_long ||
              (params_.generators[i].generator_opt_short &&
               arg == params_.generators[i].generator_opt_short)) {
            generator_enabled[i] = true;
            any_generator = true;
            opts.lang_to_generate |= params_.generators[i].lang;
            goto found;
          }
        }
        Error("unknown commandline argument: " + arg, true);
      found:;
      }
    } else {
      filenames.push_back(flatbuffers::PosixPath(argv[argi]));
    }
  }

  if (!filenames.size()) Error("missing input files", false, true);

  if (opts.proto_mode) {
    if (any_generator)
      Error("cannot generate code directly from .proto files", true);
  } else if (!any_generator && conform_to_schema.empty()) {
    Error("no options: specify at least one generator.", true);
  }

  flatbuffers::Parser conform_parser;
  if (!conform_to_schema.empty()) {
    std::string contents;
    if (!flatbuffers::LoadFile(conform_to_schema.c_str(), true, &contents))
      Error("unable to load schema: " + conform_to_schema);

    if (flatbuffers::GetExtension(conform_to_schema) ==
        reflection::SchemaExtension()) {
      LoadBinarySchema(conform_parser, conform_to_schema, contents);
    } else {
      ParseFile(conform_parser, conform_to_schema, contents,
                conform_include_directories);
    }
  }

  std::unique_ptr<flatbuffers::Parser> parser(new flatbuffers::Parser(opts));

  for (auto file_it = filenames.begin(); file_it != filenames.end();
       ++file_it) {
    auto &filename = *file_it;
    std::string contents;
    if (!flatbuffers::LoadFile(filename.c_str(), true, &contents))
      Error("unable to load file: " + filename);

    bool is_binary =
        static_cast<size_t>(file_it - filenames.begin()) >= binary_files_from;
    auto ext = flatbuffers::GetExtension(filename);
    auto is_schema = ext == "fbs" || ext == "proto";
    auto is_binary_schema = ext == reflection::SchemaExtension();
    if (is_binary) {
      parser->builder_.Clear();
      parser->builder_.PushFlatBuffer(
          reinterpret_cast<const uint8_t *>(contents.c_str()),
          contents.length());
      if (!raw_binary) {
        // Generally reading binaries that do not correspond to the schema
        // will crash, and sadly there's no way around that when the binary
        // does not contain a file identifier.
        // We'd expect that typically any binary used as a file would have
        // such an identifier, so by default we require them to match.
        if (!parser->file_identifier_.length()) {
          Error("current schema has no file_identifier: cannot test if \"" +
                filename +
                "\" matches the schema, use --raw-binary to read this file"
                " anyway.");
        } else if (!flatbuffers::BufferHasIdentifier(
                       contents.c_str(), parser->file_identifier_.c_str(), opts.size_prefixed)) {
          Error("binary \"" + filename +
                "\" does not have expected file_identifier \"" +
                parser->file_identifier_ +
                "\", use --raw-binary to read this file anyway.");
        }
      }
    } else {
      // Check if file contains 0 bytes.
      if (!is_binary_schema && contents.length() != strlen(contents.c_str())) {
        Error("input file appears to be binary: " + filename, true);
      }
      if (is_schema) {
        // If we're processing multiple schemas, make sure to start each
        // one from scratch. If it depends on previous schemas it must do
        // so explicitly using an include.
        parser.reset(new flatbuffers::Parser(opts));
      }
      if (is_binary_schema) {
        LoadBinarySchema(*parser.get(), filename, contents);
      } else {
        ParseFile(*parser.get(), filename, contents, include_directories);
        if (!is_schema && !parser->builder_.GetSize()) {
          // If a file doesn't end in .fbs, it must be json/binary. Ensure we
          // didn't just parse a schema with a different extension.
          Error("input file is neither json nor a .fbs (schema) file: " +
                    filename,
                true);
        }
      }
      if ((is_schema || is_binary_schema) && !conform_to_schema.empty()) {
        auto err = parser->ConformTo(conform_parser);
        if (!err.empty()) Error("schemas don\'t conform: " + err);
      }
      if (schema_binary) {
        parser->Serialize();
        parser->file_extension_ = reflection::SchemaExtension();
      }
    }

    std::string filebase =
        flatbuffers::StripPath(flatbuffers::StripExtension(filename));

    for (size_t i = 0; i < params_.num_generators; ++i) {
      parser->opts.lang = params_.generators[i].lang;
      if (generator_enabled[i]) {
        if (!print_make_rules) {
          flatbuffers::EnsureDirExists(output_path);
          if ((!params_.generators[i].schema_only ||
               (is_schema || is_binary_schema)) &&
              !params_.generators[i].generate(*parser.get(), output_path,
                                              filebase)) {
            Error(std::string("Unable to generate ") +
                  params_.generators[i].lang_name + " for " + filebase);
          }
        } else {
          std::string make_rule = params_.generators[i].make_rule(
              *parser.get(), output_path, filename);
          if (!make_rule.empty())
            printf("%s\n",
                   flatbuffers::WordWrap(make_rule, 80, " ", " \\").c_str());
        }
        if (grpc_enabled) {
          if (params_.generators[i].generateGRPC != nullptr) {
            if (!params_.generators[i].generateGRPC(*parser.get(), output_path,
                                                    filebase)) {
              Error(std::string("Unable to generate GRPC interface for") +
                    params_.generators[i].lang_name);
            }
          } else {
            Warn(std::string("GRPC interface generator not implemented for ") +
                 params_.generators[i].lang_name);
          }
        }
      }
    }

    if (!opts.root_type.empty()) {
      if (!parser->SetRootType(opts.root_type.c_str()))
        Error("unknown root type: " + opts.root_type);
      else if (parser->root_struct_def_->fixed)
        Error("root type must be a table");
    }

    if (opts.proto_mode) GenerateFBS(*parser.get(), output_path, filebase);

    // We do not want to generate code for the definitions in this file
    // in any files coming up next.
    parser->MarkGenerated();
  }
  return 0;
}

}  // namespace flatbuffers
