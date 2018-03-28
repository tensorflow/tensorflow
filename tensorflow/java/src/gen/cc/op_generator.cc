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

#include <string>
#include <map>
#include <vector>
#include <list>
#include <memory>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/op_parser.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"

namespace tensorflow {
namespace java {
namespace {

const char* kLicenseSnippet =
    "tensorflow/java/src/gen/resources/license.snippet.java";

const std::map<string, Type> kPrimitiveAttrTypes = {
  { "Boolean", Type::Boolean() },
  { "Byte", Type::Byte() },
  { "Character", Type::Byte() },
  { "Float", Type::Float() },
  { "Integer", Type::Long() },
  { "Long", Type::Long() },
  { "Short", Type::Long() },
  { "Double", Type::Float() },
};

enum RenderMode {
  DEFAULT,
  SINGLE_OUTPUT,
  SINGLE_LIST_OUTPUT
};

void CollectOpDependencies(const OpSpec& op, RenderMode mode,
    std::list<Type>* out) {
  out->push_back(Type::Class("Operation", "org.tensorflow"));
  out->push_back(Type::Class("OperationBuilder", "org.tensorflow"));
  out->push_back(Type::Class("Scope", "org.tensorflow.op"));
  if (mode == SINGLE_OUTPUT) {
    out->push_back(Type::Class("Output", "org.tensorflow"));
  } else if (mode == SINGLE_LIST_OUTPUT) {
    out->push_back(Type::Interface("Iterator", "java.util"));
  }
  // Don't pay attention to duplicate types in the dependency list, they will
  // be filtered out by the SourceWriter.
  for (const OpSpec::Operand& input : op.inputs()) {
    out->push_back(input.var().type());
    if (input.iterable()) {
      out->push_back(Type::Class("Operands", "org.tensorflow.op"));
    }
  }
  for (const OpSpec::Operand& output : op.outputs()) {
    out->push_back(output.var().type());
    if (output.iterable()) {
      out->push_back(Type::Class("Arrays", "java.util"));
    }
  }
  for (const OpSpec::Operand& attribute : op.attributes()) {
    out->push_back(attribute.var().type());
    if (attribute.var().type().name() == "Class") {
      out->push_back(Type::Enum("DataType", "org.tensorflow"));
    }
  }
  for (const OpSpec::Operand& option : op.options()) {
    out->push_back(option.var().type());
  }
}

void WriteSetAttrDirective(const OpSpec::Operand& attr, bool optional,
    SourceWriter* writer) {
  string var = optional ? "opts." + attr.var().name() : attr.var().name();
  if (attr.iterable()) {
    const Type& type = attr.data_type();
    std::map<string, Type>::const_iterator it =
      kPrimitiveAttrTypes.find(type.name());
    if (it != kPrimitiveAttrTypes.end()) {
      string array = attr.var().name() + "Array";
      writer->AppendType(it->second)
          .Append("[] " + array + " = new ")
          .AppendType(it->second)
          .Append("[" + var + ".size()];")
          .EndLine();
      writer->BeginBlock("for (int i = 0; i < " + array + ".length; ++i)")
          .Append(array + "[i] = " + var + ".get(i);")
          .EndLine()
          .EndBlock()
          .Append("opBuilder.setAttr(\"" + attr.graph_name() + "\", " + array)
          .Append(");")
          .EndLine();
    } else {
      writer->Append("opBuilder.setAttr(\"" + attr.graph_name() + "\", " + var)
          .Append(".toArray(new ")
          .AppendType(type)
          .Append("[" + var + ".size()]));")
          .EndLine();
    }
  } else {
    Type type = attr.var().type();
    writer->Append("opBuilder.setAttr(\"" + attr.graph_name() + "\", ");
    if (type.name() == "Class") {
      writer->Append("DataType.fromClass(" + attr.var().name() + "));");
    } else {
      writer->Append(var + ");");
    }
    writer->EndLine();
  }
}

void RenderFactoryMethod(const OpSpec& op, const Type& op_class,
    SourceWriter* writer) {
  Method factory = Method::Create("create", op_class);
  Javadoc factory_doc = Javadoc::Create(
      "Factory method to create a class to wrap a new " + op_class.name()
      + " operation to the graph.");
  Variable scope =
      Variable::Create("scope", Type::Class("Scope", "org.tensorflow.op"));
  factory.add_argument(scope);
  factory_doc.add_param_tag(scope.name(), "Current graph scope");
  for (const OpSpec::Operand& input : op.inputs()) {
    factory.add_argument(input.var());
    factory_doc.add_param_tag(input.var().name(), input.description());
  }
  for (const OpSpec::Operand& attribute : op.attributes()) {
    factory.add_argument(attribute.var());
    factory_doc.add_param_tag(attribute.var().name(), attribute.description());
  }
  if (!op.options().empty()) {
    factory.add_argument(Variable::Varargs("options", Type::Class("Options")));
    factory_doc.add_param_tag("options", "carries optional attributes values");
  }
  factory_doc.add_tag("return", "a new instance of " + op_class.name());
  writer->BeginMethod(factory, PUBLIC|STATIC, &factory_doc);
  writer->Append("OperationBuilder opBuilder = scope.graph().opBuilder(\""
      + op.graph_name() + "\", scope.makeOpName(\""
      + op_class.name() + "\"));");
  writer->EndLine();

  for (const OpSpec::Operand& input : op.inputs()) {
    if (input.iterable()) {
      writer->Append("opBuilder.addInputList(Operands.asOutputs("
          + input.var().name() + "));");
      writer->EndLine();
    } else {
      writer->Append("opBuilder.addInput(" + input.var().name()
          + ".asOutput());");
      writer->EndLine();
    }
  }
  for (const OpSpec::Operand& attribute : op.attributes()) {
    WriteSetAttrDirective(attribute, false, writer);
  }
  if (!op.options().empty()) {
    writer->BeginBlock("if (options != null)")
        .BeginBlock("for (Options opts : options)");
    for (const OpSpec::Operand& option : op.options()) {
      writer->BeginBlock("if (opts." + option.var().name() + " != null)");
      WriteSetAttrDirective(option, true, writer);
      writer->EndBlock();
    }
    writer->EndBlock().EndBlock();
  }
  writer->Append("return new ")
      .AppendType(op_class)
      .Append("(opBuilder.build());")
      .EndLine();
  writer->EndMethod();
}

void RenderConstructor(const OpSpec& op, const Type& op_class,
    SourceWriter* writer) {
  Method constructor = Method::ConstructorFor(op_class)
    .add_argument(
        Variable::Create("operation",
            Type::Class("Operation", "org.tensorflow")));
  for (const OpSpec::Operand& output : op.outputs()) {
    if (output.iterable() && !output.data_type().unknown()) {
      constructor.add_annotation(
          Annotation::Create("SuppressWarnings").attributes("\"unchecked\""));
      break;
    }
  }
  writer->BeginMethod(constructor, PRIVATE)
      .Append("super(operation);")
      .EndLine();
  if (op.outputs().size() > 0) {
    writer->Append("int outputIdx = 0;")
        .EndLine();
    for (const OpSpec::Operand& output : op.outputs()) {
      if (output.iterable()) {
        string var_length = output.var().name() + "Length";
        writer->Append("int " + var_length)
            .Append(" = operation.outputListLength(\"" + output.graph_name()
                + "\");")
            .EndLine()
            .Append(output.var().name() + " = Arrays.asList(");
        if (!output.data_type().unknown()) {
          writer->Append("(")
              .AppendType(output.var().type().parameters().front())
              .Append("[])");
        }
        writer->Append("operation.outputList(outputIdx, " + var_length + "));")
            .EndLine()
            .Append("outputIdx += " + var_length + ";")
            .EndLine();
      } else {
        writer->Append(output.var().name()
                + " = operation.output(outputIdx++);")
            .EndLine();
      }
    }
  }
  writer->EndMethod();
}

void RenderGettersAndSetters(const OpSpec& op, SourceWriter* writer) {
  for (const OpSpec::Operand& option : op.options()) {
    Method setter = Method::Create(option.var().name(), Type::Class("Options"))
        .add_argument(option.var());
    Javadoc setter_doc = Javadoc::Create()
        .add_param_tag(option.var().name(), option.description());
    writer->BeginMethod(setter, PUBLIC|STATIC, &setter_doc)
        .Append("return new Options()." + option.var().name() + "("
            + option.var().name() + ");")
        .EndLine()
        .EndMethod();
  }
  for (const OpSpec::Operand& output : op.outputs()) {
    Method getter = Method::Create(output.var().name(), output.var().type());
    Javadoc getter_doc = Javadoc::Create(output.description());
    writer->BeginMethod(getter, PUBLIC, &getter_doc)
        .Append("return " + output.var().name() + ";")
        .EndLine()
        .EndMethod();
  }
}

void RenderInterfaceImpl(const OpSpec& op, RenderMode mode,
    SourceWriter* writer) {
  OpSpec::Operand output = op.outputs().front();

  if (mode == SINGLE_OUTPUT) {
    bool cast2obj = output.data_type().unknown();
    Type return_type = Type::Class("Output", "org.tensorflow")
        .add_parameter(cast2obj ? Type::Class("Object") : output.data_type());
    Method as_output = Method::Create("asOutput", return_type)
        .add_annotation(Annotation::Create("Override"));
    if (cast2obj) {
      as_output.add_annotation(
          Annotation::Create("SuppressWarnings").attributes("\"unchecked\""));
    }
    writer->BeginMethod(as_output, PUBLIC);
    if (cast2obj) {
      writer->Append("return (").AppendType(return_type).Append(") ");
    } else {
      writer->Append("return ");
    }
    writer->Append(output.var().name() + ";")
        .EndLine()
        .EndMethod();

  } else if (mode == SINGLE_LIST_OUTPUT) {
    Type operand = Type::Interface("Operand", "org.tensorflow");
    if (output.data_type().unknown()) {
      operand.add_parameter(Type::Class("Object"));
    } else {
      operand.add_parameter(output.data_type());
    }
    Type return_type = Type::Interface("Iterator", "java.util")
        .add_parameter(operand);
    Method iterator = Method::Create("iterator", return_type)
        .add_annotation(Annotation::Create("Override"))
        .add_annotation(Annotation::Create("SuppressWarnings")
            .attributes("{\"rawtypes\", \"unchecked\"}"));
    // cast the output list using a raw List
    writer->BeginMethod(iterator, PUBLIC)
        .Append("return (" + return_type.name() + ") ")
        .Append(output.var().name() + ".iterator();")
        .EndLine()
        .EndMethod();
  }
}

void RenderOptionsClass(const OpSpec& op, SourceWriter* writer) {
  Type options_class = Type::Class("Options");
  Javadoc options_doc = Javadoc::Create(
      "Class holding optional attributes of this operation");
  writer->BeginInnerType(options_class, PUBLIC | STATIC, &options_doc);
  for (const OpSpec::Operand& option : op.options()) {
    Method setter = Method::Create(option.var().name(), options_class)
        .add_argument(option.var());
    Javadoc setter_doc = Javadoc::Create()
        .add_param_tag(option.var().name(), option.description());
    writer->BeginMethod(setter, PUBLIC, &setter_doc)
        .Append("this." + option.var().name() + " = " + option.var().name()
            + ";")
        .EndLine()
        .Append("return this;")
        .EndLine()
        .EndMethod();
  }
  writer->EndLine();
  for (const OpSpec::Operand& option : op.options()) {
    writer->WriteField(option.var(), PRIVATE);
  }
  Method constructor = Method::ConstructorFor(options_class);
  writer->BeginMethod(constructor, PRIVATE).EndMethod();
  writer->EndType();
}

void RenderEndpoint(const OpSpec& op, const OpSpec::Endpoint& endpoint,
    SourceWriter* writer) {
  RenderMode mode = DEFAULT;
  if (op.outputs().size() == 1) {
    mode = op.outputs().front().iterable() ? SINGLE_LIST_OUTPUT : SINGLE_OUTPUT;
  }
  std::list<Type> dependencies;
  CollectOpDependencies(op, mode, &dependencies);
  const Type& op_class = endpoint.type();
  writer->WriteFromFile(kLicenseSnippet)
      .EndLine()
      .Append("// This file is machine generated, DO NOT EDIT!")
      .EndLine()
      .EndLine()
      .BeginType(op_class, PUBLIC|FINAL, &dependencies, &endpoint.javadoc());
  if (!op.options().empty()) {
    RenderOptionsClass(op, writer);
  }
  RenderFactoryMethod(op, op_class, writer);
  RenderGettersAndSetters(op, writer);
  if (mode != DEFAULT) {
    RenderInterfaceImpl(op, mode, writer);
  }
  writer->EndLine();
  for (const OpSpec::Operand& output : op.outputs()) {
    writer->WriteField(output.var(), PRIVATE);
  }
  RenderConstructor(op, op_class, writer);
  writer->EndType();
}

}  // namespace

OpGenerator::OpGenerator(const string& base_package, const string& output_dir,
    const std::vector<string>& api_dirs, Env* env)
  : base_package_(base_package), output_dir_(output_dir), api_dirs_(api_dirs),
    env_(env) {
}

Status OpGenerator::Run(const OpList& op_list, const string& lib_name) {
  LOG(INFO) << "Generating Java wrappers for '" << lib_name << "' operations";
  ApiDefMap api_map(op_list);
  if (!api_dirs_.empty()) {
    // Only load api files that correspond to the requested "op_list"
    for (const auto& op : op_list.op()) {
      for (const auto& api_def_dir : api_dirs_) {
        const std::string api_def_file_pattern =
            io::JoinPath(api_def_dir, "api_def_" + op.name() + ".pbtxt");
        if (env_->FileExists(api_def_file_pattern).ok()) {
          TF_CHECK_OK(api_map.LoadFile(env_, api_def_file_pattern));
        }
      }
    }
  }
  api_map.UpdateDocs();
  for (const auto& op_def : op_list.op()) {
    const ApiDef* api_def = api_map.GetApiDef(op_def.name());
    if (api_def->visibility() != ApiDef::SKIP) {
      Status status = GenerateOp(op_def, *api_def, lib_name);
      if (status != Status::OK()) {
        LOG(ERROR) << "Fail to generate Java wrapper for operation \""
            << op_def.name() << "\"";
      }
    }
  }
  return Status::OK();
}

Status OpGenerator::GenerateOp(const OpDef& op_def, const ApiDef& api_def,
    const string& lib_name) {
  std::unique_ptr<OpSpec> op;
  OpParser op_parser(op_def, api_def, lib_name, base_package_);
  op_parser.Parse(&op);
  for (const OpSpec::Endpoint& endpoint : op->endpoints()) {
    string package_path = io::JoinPath(output_dir_,
        str_util::StringReplace(endpoint.type().package(), ".", "/", true));
    if (!env_->FileExists(package_path).ok()) {
      TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(package_path));
    }
    string file_path =
        io::JoinPath(package_path, endpoint.type().name() + ".java");
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_CHECK_OK(env_->NewWritableFile(file_path, &file));

    SourceFileWriter writer(file.get());
    RenderEndpoint(*op, endpoint, &writer);
  }
  return Status::OK();
}

}  // namespace java
}  // namespace tensorflow
