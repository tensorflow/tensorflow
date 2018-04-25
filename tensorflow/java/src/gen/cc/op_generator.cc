/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <set>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/op_generator.h"
#include "tensorflow/java/src/gen/cc/op_specs.h"

namespace tensorflow {
namespace java {
namespace {

const char* kLicenseSnippet =
    "tensorflow/java/src/gen/resources/license.java.snippet";

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
  for (const ArgumentSpec& input : op.inputs()) {
    out->push_back(input.var().type());
    if (input.iterable()) {
      out->push_back(Type::Class("Operands", "org.tensorflow.op"));
    }
  }
  for (const ArgumentSpec& output : op.outputs()) {
    out->push_back(output.var().type());
    if (output.iterable()) {
      out->push_back(Type::Class("Arrays", "java.util"));
    }
  }
  for (const AttributeSpec& attribute : op.attributes()) {
    out->push_back(attribute.var().type());
    if (attribute.var().type().name() == "Class") {
      out->push_back(Type::Enum("DataType", "org.tensorflow"));
    }
  }
  for (const AttributeSpec& optional_attribute : op.optional_attributes()) {
    out->push_back(optional_attribute.var().type());
  }
}

void WriteSetAttrDirective(const AttributeSpec& attr, bool optional,
    SourceWriter* writer) {
  string var = optional ? "opts." + attr.var().name() : attr.var().name();
  if (attr.iterable()) {
    const Type& type = attr.type();
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
          .Append("opBuilder.setAttr(\"" + attr.op_def_name() + "\", " + array)
          .Append(");")
          .EndLine();
    } else {
      writer->Append("opBuilder.setAttr(\"" + attr.op_def_name() + "\", " + var)
          .Append(".toArray(new ")
          .AppendType(type)
          .Append("[" + var + ".size()]));")
          .EndLine();
    }
  } else {
    Type type = attr.var().type();
    writer->Append("opBuilder.setAttr(\"" + attr.op_def_name() + "\", ");
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
  for (const ArgumentSpec& input : op.inputs()) {
    factory.add_argument(input.var());
    factory_doc.add_param_tag(input.var().name(), input.description());
  }
  for (const AttributeSpec& attribute : op.attributes()) {
    factory.add_argument(attribute.var());
    factory_doc.add_param_tag(attribute.var().name(), attribute.description());
  }
  if (!op.optional_attributes().empty()) {
    factory.add_argument(Variable::Varargs("options", Type::Class("Options")));
    factory_doc.add_param_tag("options", "carries optional attributes values");
  }
  factory_doc.add_tag("return", "a new instance of " + op_class.name());
  writer->BeginMethod(factory, PUBLIC|STATIC, &factory_doc);
  writer->Append("OperationBuilder opBuilder = scope.graph().opBuilder(\""
      + op.graph_op_name() + "\", scope.makeOpName(\""
      + op_class.name() + "\"));");
  writer->EndLine();

  for (const ArgumentSpec& input : op.inputs()) {
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
  for (const AttributeSpec& attribute : op.attributes()) {
    WriteSetAttrDirective(attribute, false, writer);
  }
  if (!op.optional_attributes().empty()) {
    writer->BeginBlock("if (options != null)")
        .BeginBlock("for (Options opts : options)");
    for (const AttributeSpec& attribute : op.optional_attributes()) {
      writer->BeginBlock("if (opts." + attribute.var().name() + " != null)");
      WriteSetAttrDirective(attribute, true, writer);
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
  for (const ArgumentSpec& output : op.outputs()) {
    if (output.iterable() && !output.type().unknown()) {
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
    for (const ArgumentSpec& output : op.outputs()) {
      if (output.iterable()) {
        string var_length = output.var().name() + "Length";
        writer->Append("int " + var_length)
            .Append(" = operation.outputListLength(\"" + output.op_def_name()
                + "\");")
            .EndLine()
            .Append(output.var().name() + " = Arrays.asList(");
        if (!output.type().unknown()) {
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
  for (const AttributeSpec& attribute : op.optional_attributes()) {
    Method setter =
        Method::Create(attribute.var().name(), Type::Class("Options"))
            .add_argument(attribute.var());
    Javadoc setter_doc = Javadoc::Create()
        .add_param_tag(attribute.var().name(), attribute.description());
    writer->BeginMethod(setter, PUBLIC|STATIC, &setter_doc)
        .Append("return new Options()." + attribute.var().name() + "("
            + attribute.var().name() + ");")
        .EndLine()
        .EndMethod();
  }
  for (const ArgumentSpec& output : op.outputs()) {
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
  ArgumentSpec output = op.outputs().front();

  if (mode == SINGLE_OUTPUT) {
    bool cast2obj = output.type().unknown();
    Type return_type = Type::Class("Output", "org.tensorflow")
        .add_parameter(cast2obj ? Type::Class("Object") : output.type());
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
    if (output.type().unknown()) {
      operand.add_parameter(Type::Class("Object"));
    } else {
      operand.add_parameter(output.type());
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

void RenderOptionsClass(const OpSpec& op, const Type& op_class,
    SourceWriter* writer) {
  Type options_class = Type::Class("Options");
  Javadoc options_doc = Javadoc::Create(
      "Optional attributes for {@link " + op_class.full_name() + "}");
  writer->BeginInnerType(options_class, PUBLIC | STATIC, &options_doc);
  for (const AttributeSpec& attribute : op.optional_attributes()) {
    Method setter = Method::Create(attribute.var().name(), options_class)
        .add_argument(attribute.var());
    Javadoc setter_doc = Javadoc::Create()
        .add_param_tag(attribute.var().name(), attribute.description());
    writer->BeginMethod(setter, PUBLIC, &setter_doc)
        .Append("this." + attribute.var().name() + " = "
            + attribute.var().name() + ";")
        .EndLine()
        .Append("return this;")
        .EndLine()
        .EndMethod();
  }
  writer->EndLine();
  for (const AttributeSpec& optional_attribute : op.optional_attributes()) {
    writer->WriteField(optional_attribute.var(), PRIVATE);
  }
  Method constructor = Method::ConstructorFor(options_class);
  writer->BeginMethod(constructor, PRIVATE).EndMethod();
  writer->EndType();
}

inline Type ClassOf(const EndpointSpec& endpoint, const string& base_package) {
  return Type::Class(endpoint.name(),
      base_package + "." + str_util::Lowercase(endpoint.package()));
}

void GenerateOp(const OpSpec& op, const EndpointSpec& endpoint,
    const string& base_package, const string& output_dir, Env* env) {
  Type op_class(ClassOf(endpoint, base_package)
      .add_supertype(Type::Class("PrimitiveOp", "org.tensorflow.op")));
  Javadoc op_javadoc(endpoint.javadoc());

  // implement Operand (or Iterable<Operand>) if the op has only one output
  RenderMode mode = DEFAULT;
  if (op.outputs().size() == 1) {
    const ArgumentSpec& output = op.outputs().front();
    Type operand_type(output.type().unknown() ?
        Type::Class("Object") : output.type());
    Type operand_inf(Type::Interface("Operand", "org.tensorflow")
        .add_parameter(operand_type));
    if (output.iterable()) {
      mode = SINGLE_LIST_OUTPUT;
      op_class.add_supertype(Type::IterableOf(operand_inf));
    } else {
      mode = SINGLE_OUTPUT;
      op_class.add_supertype(operand_inf);
    }
  }
  // declare all outputs generics at the op class level
  std::set<string> generics;
  for (const ArgumentSpec& output : op.outputs()) {
    if (output.type().kind() == Type::GENERIC && !output.type().unknown()
        && generics.find(output.type().name()) == generics.end()) {
      op_class.add_parameter(output.type());
      op_javadoc.add_param_tag("<" + output.type().name() + ">",
          "data type of output {@code " + output.var().name() + "}");
      generics.insert(output.type().name());
    }
  }
  // handle endpoint deprecation
  if (endpoint.deprecated()) {
    op_class.add_annotation(Annotation::Create("Deprecated"));
    string explanation;
    if (!op.endpoints().front().deprecated()) {
      explanation = "use {@link " +
          ClassOf(op.endpoints().front(), base_package).full_name()
          + "} instead";
    } else {
      explanation = op.deprecation_explanation();
    }
    op_javadoc.add_tag("deprecated", explanation);
  }
  // expose the op in the Ops Graph API only if it is visible
  if (!op.hidden()) {
    op_class.add_annotation(
        Annotation::Create("Operator", "org.tensorflow.op.annotation")
          .attributes("group = \"" + endpoint.package() + "\""));
  }
  // create op class file
  string op_dir = io::JoinPath(output_dir,
      str_util::StringReplace(op_class.package(), ".", "/", true));
  if (!env->FileExists(op_dir).ok()) {
    TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(op_dir));
  }
  std::unique_ptr<tensorflow::WritableFile> op_file;
  TF_CHECK_OK(env->NewWritableFile(
      io::JoinPath(op_dir, op_class.name() + ".java"), &op_file));

  // render endpoint source code
  SourceFileWriter writer(op_file.get());
  std::list<Type> dependencies;
  CollectOpDependencies(op, mode, &dependencies);
  writer.WriteFromFile(kLicenseSnippet)
      .EndLine()
      .Append("// This file is machine generated, DO NOT EDIT!")
      .EndLine()
      .EndLine()
      .BeginType(op_class, PUBLIC|FINAL, &dependencies, &op_javadoc);
  if (!op.optional_attributes().empty()) {
    RenderOptionsClass(op, op_class, &writer);
  }
  RenderFactoryMethod(op, op_class, &writer);
  RenderGettersAndSetters(op, &writer);
  if (mode != DEFAULT) {
    RenderInterfaceImpl(op, mode, &writer);
  }
  writer.EndLine();
  for (const ArgumentSpec& output : op.outputs()) {
    writer.WriteField(output.var(), PRIVATE);
  }
  RenderConstructor(op, op_class, &writer);
  writer.EndType();
}

}  // namespace

OpGenerator::OpGenerator(const string& base_package, const string& output_dir,
    const std::vector<string>& api_dirs, Env* env)
  : base_package_(base_package), output_dir_(output_dir), api_dirs_(api_dirs),
    env_(env) {
}

Status OpGenerator::Run(const OpList& op_list) {
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
      OpSpec op(OpSpec::Create(op_def, *api_def));
      for (const EndpointSpec& endpoint : op.endpoints()) {
        GenerateOp(op, endpoint, base_package_, output_dir_, env_);
      }
    }
  }
  return Status::OK();
}

}  // namespace java
}  // namespace tensorflow
