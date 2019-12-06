//===- OpDefinitionsGen.cpp - MLIR op definitions generator ---------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// OpDefinitionsGen uses the description of operations to generate C++
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-opdefgen"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

static const char *const tblgenNamePrefix = "tblgen_";
static const char *const generatedArgName = "tblgen_arg";
static const char *const builderOpState = "tblgen_state";

// The logic to calculate the actual value range for a declared operand/result
// of an op with variadic operands/results. Note that this logic is not for
// general use; it assumes all variadic operands/results must have the same
// number of values.
//
// {0}: The list of whether each declared operand/result is variadic.
// {1}: The total number of non-variadic operands/results.
// {2}: The total number of variadic operands/results.
// {3}: The total number of actual values.
// {4}: The begin iterator of the actual values.
// {5}: "operand" or "result".
const char *sameVariadicSizeValueRangeCalcCode = R"(
  bool isVariadic[] = {{{0}};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic {5} corresponds to.
  // This assumes all static variadic {5}s have the same dynamic value count.
  int variadicSize = ({3} - {1}) / {2};
  // `index` passed in as the parameter is the static index which counts each
  // {5} (variadic or not) as size 1. So here for each previous static variadic
  // {5}, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static {5} starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {{std::next({4}, offset), std::next({4}, offset + size)};
)";

// The logic to calculate the actual value range for a declared operand/result
// of an op with variadic operands/results. Note that this logic is assumes
// the op has an attribute specifying the size of each operand/result segment
// (variadic or not).
//
// {0}: The name of the attribute specifying the segment sizes.
// {1}: The begin iterator of the actual values.
const char *attrSizedSegmentValueRangeCalcCode = R"(
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("{0}");
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned end = start + (*(sizeAttr.begin() + index)).getZExtValue();
  return {{std::next({1}, start), std::next({1}, end)};
)";

static const char *const opCommentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

//===----------------------------------------------------------------------===//
// Utility structs and functions
//===----------------------------------------------------------------------===//

// Returns whether the record has a value of the given name that can be returned
// via getValueAsString.
static inline bool hasStringAttribute(const Record &record,
                                      StringRef fieldName) {
  auto valueInit = record.getValueInit(fieldName);
  return isa<CodeInit>(valueInit) || isa<StringInit>(valueInit);
}

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (!operand.name.empty())
    return operand.name;
  else
    return formatv("{0}_{1}", generatedArgName, index);
}

// Returns true if we can use unwrapped value for the given `attr` in builders.
static bool canUseUnwrappedRawValue(const tblgen::Attribute &attr) {
  return attr.getReturnType() != attr.getStorageType() &&
         // We need to wrap the raw value into an attribute in the builder impl
         // so we need to make sure that the attribute specifies how to do that.
         !attr.getConstBuilderTemplate().empty();
}

namespace {
// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(StringRef name, raw_ostream &os) : name(name), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n\n";
  }

  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  StringRef name;
  raw_ostream &os;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Classes for C++ code emission
//===----------------------------------------------------------------------===//

// We emit the op declaration and definition into separate files: *Ops.h.inc
// and *Ops.cpp.inc. The former is to be included in the dialect *Ops.h and
// the latter for dialect *Ops.cpp. This way provides a cleaner interface.
//
// In order to do this split, we need to track method signature and
// implementation logic separately. Signature information is used for both
// declaration and definition, while implementation logic is only for
// definition. So we have the following classes for C++ code emission.

namespace {
// Class for holding the signature of an op's method for C++ code emission
class OpMethodSignature {
public:
  OpMethodSignature(StringRef retType, StringRef name, StringRef params);

  // Writes the signature as a method declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the signature as the start of a method definition to the given `os`.
  // `namePrefix` is the prefix to be prepended to the method name (typically
  // namespaces for qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  // Returns true if the given C++ `type` ends with '&' or '*', or is empty.
  static bool elideSpaceAfterType(StringRef type);

  std::string returnType;
  std::string methodName;
  std::string parameters;
};

// Class for holding the body of an op's method for C++ code emission
class OpMethodBody {
public:
  explicit OpMethodBody(bool declOnly);

  OpMethodBody &operator<<(Twine content);
  OpMethodBody &operator<<(int content);
  OpMethodBody &operator<<(const FmtObjectBase &content);

  void writeTo(raw_ostream &os) const;

private:
  // Whether this class should record method body.
  bool isEffective;
  std::string body;
};

// Class for holding an op's method for C++ code emission
class OpMethod {
public:
  // Properties (qualifiers) of class methods. Bitfield is used here to help
  // querying properties.
  enum Property {
    MP_None = 0x0,
    MP_Static = 0x1,      // Static method
    MP_Constructor = 0x2, // Constructor
    MP_Private = 0x4,     // Private method
  };

  OpMethod(StringRef retType, StringRef name, StringRef params,
           Property property, bool declOnly);

  OpMethodBody &body();

  // Returns true if this is a static method.
  bool isStatic() const;

  // Returns true if this is a private method.
  bool isPrivate() const;

  // Writes the method as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the method as a definition to the given `os`. `namePrefix` is the
  // prefix to be prepended to the method name (typically namespaces for
  // qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  Property properties;
  // Whether this method only contains a declaration.
  bool isDeclOnly;
  OpMethodSignature methodSignature;
  OpMethodBody methodBody;
};

// A class used to emit C++ classes from Tablegen.  Contains a list of public
// methods and a list of private fields to be emitted.
class Class {
public:
  explicit Class(StringRef name);

  // Creates a new method in this class.
  OpMethod &newMethod(StringRef retType, StringRef name, StringRef params = "",
                      OpMethod::Property = OpMethod::MP_None,
                      bool declOnly = false);

  OpMethod &newConstructor(StringRef params = "", bool declOnly = false);

  // Creates a new field in this class.
  void newField(StringRef type, StringRef name, StringRef defaultValue = "");

  // Writes this op's class as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the method definitions in this op's class to the given `os`.
  void writeDefTo(raw_ostream &os) const;

  // Returns the C++ class name of the op.
  StringRef getClassName() const { return className; }

protected:
  std::string className;
  SmallVector<OpMethod, 8> methods;
  SmallVector<std::string, 4> fields;
};

// Class for holding an op for C++ code emission
class OpClass : public Class {
public:
  explicit OpClass(StringRef name, StringRef extraClassDeclaration = "");

  // Sets whether this OpClass should generate the using directive for its
  // associate operand adaptor class.
  void setHasOperandAdaptorClass(bool has);

  // Adds an op trait.
  void addTrait(Twine trait);

  // Writes this op's class as a declaration to the given `os`.  Redefines
  // Class::writeDeclTo to also emit traits and extra class declarations.
  void writeDeclTo(raw_ostream &os) const;

private:
  StringRef extraClassDeclaration;
  SmallVector<std::string, 4> traits;
  bool hasOperandAdaptor;
};
} // end anonymous namespace

OpMethodSignature::OpMethodSignature(StringRef retType, StringRef name,
                                     StringRef params)
    : returnType(retType), methodName(name), parameters(params) {}

void OpMethodSignature::writeDeclTo(raw_ostream &os) const {
  os << returnType << (elideSpaceAfterType(returnType) ? "" : " ") << methodName
     << "(" << parameters << ")";
}

void OpMethodSignature::writeDefTo(raw_ostream &os,
                                   StringRef namePrefix) const {
  // We need to remove the default values for parameters in method definition.
  // TODO(antiagainst): We are using '=' and ',' as delimiters for parameter
  // initializers. This is incorrect for initializer list with more than one
  // element. Change to a more robust approach.
  auto removeParamDefaultValue = [](StringRef params) {
    std::string result;
    std::pair<StringRef, StringRef> parts;
    while (!params.empty()) {
      parts = params.split("=");
      result.append(result.empty() ? "" : ", ");
      result.append(parts.first);
      params = parts.second.split(",").second;
    }
    return result;
  };

  os << returnType << (elideSpaceAfterType(returnType) ? "" : " ") << namePrefix
     << (namePrefix.empty() ? "" : "::") << methodName << "("
     << removeParamDefaultValue(parameters) << ")";
}

bool OpMethodSignature::elideSpaceAfterType(StringRef type) {
  return type.empty() || type.endswith("&") || type.endswith("*");
}

OpMethodBody::OpMethodBody(bool declOnly) : isEffective(!declOnly) {}

OpMethodBody &OpMethodBody::operator<<(Twine content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

OpMethodBody &OpMethodBody::operator<<(int content) {
  if (isEffective)
    body.append(std::to_string(content));
  return *this;
}

OpMethodBody &OpMethodBody::operator<<(const FmtObjectBase &content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

void OpMethodBody::writeTo(raw_ostream &os) const {
  auto bodyRef = StringRef(body).drop_while([](char c) { return c == '\n'; });
  os << bodyRef;
  if (bodyRef.empty() || bodyRef.back() != '\n')
    os << "\n";
}

OpMethod::OpMethod(StringRef retType, StringRef name, StringRef params,
                   OpMethod::Property property, bool declOnly)
    : properties(property), isDeclOnly(declOnly),
      methodSignature(retType, name, params), methodBody(declOnly) {}

OpMethodBody &OpMethod::body() { return methodBody; }

bool OpMethod::isStatic() const { return properties & MP_Static; }

bool OpMethod::isPrivate() const { return properties & MP_Private; }

void OpMethod::writeDeclTo(raw_ostream &os) const {
  os.indent(2);
  if (isStatic())
    os << "static ";
  methodSignature.writeDeclTo(os);
  os << ";";
}

void OpMethod::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  if (isDeclOnly)
    return;

  methodSignature.writeDefTo(os, namePrefix);
  os << " {\n";
  methodBody.writeTo(os);
  os << "}";
}

Class::Class(StringRef name) : className(name) {}

OpMethod &Class::newMethod(StringRef retType, StringRef name, StringRef params,
                           OpMethod::Property property, bool declOnly) {
  methods.emplace_back(retType, name, params, property, declOnly);
  return methods.back();
}

OpMethod &Class::newConstructor(StringRef params, bool declOnly) {
  return newMethod("", getClassName(), params, OpMethod::MP_Constructor,
                   declOnly);
}

void Class::newField(StringRef type, StringRef name, StringRef defaultValue) {
  std::string varName = formatv("{0} {1}", type, name).str();
  std::string field = defaultValue.empty()
                          ? varName
                          : formatv("{0} = {1}", varName, defaultValue).str();
  fields.push_back(std::move(field));
}

void Class::writeDeclTo(raw_ostream &os) const {
  bool hasPrivateMethod = false;
  os << "class " << className << " {\n";
  os << "public:\n";
  for (const auto &method : methods) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << '\n';
    } else {
      hasPrivateMethod = true;
    }
  }
  os << '\n';
  os << "private:\n";
  if (hasPrivateMethod) {
    for (const auto &method : methods) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << '\n';
      }
    }
    os << '\n';
  }
  for (const auto &field : fields)
    os.indent(2) << field << ";\n";
  os << "};\n";
}

void Class::writeDefTo(raw_ostream &os) const {
  for (const auto &method : methods) {
    method.writeDefTo(os, className);
    os << "\n\n";
  }
}

OpClass::OpClass(StringRef name, StringRef extraClassDeclaration)
    : Class(name), extraClassDeclaration(extraClassDeclaration),
      hasOperandAdaptor(true) {}

void OpClass::setHasOperandAdaptorClass(bool has) { hasOperandAdaptor = has; }

// Adds the given trait to this op.
void OpClass::addTrait(Twine trait) { traits.push_back(trait.str()); }

void OpClass::writeDeclTo(raw_ostream &os) const {
  os << "class " << className << " : public Op<" << className;
  for (const auto &trait : traits)
    os << ", " << trait;
  os << "> {\npublic:\n";
  os << "  using Op::Op;\n";
  if (hasOperandAdaptor)
    os << "  using OperandAdaptor = " << className << "OperandAdaptor;\n";

  bool hasPrivateMethod = false;
  for (const auto &method : methods) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << "\n";
    } else {
      hasPrivateMethod = true;
    }
  }

  // TODO: Add line control markers to make errors easier to debug.
  if (!extraClassDeclaration.empty())
    os << extraClassDeclaration << "\n";

  if (hasPrivateMethod) {
    os << '\n';
    os << "private:\n";
    for (const auto &method : methods) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << "\n";
      }
    }
  }

  os << "};\n";
}

//===----------------------------------------------------------------------===//
// Op emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit a record into the given output stream.
class OpEmitter {
public:
  static void emitDecl(const Operator &op, raw_ostream &os);
  static void emitDef(const Operator &op, raw_ostream &os);

private:
  OpEmitter(const Operator &op);

  void emitDecl(raw_ostream &os);
  void emitDef(raw_ostream &os);

  // Generates the OpAsmOpInterface for this operation if possible.
  void genOpAsmInterface();

  // Generates the `getOperationName` method for this op.
  void genOpNameGetter();

  // Generates getters for the attributes.
  void genAttrGetters();

  // Generates getters for named operands.
  void genNamedOperandGetters();

  // Generates getters for named results.
  void genNamedResultGetters();

  // Generates getters for named regions.
  void genNamedRegionGetters();

  // Generates builder methods for the operation.
  void genBuilder();

  // Generates the build() method that takes each result-type/operand/attribute
  // as a stand-alone parameter. Attributes will take wrapped mlir::Attribute
  // values. The generated build() method also requires specifying result types
  // for all results.
  void genSeparateParamWrappedAttrBuilder();

  // Generates the build() method that takes each result-type/operand/attribute
  // as a stand-alone parameter. Attributes will take raw values without
  // mlir::Attribute wrapper. The generated build() method also requires
  // specifying result types for all results.
  void genSeparateParamUnwrappedAttrBuilder();

  // Generates the build() method that takes a single parameter for all the
  // result types and a separate parameter for each operand/attribute.
  void genCollectiveTypeParamBuilder();

  // Generates the build() method that takes each operand/attribute as a
  // stand-alone parameter. The generated build() method uses first operand's
  // type as all results' types.
  void genUseOperandAsResultTypeSeparateParamBuilder();

  // Generates the build() method that takes all operands/attributes
  // collectively as one parameter. The generated build() method uses first
  // operand's type as all results' types.
  void genUseOperandAsResultTypeCollectiveParamBuilder();

  // Generates the build() method that takes each operand/attribute as a
  // stand-alone parameter. The generated build() method uses first attribute's
  // type as all result's types.
  void genUseAttrAsResultTypeBuilder();

  // Generates the build() method that takes all result types collectively as
  // one parameter. Similarly for operands and attributes.
  void genCollectiveParamBuilder();

  // The kind of parameter to generate for result types in builders.
  enum class TypeParamKind {
    None,       // No result type in parameter list.
    Separate,   // A separate parameter for each result type.
    Collective, // An ArrayRef<Type> for all result types.
  };

  // The kind of parameter to generate for attributes in builders.
  enum class AttrParamKind {
    WrappedAttr,    // A wrapped MLIR Attribute instance.
    UnwrappedValue, // A raw value without MLIR Attribute wrapper.
  };

  // Builds the parameter list for build() method of this op. This method writes
  // to `paramList` the comma-separated parameter list and updates
  // `resultTypeNames` with the names for parameters for specifying result
  // types. The given `typeParamKind` and `attrParamKind` controls how result
  // types and attributes are placed in the parameter list.
  void buildParamList(std::string &paramList,
                      SmallVectorImpl<std::string> &resultTypeNames,
                      TypeParamKind typeParamKind,
                      AttrParamKind attrParamKind = AttrParamKind::WrappedAttr);

  // Adds op arguments and regions into operation state for build() methods.
  void genCodeForAddingArgAndRegionForBuilder(OpMethodBody &body,
                                              bool isRawValueAttr = false);

  // Generates canonicalizer declaration for the operation.
  void genCanonicalizerDecls();

  // Generates the folder declaration for the operation.
  void genFolderDecls();

  // Generates the parser for the operation.
  void genParser();

  // Generates the printer for the operation.
  void genPrinter();

  // Generates verify method for the operation.
  void genVerifier();

  // Generates verify statements for operands and results in the operation.
  // The generated code will be attached to `body`.
  void genOperandResultVerifier(OpMethodBody &body,
                                Operator::value_range values,
                                StringRef valueKind);

  // Generates verify statements for regions in the operation.
  // The generated code will be attached to `body`.
  void genRegionVerifier(OpMethodBody &body);

  // Generates the traits used by the object.
  void genTraits();

  // Generate the OpInterface methods.
  void genOpInterfaceMethods();

private:
  // The TableGen record for this op.
  // TODO(antiagainst,zinenko): OpEmitter should not have a Record directly,
  // it should rather go through the Operator for better abstraction.
  const Record &def;

  // The wrapper operator class for querying information from this op.
  Operator op;

  // The C++ code builder for this op
  OpClass opClass;

  // The format context for verification code generation.
  FmtContext verifyCtx;
};
} // end anonymous namespace

OpEmitter::OpEmitter(const Operator &op)
    : def(op.getDef()), op(op),
      opClass(op.getCppClassName(), op.getExtraClassDeclaration()) {
  verifyCtx.withOp("(*this->getOperation())");

  genTraits();
  // Generate C++ code for various op methods. The order here determines the
  // methods in the generated file.
  genOpAsmInterface();
  genOpNameGetter();
  genNamedOperandGetters();
  genNamedResultGetters();
  genNamedRegionGetters();
  genAttrGetters();
  genBuilder();
  genParser();
  genPrinter();
  genVerifier();
  genCanonicalizerDecls();
  genFolderDecls();
  genOpInterfaceMethods();
}

void OpEmitter::emitDecl(const Operator &op, raw_ostream &os) {
  OpEmitter(op).emitDecl(os);
}

void OpEmitter::emitDef(const Operator &op, raw_ostream &os) {
  OpEmitter(op).emitDef(os);
}

void OpEmitter::emitDecl(raw_ostream &os) { opClass.writeDeclTo(os); }

void OpEmitter::emitDef(raw_ostream &os) { opClass.writeDefTo(os); }

void OpEmitter::genAttrGetters() {
  FmtContext fctx;
  fctx.withBuilder("mlir::Builder(this->getContext())");
  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;

    auto &method = opClass.newMethod(attr.getReturnType(), name);
    auto &body = method.body();

    // Emit the derived attribute body.
    if (attr.isDerivedAttr()) {
      body << "  " << attr.getDerivedCodeBody() << "\n";
      continue;
    }

    // Emit normal emitter.

    // Return the queried attribute with the correct return type.
    auto attrVal =
        (attr.hasDefaultValue() || attr.isOptional())
            ? formatv("this->getAttr(\"{0}\").dyn_cast_or_null<{1}>()", name,
                      attr.getStorageType())
            : formatv("this->getAttr(\"{0}\").cast<{1}>()", name,
                      attr.getStorageType());
    body << "  auto attr = " << attrVal << ";\n";
    if (attr.hasDefaultValue()) {
      // Returns the default value if not set.
      // TODO: this is inefficient, we are recreating the attribute for every
      // call. This should be set instead.
      std::string defaultValue =
          tgfmt(attr.getConstBuilderTemplate(), &fctx, attr.getDefaultValue());
      body << "    if (!attr)\n      return "
           << tgfmt(attr.getConvertFromStorageCall(),
                    &fctx.withSelf(defaultValue))
           << ";\n";
    }
    body << "  return "
         << tgfmt(attr.getConvertFromStorageCall(), &fctx.withSelf("attr"))
         << ";\n";
  }
}

// Generates the named operand getter methods for the given Operator `op` and
// puts them in `opClass`.  Uses `rangeType` as the return type of getters that
// return a range of operands (individual operands are `Value *` and each
// element in the range must also be `Value *`); use `rangeBeginCall` to get an
// iterator to the beginning of the operand range; use `rangeSizeCall` to obtain
// the number of operands. `getOperandCallPattern` contains the code necessary
// to obtain a single operand whose position will be substituted instead of
// "{0}" marker in the pattern.  Note that the pattern should work for any kind
// of ops, in particular for one-operand ops that may not have the
// `getOperand(unsigned)` method.
static void generateNamedOperandGetters(const Operator &op, Class &opClass,
                                        StringRef rangeType,
                                        StringRef rangeBeginCall,
                                        StringRef rangeSizeCall,
                                        StringRef getOperandCallPattern) {
  const int numOperands = op.getNumOperands();
  const int numVariadicOperands = op.getNumVariadicOperands();
  const int numNormalOperands = numOperands - numVariadicOperands;

  const auto *sameVariadicSize =
      op.getTrait("OpTrait::SameVariadicOperandSize");
  const auto *attrSizedOperands =
      op.getTrait("OpTrait::AttrSizedOperandSegments");

  if (numVariadicOperands > 1 && !sameVariadicSize && !attrSizedOperands) {
    PrintFatalError(op.getLoc(), "op has multiple variadic operands but no "
                                 "specification over their sizes");
  }

  if (numVariadicOperands < 2 && attrSizedOperands) {
    PrintFatalError(op.getLoc(), "op must have at least two variadic operands "
                                 "to use 'AttrSizedOperandSegments' trait");
  }

  if (attrSizedOperands && sameVariadicSize) {
    PrintFatalError(op.getLoc(),
                    "op cannot have both 'AttrSizedOperandSegments' and "
                    "'SameVariadicOperandSize' traits");
  }

  // First emit a "sink" getter method upon which we layer all nicer named
  // getter methods.
  auto &m = opClass.newMethod(rangeType, "getODSOperands", "unsigned index");

  if (numVariadicOperands == 0) {
    // We still need to match the return type, which is a range.
    m.body() << "  return {std::next(" << rangeBeginCall
             << ", index), std::next(" << rangeBeginCall << ", index + 1)};";
  } else if (attrSizedOperands) {
    m.body() << formatv(attrSizedSegmentValueRangeCalcCode,
                        "operand_segment_sizes", rangeBeginCall);
  } else {
    // Because the op can have arbitrarily interleaved variadic and non-variadic
    // operands, we need to embed a list in the "sink" getter method for
    // calculation at run-time.
    llvm::SmallVector<StringRef, 4> isVariadic;
    isVariadic.reserve(numOperands);
    for (int i = 0; i < numOperands; ++i) {
      isVariadic.push_back(llvm::toStringRef(op.getOperand(i).isVariadic()));
    }
    std::string isVariadicList = llvm::join(isVariadic, ", ");

    m.body() << formatv(sameVariadicSizeValueRangeCalcCode, isVariadicList,
                        numNormalOperands, numVariadicOperands, rangeSizeCall,
                        rangeBeginCall, "operand");
  }

  // Then we emit nicer named getter methods by redirecting to the "sink" getter
  // method.

  for (int i = 0; i != numOperands; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;

    if (operand.isVariadic()) {
      auto &m = opClass.newMethod(rangeType, operand.name);
      m.body() << "  return getODSOperands(" << i << ");";
    } else {
      auto &m = opClass.newMethod("Value *", operand.name);
      m.body() << "  return *getODSOperands(" << i << ").begin();";
    }
  }
}

void OpEmitter::genNamedOperandGetters() {
  if (op.getTrait("OpTrait::AttrSizedOperandSegments"))
    opClass.setHasOperandAdaptorClass(false);

  generateNamedOperandGetters(
      op, opClass, /*rangeType=*/"Operation::operand_range",
      /*rangeBeginCall=*/"getOperation()->operand_begin()",
      /*rangeSizeCall=*/"getOperation()->getNumOperands()",
      /*getOperandCallPattern=*/"getOperation()->getOperand({0})");
}

void OpEmitter::genNamedResultGetters() {
  const int numResults = op.getNumResults();
  const int numVariadicResults = op.getNumVariadicResults();
  const int numNormalResults = numResults - numVariadicResults;

  // If we have more than one variadic results, we need more complicated logic
  // to calculate the value range for each result.

  const auto *sameVariadicSize = op.getTrait("OpTrait::SameVariadicResultSize");
  const auto *attrSizedResults =
      op.getTrait("OpTrait::AttrSizedResultSegments");

  if (numVariadicResults > 1 && !sameVariadicSize && !attrSizedResults) {
    PrintFatalError(op.getLoc(), "op has multiple variadic results but no "
                                 "specification over their sizes");
  }

  if (numVariadicResults < 2 && attrSizedResults) {
    PrintFatalError(op.getLoc(), "op must have at least two variadic results "
                                 "to use 'AttrSizedResultSegments' trait");
  }

  if (attrSizedResults && sameVariadicSize) {
    PrintFatalError(op.getLoc(),
                    "op cannot have both 'AttrSizedResultSegments' and "
                    "'SameVariadicResultSize' traits");
  }

  auto &m = opClass.newMethod("Operation::result_range", "getODSResults",
                              "unsigned index");

  if (numVariadicResults == 0) {
    m.body() << "  return {std::next(getOperation()->result_begin(), index), "
                "std::next(getOperation()->result_begin(), index + 1)};";
  } else if (attrSizedResults) {
    m.body() << formatv(attrSizedSegmentValueRangeCalcCode,
                        "result_segment_sizes",
                        "getOperation()->result_begin()");
  } else {
    llvm::SmallVector<StringRef, 4> isVariadic;
    isVariadic.reserve(numResults);
    for (int i = 0; i < numResults; ++i) {
      isVariadic.push_back(llvm::toStringRef(op.getResult(i).isVariadic()));
    }
    std::string isVariadicList = llvm::join(isVariadic, ", ");

    m.body() << formatv(sameVariadicSizeValueRangeCalcCode, isVariadicList,
                        numNormalResults, numVariadicResults,
                        "getOperation()->getNumResults()",
                        "getOperation()->result_begin()", "result");
  }

  for (int i = 0; i != numResults; ++i) {
    const auto &result = op.getResult(i);
    if (result.name.empty())
      continue;

    if (result.isVariadic()) {
      auto &m = opClass.newMethod("Operation::result_range", result.name);
      m.body() << "  return getODSResults(" << i << ");";
    } else {
      auto &m = opClass.newMethod("Value *", result.name);
      m.body() << "  return *getODSResults(" << i << ").begin();";
    }
  }
}

void OpEmitter::genNamedRegionGetters() {
  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (!region.name.empty()) {
      auto &m = opClass.newMethod("Region &", region.name);
      m.body() << formatv("  return this->getOperation()->getRegion({0});", i);
    }
  }
}

void OpEmitter::genSeparateParamWrappedAttrBuilder() {
  std::string paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  buildParamList(paramList, resultNames, TypeParamKind::Separate);

  auto &m = opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
  genCodeForAddingArgAndRegionForBuilder(m.body());

  // Push all result types to the operation state
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    m.body() << "  " << builderOpState << ".addTypes(" << resultNames[i]
             << ");\n";
  }
}

void OpEmitter::genSeparateParamUnwrappedAttrBuilder() {
  // If this op does not have native attributes at all, return directly to avoid
  // redefining builders.
  if (op.getNumNativeAttributes() == 0)
    return;

  bool canGenerate = false;
  // We are generating builders that take raw values for attributes. We need to
  // make sure the native attributes have a meaningful "unwrapped" value type
  // different from the wrapped mlir::Attribute type to avoid redefining
  // builders. This checks for the op has at least one such native attribute.
  for (int i = 0, e = op.getNumNativeAttributes(); i < e; ++i) {
    NamedAttribute &namedAttr = op.getAttribute(i);
    if (canUseUnwrappedRawValue(namedAttr.attr)) {
      canGenerate = true;
      break;
    }
  }
  if (!canGenerate)
    return;

  std::string paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  buildParamList(paramList, resultNames, TypeParamKind::Separate,
                 AttrParamKind::UnwrappedValue);

  auto &m = opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
  genCodeForAddingArgAndRegionForBuilder(m.body(), /*isRawValueAttr=*/true);

  // Push all result types to the operation state.
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    m.body() << "  " << builderOpState << ".addTypes(" << resultNames[i]
             << ");\n";
  }
}

void OpEmitter::genCollectiveTypeParamBuilder() {
  auto numResults = op.getNumResults();

  // If this op has no results, then just skip generating this builder.
  // Otherwise we are generating the same signature as the separate-parameter
  // builder.
  if (numResults == 0)
    return;

  // Similarly for ops with one single variadic result, which will also have one
  // `ArrayRef<Type>` parameter for the result type.
  if (numResults == 1 && op.getResult(0).isVariadic())
    return;

  std::string paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  buildParamList(paramList, resultNames, TypeParamKind::Collective);

  auto &m = opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
  genCodeForAddingArgAndRegionForBuilder(m.body());

  // Push all result types to the operation state
  m.body() << formatv("  {0}.addTypes(resultTypes);\n", builderOpState);
}

void OpEmitter::genUseOperandAsResultTypeCollectiveParamBuilder() {
  // If this op has a variadic result, we cannot generate this builder because
  // we don't know how many results to create.
  if (op.getNumVariadicResults() != 0)
    return;

  int numResults = op.getNumResults();

  // Signature
  std::string params =
      std::string("Builder *, OperationState &") + builderOpState +
      ", ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

  // Result types
  SmallVector<std::string, 2> resultTypes(numResults, "operands[0]->getType()");
  body << "  " << builderOpState << ".addTypes({"
       << llvm::join(resultTypes, ", ") << "});\n\n";

  // Operands
  body << "  " << builderOpState << ".addOperands(operands);\n\n";

  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    for (int i = 0; i < numRegions; ++i)
      m.body() << "  (void)" << builderOpState << ".addRegion();\n";
  }
}

void OpEmitter::genUseOperandAsResultTypeSeparateParamBuilder() {
  std::string paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  buildParamList(paramList, resultNames, TypeParamKind::None);

  auto &m = opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
  genCodeForAddingArgAndRegionForBuilder(m.body());

  auto numResults = op.getNumResults();
  if (numResults == 0)
    return;

  // Push all result types to the operation state
  const char *index = op.getOperand(0).isVariadic() ? ".front()" : "";
  std::string resultType =
      formatv("{0}{1}->getType()", getArgumentName(op, 0), index).str();
  m.body() << "  " << builderOpState << ".addTypes({" << resultType;
  for (int i = 1; i != numResults; ++i)
    m.body() << ", " << resultType;
  m.body() << "});\n\n";
}

void OpEmitter::genUseAttrAsResultTypeBuilder() {
  std::string params =
      std::string("Builder *, OperationState &") + builderOpState +
      ", ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

  // Push all result types to the operation state
  std::string resultType;
  const auto &namedAttr = op.getAttribute(0);

  body << "  for (auto attr : attributes) {\n";
  body << "    if (attr.first != \"" << namedAttr.name << "\") continue;\n";
  if (namedAttr.attr.isTypeAttr()) {
    resultType = "attr.second.cast<TypeAttr>().getValue()";
  } else {
    resultType = "attr.second.getType()";
  }
  SmallVector<std::string, 2> resultTypes(op.getNumResults(), resultType);
  body << "    " << builderOpState << ".addTypes({"
       << llvm::join(resultTypes, ", ") << "});\n";
  body << "  }\n";

  // Operands
  body << "  " << builderOpState << ".addOperands(operands);\n\n";
  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";
}

void OpEmitter::genBuilder() {
  // Handle custom builders if provided.
  // TODO(antiagainst): Create wrapper class for OpBuilder to hide the native
  // TableGen API calls here.
  {
    auto *listInit = dyn_cast_or_null<ListInit>(def.getValueInit("builders"));
    if (listInit) {
      for (Init *init : listInit->getValues()) {
        Record *builderDef = cast<DefInit>(init)->getDef();
        StringRef params = builderDef->getValueAsString("params");
        StringRef body = builderDef->getValueAsString("body");
        bool hasBody = !body.empty();

        auto &method =
            opClass.newMethod("void", "build", params, OpMethod::MP_Static,
                              /*declOnly=*/!hasBody);
        if (hasBody)
          method.body() << body;
      }
    }
    if (op.skipDefaultBuilders()) {
      if (!listInit || listInit->empty())
        PrintFatalError(
            op.getLoc(),
            "default builders are skipped and no custom builders provided");
      return;
    }
  }

  // Generate default builders that requires all result type, operands, and
  // attributes as parameters.

  // We generate three builders here:
  // 1. one having a stand-alone parameter for each result type / operand /
  //    attribute, and
  genSeparateParamWrappedAttrBuilder();
  genSeparateParamUnwrappedAttrBuilder();
  // 2. one having a stand-alone parameter for each operand / attribute and
  //    an aggregated parameter for all result types, and
  genCollectiveTypeParamBuilder();
  // 3. one having an aggregated parameter for all result types / operands /
  //    attributes, and
  genCollectiveParamBuilder();
  // 4. one having a stand-alone parameter for each operand and attribute,
  //    use the first operand or attribute's type as all result types
  // to facilitate different call patterns.
  if (op.getNumVariadicResults() == 0) {
    if (op.getTrait("OpTrait::SameOperandsAndResultType")) {
      genUseOperandAsResultTypeSeparateParamBuilder();
      genUseOperandAsResultTypeCollectiveParamBuilder();
    }
    if (op.getTrait("OpTrait::FirstAttrDerivedResultType"))
      genUseAttrAsResultTypeBuilder();
  }
}

void OpEmitter::genCollectiveParamBuilder() {
  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariadicResults();
  int numNonVariadicResults = numResults - numVariadicResults;

  int numOperands = op.getNumOperands();
  int numVariadicOperands = op.getNumVariadicOperands();
  int numNonVariadicOperands = numOperands - numVariadicOperands;
  // Signature
  std::string params =
      std::string("Builder *, OperationState &") + builderOpState +
      ", ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, "
      "ArrayRef<NamedAttribute> attributes";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

  // Result types
  if (numVariadicResults == 0 || numNonVariadicResults != 0)
    body << "  assert(resultTypes.size()"
         << (numVariadicResults != 0 ? " >= " : " == ") << numNonVariadicResults
         << "u && \"mismatched number of return types\");\n";
  body << "  " << builderOpState << ".addTypes(resultTypes);\n";

  // Operands
  if (numVariadicOperands == 0 || numNonVariadicOperands != 0)
    body << "  assert(operands.size()"
         << (numVariadicOperands != 0 ? " >= " : " == ")
         << numNonVariadicOperands
         << "u && \"mismatched number of parameters\");\n";
  body << "  " << builderOpState << ".addOperands(operands);\n\n";

  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    for (int i = 0; i < numRegions; ++i)
      m.body() << "  (void)" << builderOpState << ".addRegion();\n";
  }
}

void OpEmitter::buildParamList(std::string &paramList,
                               SmallVectorImpl<std::string> &resultTypeNames,
                               TypeParamKind typeParamKind,
                               AttrParamKind attrParamKind) {
  resultTypeNames.clear();
  auto numResults = op.getNumResults();
  resultTypeNames.reserve(numResults);

  paramList = "Builder *tblgen_builder, OperationState &";
  paramList.append(builderOpState);

  switch (typeParamKind) {
  case TypeParamKind::None:
    break;
  case TypeParamKind::Separate: {
    // Add parameters for all return types
    for (int i = 0; i < numResults; ++i) {
      const auto &result = op.getResult(i);
      std::string resultName = result.name;
      if (resultName.empty())
        resultName = formatv("resultType{0}", i);

      paramList.append(result.isVariadic() ? ", ArrayRef<Type> " : ", Type ");
      paramList.append(resultName);

      resultTypeNames.emplace_back(std::move(resultName));
    }
  } break;
  case TypeParamKind::Collective: {
    paramList.append(", ArrayRef<Type> resultTypes");
    resultTypeNames.push_back("resultTypes");
  } break;
  }

  // Add parameters for all arguments (operands and attributes).

  int numOperands = 0;
  int numAttrs = 0;

  int defaultValuedAttrStartIndex = op.getNumArgs();
  if (attrParamKind == AttrParamKind::UnwrappedValue) {
    // Calculate the start index from which we can attach default values in the
    // builder declaration.
    for (int i = op.getNumArgs() - 1; i >= 0; --i) {
      auto *namedAttr = op.getArg(i).dyn_cast<tblgen::NamedAttribute *>();
      if (!namedAttr || !namedAttr->attr.hasDefaultValue())
        break;

      if (!canUseUnwrappedRawValue(namedAttr->attr))
        break;

      // Creating an APInt requires us to provide bitwidth, value, and
      // signedness, which is complicated compared to others. Similarly
      // for APFloat.
      // TODO(b/144412160) Adjust the 'returnType' field of such attributes
      // to support them.
      StringRef retType = namedAttr->attr.getReturnType();
      if (retType == "APInt" || retType == "APFloat")
        break;

      defaultValuedAttrStartIndex = i;
    }
  }

  for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (argument.is<tblgen::NamedTypeConstraint *>()) {
      const auto &operand = op.getOperand(numOperands);
      paramList.append(operand.isVariadic() ? ", ArrayRef<Value *> "
                                            : ", Value *");
      paramList.append(getArgumentName(op, numOperands));
      ++numOperands;
    } else {
      const auto &namedAttr = op.getAttribute(numAttrs);
      const auto &attr = namedAttr.attr;
      paramList.append(", ");

      if (attr.isOptional())
        paramList.append("/*optional*/");

      switch (attrParamKind) {
      case AttrParamKind::WrappedAttr:
        paramList.append(attr.getStorageType());
        break;
      case AttrParamKind::UnwrappedValue:
        if (canUseUnwrappedRawValue(attr)) {
          paramList.append(attr.getReturnType());
        } else {
          paramList.append(attr.getStorageType());
        }
        break;
      }
      paramList.append(" ");
      paramList.append(namedAttr.name);

      // Attach default value if requested and possible.
      if (attrParamKind == AttrParamKind::UnwrappedValue &&
          i >= defaultValuedAttrStartIndex) {
        bool isString = attr.getReturnType() == "StringRef";
        paramList.append(" = ");
        if (isString)
          paramList.append("\"");
        paramList.append(attr.getDefaultValue());
        if (isString)
          paramList.append("\"");
      }
      ++numAttrs;
    }
  }
}

void OpEmitter::genCodeForAddingArgAndRegionForBuilder(OpMethodBody &body,
                                                       bool isRawValueAttr) {
  // Push all operands to the result
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    body << "  " << builderOpState << ".addOperands(" << getArgumentName(op, i)
         << ");\n";
  }

  // Push all attributes to the result
  for (const auto &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (!attr.isDerivedAttr()) {
      bool emitNotNullCheck = attr.isOptional();
      if (emitNotNullCheck) {
        body << formatv("  if ({0}) ", namedAttr.name) << "{\n";
      }
      if (isRawValueAttr && canUseUnwrappedRawValue(attr)) {
        // If this is a raw value, then we need to wrap it in an Attribute
        // instance.
        FmtContext fctx;
        fctx.withBuilder("(*tblgen_builder)");
        std::string value =
            tgfmt(attr.getConstBuilderTemplate(), &fctx, namedAttr.name);
        body << formatv("  {0}.addAttribute(\"{1}\", {2});\n", builderOpState,
                        namedAttr.name, value);
      } else {
        body << formatv("  {0}.addAttribute(\"{1}\", {1});\n", builderOpState,
                        namedAttr.name);
      }
      if (emitNotNullCheck) {
        body << "  }\n";
      }
    }
  }

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    for (int i = 0; i < numRegions; ++i)
      body << "  (void)" << builderOpState << ".addRegion();\n";
  }
}

void OpEmitter::genCanonicalizerDecls() {
  if (!def.getValueAsBit("hasCanonicalizer"))
    return;

  const char *const params =
      "OwningRewritePatternList &results, MLIRContext *context";
  opClass.newMethod("void", "getCanonicalizationPatterns", params,
                    OpMethod::MP_Static, /*declOnly=*/true);
}

void OpEmitter::genFolderDecls() {
  bool hasSingleResult =
      op.getNumResults() == 1 && op.getNumVariadicResults() == 0;

  if (def.getValueAsBit("hasFolder")) {
    if (hasSingleResult) {
      const char *const params = "ArrayRef<Attribute> operands";
      opClass.newMethod("OpFoldResult", "fold", params, OpMethod::MP_None,
                        /*declOnly=*/true);
    } else {
      const char *const params = "ArrayRef<Attribute> operands, "
                                 "SmallVectorImpl<OpFoldResult> &results";
      opClass.newMethod("LogicalResult", "fold", params, OpMethod::MP_None,
                        /*declOnly=*/true);
    }
  }
}

void OpEmitter::genOpInterfaceMethods() {
  for (const auto &trait : op.getTraits()) {
    auto opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait);
    if (!opTrait || !opTrait->shouldDeclareMethods())
      continue;
    auto interface = opTrait->getOpInterface();
    for (auto method : interface.getMethods()) {
      // Don't declare if the method has a body.
      if (method.getBody())
        continue;
      std::string args;
      llvm::raw_string_ostream os(args);
      mlir::interleaveComma(method.getArguments(), os,
                            [&](const OpInterfaceMethod::Argument &arg) {
                              os << arg.type << " " << arg.name;
                            });
      opClass.newMethod(method.getReturnType(), method.getName(), os.str(),
                        method.isStatic() ? OpMethod::MP_Static
                                          : OpMethod::MP_None,
                        /*declOnly=*/true);
    }
  }
}

void OpEmitter::genParser() {
  if (!hasStringAttribute(def, "parser"))
    return;

  auto &method = opClass.newMethod(
      "ParseResult", "parse", "OpAsmParser &parser, OperationState &result",
      OpMethod::MP_Static);
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto parser = def.getValueAsString("parser").ltrim().rtrim(" \t\v\f\r");
  method.body() << "  " << tgfmt(parser, &fctx);
}

void OpEmitter::genPrinter() {
  auto valueInit = def.getValueInit("printer");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto &method = opClass.newMethod("void", "print", "OpAsmPrinter &p");
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto printer = codeInit->getValue().ltrim().rtrim(" \t\v\f\r");
  method.body() << "  " << tgfmt(printer, &fctx);
}

void OpEmitter::genVerifier() {
  auto valueInit = def.getValueInit("verifier");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  bool hasCustomVerify = codeInit && !codeInit->getValue().empty();

  auto &method = opClass.newMethod("LogicalResult", "verify", /*params=*/"");
  auto &body = method.body();

  // Populate substitutions for attributes and named operands and results.
  for (const auto &namedAttr : op.getAttributes())
    verifyCtx.addSubst(namedAttr.name,
                       formatv("this->getAttr(\"{0}\")", namedAttr.name));
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    auto &value = op.getOperand(i);
    // Skip from from first variadic operands for now. Else getOperand index
    // used below doesn't match.
    if (value.isVariadic())
      break;
    if (!value.name.empty())
      verifyCtx.addSubst(
          value.name, formatv("(*this->getOperation()->getOperand({0}))", i));
  }
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    auto &value = op.getResult(i);
    // Skip from from first variadic results for now. Else getResult index used
    // below doesn't match.
    if (value.isVariadic())
      break;
    if (!value.name.empty())
      verifyCtx.addSubst(value.name,
                         formatv("(*this->getOperation()->getResult({0}))", i));
  }

  // Verify the attributes have the correct type.
  for (const auto &namedAttr : op.getAttributes()) {
    const auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr())
      continue;

    auto attrName = namedAttr.name;
    // Prefix with `tblgen_` to avoid hiding the attribute accessor.
    auto varName = tblgenNamePrefix + attrName;
    body << formatv("  auto {0} = this->getAttr(\"{1}\");\n", varName,
                    attrName);

    bool allowMissingAttr = attr.hasDefaultValue() || attr.isOptional();
    if (allowMissingAttr) {
      // If the attribute has a default value, then only verify the predicate if
      // set. This does effectively assume that the default value is valid.
      // TODO: verify the debug value is valid (perhaps in debug mode only).
      body << "  if (" << varName << ") {\n";
    } else {
      body << "  if (!" << varName
           << ") return emitOpError(\"requires attribute '" << attrName
           << "'\");\n  {\n";
    }

    auto attrPred = attr.getPredicate();
    if (!attrPred.isNull()) {
      body << tgfmt(
          "    if (!($0)) return emitOpError(\"attribute '$1' "
          "failed to satisfy constraint: $2\");\n",
          /*ctx=*/nullptr,
          tgfmt(attrPred.getCondition(), &verifyCtx.withSelf(varName)),
          attrName, attr.getDescription());
    }

    body << "  }\n";
  }

  const char *code = R"(
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("{0}");
  auto numElements = sizeAttr.getType().cast<ShapedType>().getNumElements();
  if (numElements != {1}) {{
    return emitOpError("'{0}' attribute for specifiying {2} segments "
                       "must have {1} elements");
  }
  )";

  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::PredOpTrait>(&trait)) {
      body << tgfmt("  if (!($0)) {\n    "
                    "return emitOpError(\"failed to verify that $1\");\n  }\n",
                    &verifyCtx, tgfmt(t->getPredTemplate(), &verifyCtx),
                    t->getDescription());
    } else if (auto *t = dyn_cast<tblgen::NativeOpTrait>(&trait)) {
      if (t->getTrait() == "OpTrait::AttrSizedOperandSegments") {
        body << formatv(code, "operand_segment_sizes", op.getNumOperands(),
                        "operand");
      } else if (t->getTrait() == "OpTrait::AttrSizedResultSegments") {
        body << formatv(code, "result_segment_sizes", op.getNumResults(),
                        "result");
      }
    }
  }

  // These should happen after we verified the traits because
  // getODSOperands()/getODSResults() may depend on traits (e.g.,
  // AttrSizedOperandSegments/AttrSizedResultSegments).
  genOperandResultVerifier(body, op.getOperands(), "operand");
  genOperandResultVerifier(body, op.getResults(), "result");

  genRegionVerifier(body);

  if (hasCustomVerify) {
    FmtContext fctx;
    fctx.addSubst("cppClass", opClass.getClassName());
    auto printer = codeInit->getValue().ltrim().rtrim(" \t\v\f\r");
    body << "  " << tgfmt(printer, &fctx);
  } else {
    body << "  return mlir::success();\n";
  }
}

void OpEmitter::genOperandResultVerifier(OpMethodBody &body,
                                         Operator::value_range values,
                                         StringRef valueKind) {
  FmtContext fctx;

  body << "  {\n";
  body << "    unsigned index = 0; (void)index;\n";

  for (auto staticValue : llvm::enumerate(values)) {
    if (!staticValue.value().hasPredicate())
      continue;

    // Emit a loop to check all the dynamic values in the pack.
    body << formatv("    for (Value *v : getODS{0}{1}s({2})) {{\n",
                    // Capitalize the first letter to match the function name
                    valueKind.substr(0, 1).upper(), valueKind.substr(1),
                    staticValue.index());

    auto constraint = staticValue.value().constraint;

    body << "      (void)v;\n"
         << "      if (!("
         << tgfmt(constraint.getConditionTemplate(),
                  &fctx.withSelf("v->getType()"))
         << ")) {\n"
         << formatv("        return emitOpError(\"{0} #\") << index "
                    "<< \" must be {1}, but got \" << v->getType();\n",
                    valueKind, constraint.getDescription())
         << "      }\n" // if
         << "      ++index;\n"
         << "    }\n"; // for
  }

  body << "  }\n";
}

void OpEmitter::genRegionVerifier(OpMethodBody &body) {
  unsigned numRegions = op.getNumRegions();

  // Verify this op has the correct number of regions
  body << formatv(
      "  if (this->getOperation()->getNumRegions() != {0}) {\n    "
      "return emitOpError(\"has incorrect number of regions: expected {0} but "
      "found \") << this->getOperation()->getNumRegions();\n  }\n",
      numRegions);

  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);

    std::string name = formatv("#{0}", i);
    if (!region.name.empty()) {
      name += formatv(" ('{0}')", region.name);
    }

    auto getRegion = formatv("this->getOperation()->getRegion({0})", i).str();
    auto constraint = tgfmt(region.constraint.getConditionTemplate(),
                            &verifyCtx.withSelf(getRegion))
                          .str();

    body << formatv("  if (!({0})) {\n    "
                    "return emitOpError(\"region {1} failed to verify "
                    "constraint: {2}\");\n  }\n",
                    constraint, name, region.constraint.getDescription());
  }
}

void OpEmitter::genTraits() {
  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariadicResults();

  // Add return size trait.
  if (numVariadicResults != 0) {
    if (numResults == numVariadicResults)
      opClass.addTrait("OpTrait::VariadicResults");
    else
      opClass.addTrait("OpTrait::AtLeastNResults<" +
                       Twine(numResults - numVariadicResults) + ">::Impl");
  } else {
    switch (numResults) {
    case 0:
      opClass.addTrait("OpTrait::ZeroResult");
      break;
    case 1:
      opClass.addTrait("OpTrait::OneResult");
      break;
    default:
      opClass.addTrait("OpTrait::NResults<" + Twine(numResults) + ">::Impl");
      break;
    }
  }

  for (const auto &trait : op.getTraits()) {
    if (auto opTrait = dyn_cast<tblgen::NativeOpTrait>(&trait))
      opClass.addTrait(opTrait->getTrait());
    else if (auto opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait))
      opClass.addTrait(opTrait->getTrait());
  }

  // Add variadic size trait and normal op traits.
  int numOperands = op.getNumOperands();
  int numVariadicOperands = op.getNumVariadicOperands();

  // Add operand size trait.
  if (numVariadicOperands != 0) {
    if (numOperands == numVariadicOperands)
      opClass.addTrait("OpTrait::VariadicOperands");
    else
      opClass.addTrait("OpTrait::AtLeastNOperands<" +
                       Twine(numOperands - numVariadicOperands) + ">::Impl");
  } else {
    switch (numOperands) {
    case 0:
      opClass.addTrait("OpTrait::ZeroOperands");
      break;
    case 1:
      opClass.addTrait("OpTrait::OneOperand");
      break;
    default:
      opClass.addTrait("OpTrait::NOperands<" + Twine(numOperands) + ">::Impl");
      break;
    }
  }
}

void OpEmitter::genOpNameGetter() {
  auto &method = opClass.newMethod("StringRef", "getOperationName",
                                   /*params=*/"", OpMethod::MP_Static);
  method.body() << "  return \"" << op.getOperationName() << "\";\n";
}

void OpEmitter::genOpAsmInterface() {
  // If the user only has one results or specifically added the Asm trait,
  // then don't generate it for them. We specifically only handle multi result
  // operations, because the name of a single result in the common case is not
  // interesting(generally 'result'/'output'/etc.).
  // TODO: We could also add a flag to allow operations to opt in to this
  // generation, even if they only have a single operation.
  int numResults = op.getNumResults();
  if (numResults <= 1 || op.getTrait("OpAsmOpInterface::Trait"))
    return;

  SmallVector<StringRef, 4> resultNames(numResults);
  for (int i = 0; i != numResults; ++i)
    resultNames[i] = op.getResultName(i);

  // Don't add the trait if none of the results have a valid name.
  if (llvm::all_of(resultNames, [](StringRef name) { return name.empty(); }))
    return;
  opClass.addTrait("OpAsmOpInterface::Trait");

  // Generate the right accessor for the number of results.
  auto &method = opClass.newMethod("void", "getAsmResultNames",
                                   "OpAsmSetValueNameFn setNameFn");
  auto &body = method.body();
  for (int i = 0; i != numResults; ++i) {
    body << "  auto resultGroup" << i << " = getODSResults(" << i << ");\n"
         << "  if (!llvm::empty(resultGroup" << i << "))\n"
         << "    setNameFn(*resultGroup" << i << ".begin(), \""
         << resultNames[i] << "\");\n";
  }
}

//===----------------------------------------------------------------------===//
// OpOperandAdaptor emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit Op operand adaptors to an output stream.  Operand
// adaptors are wrappers around ArrayRef<Value *> that provide named operand
// getters identical to those defined in the Op.
class OpOperandAdaptorEmitter {
public:
  static void emitDecl(const Operator &op, raw_ostream &os);
  static void emitDef(const Operator &op, raw_ostream &os);

private:
  explicit OpOperandAdaptorEmitter(const Operator &op);

  Class adapterClass;
};
} // end namespace

OpOperandAdaptorEmitter::OpOperandAdaptorEmitter(const Operator &op)
    : adapterClass(op.getCppClassName().str() + "OperandAdaptor") {
  adapterClass.newField("ArrayRef<Value *>", "tblgen_operands");
  auto &constructor = adapterClass.newConstructor("ArrayRef<Value *> values");
  constructor.body() << "  tblgen_operands = values;\n";

  generateNamedOperandGetters(op, adapterClass,
                              /*rangeType=*/"ArrayRef<Value *>",
                              /*rangeBeginCall=*/"tblgen_operands.begin()",
                              /*rangeSizeCall=*/"tblgen_operands.size()",
                              /*getOperandCallPattern=*/"tblgen_operands[{0}]");
}

void OpOperandAdaptorEmitter::emitDecl(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adapterClass.writeDeclTo(os);
}

void OpOperandAdaptorEmitter::emitDef(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adapterClass.writeDefTo(os);
}

// Emits the opcode enum and op classes.
static void emitOpClasses(const std::vector<Record *> &defs, raw_ostream &os,
                          bool emitDecl) {
  IfDefScope scope("GET_OP_CLASSES", os);
  // First emit forward declaration for each class, this allows them to refer
  // to each others in traits for example.
  if (emitDecl) {
    for (auto *def : defs) {
      Operator op(*def);
      os << "class " << op.getCppClassName() << ";\n";
    }
  }
  for (auto *def : defs) {
    Operator op(*def);
    const auto *attrSizedOperands =
        op.getTrait("OpTrait::AttrSizedOperandSegments");
    if (emitDecl) {
      os << formatv(opCommentHeader, op.getQualCppClassName(), "declarations");
      // We cannot generate the operand adaptor class if operand getters depend
      // on an attribute.
      if (!attrSizedOperands)
        OpOperandAdaptorEmitter::emitDecl(op, os);
      OpEmitter::emitDecl(op, os);
    } else {
      os << formatv(opCommentHeader, op.getQualCppClassName(), "definitions");
      if (!attrSizedOperands)
        OpOperandAdaptorEmitter::emitDef(op, os);
      OpEmitter::emitDef(op, os);
    }
  }
}

// Emits a comma-separated list of the ops.
static void emitOpList(const std::vector<Record *> &defs, raw_ostream &os) {
  IfDefScope scope("GET_OP_LIST", os);

  interleave(
      // TODO: We are constructing the Operator wrapper instance just for
      // getting it's qualified class name here. Reduce the overhead by having a
      // lightweight version of Operator class just for that purpose.
      defs, [&os](Record *def) { os << Operator(def).getQualCppClassName(); },
      [&os]() { os << ",\n"; });
}

static bool emitOpDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Declarations", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpClasses(defs, os, /*emitDecl=*/true);

  return false;
}

static bool emitOpDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Definitions", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpList(defs, os);
  emitOpClasses(defs, os, /*emitDecl=*/false);

  return false;
}

static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpDecls(records, os);
               });

static mlir::GenRegistration genOpDefs("gen-op-defs", "Generate op definitions",
                                       [](const RecordKeeper &records,
                                          raw_ostream &os) {
                                         return emitOpDefs(records, os);
                                       });
