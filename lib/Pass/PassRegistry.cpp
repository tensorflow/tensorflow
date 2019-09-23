//===- PassRegistry.cpp - Pass Registration Utilities ---------------------===//
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

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace detail;

/// Static mapping of all of the registered passes.
static llvm::ManagedStatic<llvm::DenseMap<const PassID *, PassInfo>>
    passRegistry;

/// Static mapping of all of the registered pass pipelines.
static llvm::ManagedStatic<llvm::StringMap<PassPipelineInfo>>
    passPipelineRegistry;

/// Utility to create a default registry function from a pass instance.
static PassRegistryFunction
buildDefaultRegistryFn(PassAllocatorFunction allocator) {
  return [=](OpPassManager &pm) { pm.addPass(allocator()); };
}

//===----------------------------------------------------------------------===//
// PassPipelineInfo
//===----------------------------------------------------------------------===//

/// Constructor that accepts a pass allocator function instead of the standard
/// registry function. This is useful for registering specializations of
/// existing passes.
PassPipelineRegistration::PassPipelineRegistration(
    StringRef arg, StringRef description, PassAllocatorFunction allocator) {
  registerPassPipeline(arg, description, buildDefaultRegistryFn(allocator));
}

void mlir::registerPassPipeline(StringRef arg, StringRef description,
                                const PassRegistryFunction &function) {
  PassPipelineInfo pipelineInfo(arg, description, function);
  bool inserted = passPipelineRegistry->try_emplace(arg, pipelineInfo).second;
  assert(inserted && "Pass pipeline registered multiple times");
  (void)inserted;
}

//===----------------------------------------------------------------------===//
// PassInfo
//===----------------------------------------------------------------------===//

PassInfo::PassInfo(StringRef arg, StringRef description, const PassID *passID,
                   PassAllocatorFunction allocator)
    : PassRegistryEntry(arg, description, buildDefaultRegistryFn(allocator)) {}

void mlir::registerPass(StringRef arg, StringRef description,
                        const PassID *passID,
                        const PassAllocatorFunction &function) {
  PassInfo passInfo(arg, description, passID, function);
  bool inserted = passRegistry->try_emplace(passID, passInfo).second;
  assert(inserted && "Pass registered multiple times");
  (void)inserted;
}

/// Returns the pass info for the specified pass class or null if unknown.
const PassInfo *mlir::Pass::lookupPassInfo(const PassID *passID) {
  auto it = passRegistry->find(passID);
  if (it == passRegistry->end())
    return nullptr;
  return &it->getSecond();
}

//===----------------------------------------------------------------------===//
// TextualPassPipeline Parser
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a textual description of a pass pipeline.
class TextualPipeline {
public:
  /// Try to initialize this pipeline with the given pipeline text.
  /// `errorStream` is the output stream to emit errors to.
  LogicalResult initialize(StringRef text, raw_ostream &errorStream);

  /// Add the internal pipeline elements to the provided pass manager.
  void addToPipeline(OpPassManager &pm) const;

private:
  /// A functor used to emit errors found during pipeline handling. The first
  /// parameter corresponds to the raw location within the pipeline string. This
  /// should always return failure.
  using ErrorHandlerT = function_ref<LogicalResult(const char *, llvm::Twine)>;

  /// A struct to capture parsed pass pipeline names.
  ///
  /// A pipeline is defined as a series of names, each of which may in itself
  /// recursively contain a nested pipeline. A name is either the name of a pass
  /// (e.g. "cse") or the name of an operation type (e.g. "func"). If the name
  /// is the name of a pass, the InnerPipeline is empty, since passes cannot
  /// contain inner pipelines.
  struct PipelineElement {
    PipelineElement(StringRef name) : name(name), registryEntry(nullptr) {}

    StringRef name;
    const PassRegistryEntry *registryEntry;
    std::vector<PipelineElement> innerPipeline;
  };

  /// Parse the given pipeline text into the internal pipeline vector. This
  /// function only parses the structure of the pipeline, and does not resolve
  /// its elements.
  LogicalResult parsePipelineText(StringRef text, ErrorHandlerT errorHandler);

  /// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
  /// the corresponding registry entry.
  LogicalResult
  resolvePipelineElements(MutableArrayRef<PipelineElement> elements,
                          ErrorHandlerT errorHandler);

  /// Resolve a single element of the pipeline.
  LogicalResult resolvePipelineElement(PipelineElement &element,
                                       ErrorHandlerT errorHandler);

  /// Add the given pipeline elements to the provided pass manager.
  void addToPipeline(ArrayRef<PipelineElement> elements,
                     OpPassManager &pm) const;

  std::vector<PipelineElement> pipeline;
};

} // end anonymous namespace

/// Try to initialize this pipeline with the given pipeline text. An option is
/// given to enable accurate error reporting.
LogicalResult TextualPipeline::initialize(StringRef text,
                                          raw_ostream &errorStream) {
  // Build a source manager to use for error reporting.
  llvm::SourceMgr pipelineMgr;
  pipelineMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(
                                     text, "MLIR Textual PassPipeline Parser"),
                                 llvm::SMLoc());
  auto errorHandler = [&](const char *rawLoc, llvm::Twine msg) {
    pipelineMgr.PrintMessage(errorStream, llvm::SMLoc::getFromPointer(rawLoc),
                             llvm::SourceMgr::DK_Error, msg);
    return failure();
  };

  // Parse the provided pipeline string.
  if (failed(parsePipelineText(text, errorHandler)))
    return failure();
  return resolvePipelineElements(pipeline, errorHandler);
}

/// Add the internal pipeline elements to the provided pass manager.
void TextualPipeline::addToPipeline(OpPassManager &pm) const {
  addToPipeline(pipeline, pm);
}

/// Parse the given pipeline text into the internal pipeline vector. This
/// function only parses the structure of the pipeline, and does not resolve
/// its elements.
LogicalResult TextualPipeline::parsePipelineText(StringRef text,
                                                 ErrorHandlerT errorHandler) {
  SmallVector<std::vector<PipelineElement> *, 4> pipelineStack = {&pipeline};
  for (;;) {
    std::vector<PipelineElement> &pipeline = *pipelineStack.back();
    size_t pos = text.find_first_of(",()");
    pipeline.emplace_back(/*name=*/text.substr(0, pos).trim());

    // If we have a single terminating name, we're done.
    if (pos == text.npos)
      break;

    char sep = text[pos];
    text = text.substr(pos + 1);

    // Just a name ending in a comma, continue.
    if (sep == ',')
      continue;

    if (sep == '(') {
      // Push the inner pipeline onto the stack to continue processing.
      pipelineStack.push_back(&pipeline.back().innerPipeline);
      continue;
    }

    // When handling the close parenthesis, we greedily consume them to avoid
    // empty strings in the pipeline.
    assert(sep == ')' && "Bogus separator!");
    do {
      // If we try to pop the outer pipeline we have unbalanced parentheses.
      if (pipelineStack.size() == 1)
        return errorHandler(/*rawLoc=*/text.data() - 1,
                            "encountered extra closing ')' creating unbalanced "
                            "parentheses while parsing pipeline");

      pipelineStack.pop_back();
    } while (text.consume_front(")"));

    // Check if we've finished parsing.
    if (text.empty())
      break;

    // Otherwise, the end of an inner pipeline always has to be followed by
    // a comma, and then we can continue.
    if (!text.consume_front(","))
      return errorHandler(text.data(), "expected ',' after parsing pipeline");
  }

  // Check for unbalanced parentheses.
  if (pipelineStack.size() > 1)
    return errorHandler(
        text.data(),
        "encountered unbalanced parentheses while parsing pipeline");

  assert(pipelineStack.back() == &pipeline &&
         "wrong pipeline at the bottom of the stack");
  return success();
}

/// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
/// the corresponding registry entry.
LogicalResult TextualPipeline::resolvePipelineElements(
    MutableArrayRef<PipelineElement> elements, ErrorHandlerT errorHandler) {
  for (auto &elt : elements)
    if (failed(resolvePipelineElement(elt, errorHandler)))
      return failure();
  return success();
}

/// Resolve a single element of the pipeline.
LogicalResult
TextualPipeline::resolvePipelineElement(PipelineElement &element,
                                        ErrorHandlerT errorHandler) {
  // If the inner pipeline of this element is not empty, this is an operation
  // pipeline.
  if (!element.innerPipeline.empty())
    return resolvePipelineElements(element.innerPipeline, errorHandler);

  // Otherwise, this must be a pass or pass pipeline.
  // Check to see if a pipeline was registered with this name.
  auto pipelineRegistryIt = passPipelineRegistry->find(element.name);
  if (pipelineRegistryIt != passPipelineRegistry->end()) {
    element.registryEntry = &pipelineRegistryIt->second;
    return success();
  }

  // If not, then this must be a specific pass name.
  for (auto &passIt : *passRegistry) {
    if (passIt.second.getPassArgument() == element.name) {
      element.registryEntry = &passIt.second;
      return success();
    }
  }

  // Emit an error for the unknown pass.
  auto *rawLoc = element.name.data();
  return errorHandler(rawLoc, "'" + element.name +
                                  "' does not refer to a "
                                  "registered pass or pass pipeline");
}

/// Add the given pipeline elements to the provided pass manager.
void TextualPipeline::addToPipeline(ArrayRef<PipelineElement> elements,
                                    OpPassManager &pm) const {
  for (auto &elt : elements) {
    if (elt.registryEntry)
      elt.registryEntry->addToPipeline(pm);
    else
      addToPipeline(elt.innerPipeline, pm.nest(elt.name));
  }
}

/// This function parses the textual representation of a pass pipeline, and adds
/// the result to 'pm' on success. This function returns failure if the given
/// pipeline was invalid. 'errorStream' is an optional parameter that, if
/// non-null, will be used to emit errors found during parsing.
LogicalResult mlir::parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                      raw_ostream &errorStream) {
  TextualPipeline pipelineParser;
  if (failed(pipelineParser.initialize(pipeline, errorStream)))
    return failure();
  pipelineParser.addToPipeline(pm);
  return success();
}

//===----------------------------------------------------------------------===//
// PassNameParser
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents the possible data entries in a parsed pass pipeline
/// list.
struct PassArgData {
  PassArgData() : registryEntry(nullptr) {}
  PassArgData(const PassRegistryEntry *registryEntry)
      : registryEntry(registryEntry) {}

  /// This field is used when the parsed option corresponds to a registered pass
  /// or pass pipeline.
  const PassRegistryEntry *registryEntry;

  /// This field is used when the parsed option corresponds to an explicit
  /// pipeline.
  TextualPipeline pipeline;
};
} // end anonymous namespace

namespace llvm {
namespace cl {
/// Define a valid OptionValue for the command line pass argument.
template <>
struct OptionValue<PassArgData> final
    : OptionValueBase<PassArgData, /*isClass=*/true> {
  OptionValue(const PassArgData &value) { this->setValue(value); }
  OptionValue() = default;
  void anchor() override {}

  bool hasValue() const { return true; }
  const PassArgData &getValue() const { return value; }
  void setValue(const PassArgData &value) { this->value = value; }

  PassArgData value;
};
} // end namespace cl
} // end namespace llvm

namespace {

/// The name for the command line option used for parsing the textual pass
/// pipeline.
static constexpr llvm::StringLiteral passPipelineArg = "pass-pipeline";

/// Adds command line option for each registered pass or pass pipeline, as well
/// as textual pass pipelines.
struct PassNameParser : public llvm::cl::parser<PassArgData> {
  PassNameParser(llvm::cl::Option &opt) : llvm::cl::parser<PassArgData>(opt) {}

  void initialize();
  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const override;
  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             PassArgData &value);
};
} // namespace

void PassNameParser::initialize() {
  llvm::cl::parser<PassArgData>::initialize();

  /// Add an entry for the textual pass pipeline option.
  addLiteralOption(passPipelineArg, PassArgData(),
                   "A textual description of a pass pipeline to run");

  /// Add the pass entries.
  for (const auto &kv : *passRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
  /// Add the pass pipeline entries.
  for (const auto &kv : *passPipelineRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
}

void PassNameParser::printOptionInfo(const llvm::cl::Option &O,
                                     size_t GlobalWidth) const {
  PassNameParser *TP = const_cast<PassNameParser *>(this);
  llvm::array_pod_sort(TP->Values.begin(), TP->Values.end(),
                       [](const PassNameParser::OptionInfo *VT1,
                          const PassNameParser::OptionInfo *VT2) {
                         return VT1->Name.compare(VT2->Name);
                       });
  llvm::cl::parser<PassArgData>::printOptionInfo(O, GlobalWidth);
}

bool PassNameParser::parse(llvm::cl::Option &opt, StringRef argName,
                           StringRef arg, PassArgData &value) {
  // Handle the pipeline option explicitly.
  if (argName == passPipelineArg)
    return failed(value.pipeline.initialize(arg, llvm::errs()));

  // Otherwise, default to the base for handling.
  return llvm::cl::parser<PassArgData>::parse(opt, argName, arg, value);
}

//===----------------------------------------------------------------------===//
// PassPipelineCLParser
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct PassPipelineCLParserImpl {
  PassPipelineCLParserImpl(StringRef arg, StringRef description)
      : passList(arg, llvm::cl::desc(description)) {
    passList.setValueExpectedFlag(llvm::cl::ValueExpected::ValueOptional);
  }

  /// The set of passes and pass pipelines to run.
  llvm::cl::list<PassArgData, bool, PassNameParser> passList;
};
} // end namespace detail
} // end namespace mlir

/// Construct a pass pipeline parser with the given command line description.
PassPipelineCLParser::PassPipelineCLParser(StringRef arg, StringRef description)
    : impl(std::make_unique<detail::PassPipelineCLParserImpl>(arg,
                                                              description)) {}
PassPipelineCLParser::~PassPipelineCLParser() {}

/// Returns true if this parser contains any valid options to add.
bool PassPipelineCLParser::hasAnyOccurrences() const {
  return impl->passList.getNumOccurrences() != 0;
}

/// Returns true if the given pass registry entry was registered at the
/// top-level of the parser, i.e. not within an explicit textual pipeline.
bool PassPipelineCLParser::contains(const PassRegistryEntry *entry) const {
  return llvm::any_of(impl->passList, [&](const PassArgData &data) {
    return data.registryEntry == entry;
  });
}

/// Adds the passes defined by this parser entry to the given pass manager.
void PassPipelineCLParser::addToPipeline(OpPassManager &pm) const {
  for (auto &passIt : impl->passList) {
    if (passIt.registryEntry)
      passIt.registryEntry->addToPipeline(pm);
    else
      passIt.pipeline.addToPipeline(pm);
  }
}
