/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// A tool for interactively exploring graphviz dumps of HLO graphs.
//
// Input can be a binary HloSnapshot proto, a binary HloProto proto, or a
// textual HLO string.
//
// Generated visualization is opened in a new default browser window using
// /usr/bin/sensible-browser.

#include <stdio.h>
#include <unistd.h>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tools/hlo_extractor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/util/command_line_flags.h"
#if defined(PLATFORM_GOOGLE)
#include "util/readline/readline.h"
#endif

namespace xla {
namespace tools {
namespace {

bool ReadLine(const char *prompt, string *line) {
#if defined(PLATFORM_GOOGLE)
  return util::ReadLine(prompt, line);
#else
  std::cout << prompt;
  std::getline(std::cin, *line);
  return std::cin.good();
#endif
}

// Command-line opts to this tool.  See main() for descriptions of these
// fields.
struct Options {
  string hlo_snapshot;
  string hlo_proto;
  string hlo_text;
  string platform;
  string browser;
};

const char* const kUsage = R"(
This tool lets you load an XLA dump and then interactively explore its graphical
representation.

Most models are too large to visualize in their entirety using graphviz, but
it's still useful to be able to look at the nodes "near" a particular node of
interest.

If you pass --platform, this tool will compile the HloModule for the given
platform.  This means that if you acquired your proto from a binary running at a
particular CL, the HLO graph it ran isn't necessarily the same as the one shown
here, unless this program was built at the same CL (and our compiler is
deterministic :).

Be patient when starting this program if you give it a large input; it has to
compile the whole thing.

Usage:

  interactive_graphviz -- \
    --{hlo_snapshot,hlo_proto,hlo_text}=path/to/binary_proto
    --platform={CUDA,CPU,...}
)";

// Unless an explicit width is specified, we will render a neighborhood of
// kDefaultWidth nodes around the requested instruction.
constexpr int64 kDefaultWidth = 2;

// When printing all paths between two nodes, we print out only this many nodes
// by default, truncating the graph if there are more nodes than this in the
// all-paths set.
constexpr int64 kDefaultMaxNumNodesInAllPaths = 100;

using absl::EqualsIgnoreCase;

// A global control for whether backend configuration display is enabled.
bool show_backend_config = true;

HloInstruction* FindInstruction(const HloModule& module, string node_name) {
  if (absl::StartsWith(node_name, "%")) {
    node_name.erase(node_name.begin());
  }
  for (const auto& computation : module.computations()) {
    auto instrs = computation->instructions();
    auto it = absl::c_find_if(instrs, [&](const HloInstruction* instr) {
      // Try with and without "%" at the beginning of the node name.
      return EqualsIgnoreCase(instr->name(), node_name) ||
             EqualsIgnoreCase(instr->name(), absl::StrCat("%", node_name));
    });
    if (it != instrs.end()) {
      return *it;
    }
  }
  return nullptr;
}

HloComputation* FindComputation(const HloModule& module,
                                const string& comp_name) {
  for (auto* computation : module.computations()) {
    if (EqualsIgnoreCase(computation->name(), comp_name)) {
      return computation;
    }
  }
  return nullptr;
}

// Print a help message describing the various available commands.
void DoHelpCommand() {
  std::cout << R"(Commands:
  <instruction> [<width>] [/ <boundary_instruction>+]
    Renders a neighborhood of <width> nodes around <instruction>, without going
    beyond the optional boundary instructions.  If <width> is not provided, 
    the default value is )"
            << kDefaultWidth << R"(.
  allpaths <instruction> <instruction> [<n>]
    Renders a subset of all paths from one instruction to the other.  Either
    order of nodes is accepted.  Shows the <n> nodes in the all-paths set on the
    shortest paths; default is )"
            << kDefaultMaxNumNodesInAllPaths << R"(.
  <computation>
    Renders all nodes in <computation>.
  backend_config [on|off]
    Controls whether backend operation configuration information is printed.
  list [name|op_name|op_type] <pattern>
    Lists all instructions whose name, metadata op_name, or metadata op_type
    contains <pattern> as a substring.
  list computations
    Lists all computations in the module.
  info <instruction>
  info <computation>
    Prints information about <instruction> or <computation>.
  extract <instruction> <height>
    Creates a new HLO module with <instruction> as entry computation root. If
    <height> is specified, the new computation contains nodes up to <height>
    nodes above the root.
  help
    Prints this usage information.)"
            << std::endl;
}

// Turn metadata-printing on or off.
void DoBackendConfigCommand(const std::vector<string>& tokens) {
  if (tokens.size() == 2 && tokens[1] == "on") {
    show_backend_config = true;
  } else if (tokens.size() == 2 && tokens[1] == "off") {
    show_backend_config = false;
  } else if (tokens.size() != 1) {
    std::cerr << "(Illegal backend_config value.  Use either 'on' or 'off'.)"
              << std::endl;
  }
  std::cout << "Backend configuration display "
            << (show_backend_config ? "ON" : "OFF") << std::endl;
}

// List all computations in the module.
void DoListComputationsCommand(const HloModule& module,
                               const std::vector<string>& tokens) {
  if (tokens.size() > 2) {
    std::cout << R"(Illegal syntax; "list computations" takes no arguments.)";
    return;
  }
  if (module.entry_computation() != nullptr) {
    std::cout << "Entry computation:" << std::endl;
    std::cout << "  " << module.entry_computation()->name() << std::endl
              << std::endl;
  }
  std::cout << "Subcomputations:" << std::endl;
  std::vector<string> names;
  for (const auto& computation : module.computations()) {
    if (computation == module.entry_computation()) {
      continue;
    }
    std::cout << "  " << computation->name() << std::endl;
  }
}

// List all instructions matching a pattern.
void DoListCommand(const HloModule& module, const std::vector<string>& tokens) {
  string pattern = "";
  string type = "name";
  if (tokens.size() == 2) {
    pattern = tokens[1];
  } else if (tokens.size() == 3) {
    type = tokens[1];
    pattern = tokens[2];
  } else {
    std::cout << "Illegal list query syntax. Use "
              << R"("list [name|op_name|op_type] pattern".)" << std::endl;
    return;
  }

  std::cout << "Query results:" << std::endl;
  for (const auto& computation : module.computations()) {
    for (const auto& instr : computation->instructions()) {
      if ((type == "name" && instr->name().find(pattern) != string::npos) ||
          (type == "op_name" &&
           instr->metadata().op_name().find(pattern) != string::npos) ||
          (type == "op_type" &&
           instr->metadata().op_type().find(pattern) != string::npos)) {
        std::cout << "  " << instr->name();
        std::cout << ", op_name '" << instr->metadata().op_name() << "'";
        std::cout << ", op_type '" << instr->metadata().op_type() << "'";
        std::cout << std::endl;
      }
    }
  }
}

// Print info about an instruction or computation.
void DoInfoCommand(const HloModule& module, const std::vector<string>& tokens) {
  if (tokens.size() != 2) {
    std::cerr << "Illegal info query syntax. Use "
              << R"("info name".)";
    return;
  }
  string node_name = tokens[1];

  const HloInstruction* instr = FindInstruction(module, node_name);
  const HloComputation* comp = FindComputation(module, node_name);
  if (!instr && !comp) {
    std::cerr << "Couldn't find HloInstruction or HloComputation named "
              << node_name << std::endl;
    return;
  }

  if (comp != nullptr) {
    std::cout << "HloComputation " << comp->name() << std::endl;
    if (comp->IsFusionComputation()) {
      std::cout << "  Fusion instruction: " << comp->FusionInstruction()->name()
                << std::endl;
    }
    std::cout << "  Parameters:" << std::endl;
    for (const auto& param : comp->parameter_instructions()) {
      std::cout << "    " << param->name() << " ("
                << ShapeUtil::HumanStringWithLayout(param->shape()) << ")"
                << std::endl;
    }
    HloInstruction* root = comp->root_instruction();
    std::cout << "  Root instruction: " << root->name() << " ("
              << ShapeUtil::HumanStringWithLayout(root->shape()) << ")"
              << std::endl;

    auto embedded_computations = comp->MakeEmbeddedComputationsList();
    std::cout << "  " << embedded_computations.size() << " embedded computation"
              << (embedded_computations.size() != 1 ? "s" : "")
              << (!embedded_computations.empty() ? ":" : ".") << std::endl;
    for (const HloComputation* c : embedded_computations) {
      std::cout << "    " << c->name() << std::endl;
    }

    // Find which computations reference comp as an embedded computation.
    std::vector<const HloComputation*> users;
    for (const HloComputation* c : module.computations()) {
      if (absl::c_linear_search(c->MakeEmbeddedComputationsList(), comp)) {
        users.push_back(c);
      }
    }
    std::cout << "  Used by " << users.size() << " computation"
              << (users.size() != 1 ? "s" : "") << (!users.empty() ? ":" : ".");
    for (const HloComputation* c : users) {
      std::cout << "    " << c->name() << std::endl;
    }
  } else {
    std::cout << "HloInstruction " << instr->name() << std::endl;
    std::cout << "  Parent computation: " << instr->parent()->name()
              << std::endl;
    std::cout << "  Opcode: " << HloOpcodeString(instr->opcode()) << std::endl;
    std::cout << "  Shape: " << ShapeUtil::HumanStringWithLayout(instr->shape())
              << std::endl;
    std::cout << "  Metadata:" << std::endl;
    if (!instr->metadata().op_name().empty()) {
      std::cout << "    Name: " << instr->metadata().op_name() << std::endl;
    }
    if (!instr->metadata().op_type().empty()) {
      std::cout << "    Type: " << instr->metadata().op_type() << std::endl;
    }
    if (!instr->raw_backend_config_string().empty()) {
      std::cout << "  Backend configuration: "
                << instr->raw_backend_config_string() << std::endl;
    }
    if (instr->opcode() == HloOpcode::kFusion) {
      std::cout << "  Fusion kind: " << xla::ToString(instr->fusion_kind())
                << std::endl;
      std::cout << "  Fusion computation: "
                << instr->fused_instructions_computation()->name() << std::endl;
      std::cout << "  Fused computation root: "
                << instr->fused_expression_root()->name() << std::endl;
    }
    std::cout << "  Operands:" << std::endl;
    for (HloInstruction* operand : instr->operands()) {
      std::cout << "    " << operand->name() << " ("
                << ShapeUtil::HumanStringWithLayout(operand->shape()) << ")"
                << std::endl;
    }
    std::cout << "  Users:" << std::endl;
    for (HloInstruction* user : instr->users()) {
      std::cout << "    " << user->name() << std::endl;
    }
    if (instr->parent()->root_instruction() == instr) {
      std::cout << "  Root instruction of " << instr->parent()->name()
                << std::endl;
    }
  }
}

void DoExtractCommand(const HloModule& module,
                      absl::Span<const string> tokens) {
  if (tokens.size() > 3) {
    std::cerr << R"(Illegal input.  Enter e.g. "extract %fusion.1 2")"
              << std::endl;
    return;
  }

  // Find the node with the given name.
  string node_name = tokens[1];
  HloInstruction* instr = FindInstruction(module, node_name);
  if (!instr) {
    std::cerr << "Couldn't find HloInstruction named " << node_name << "."
              << std::endl;
    return;
  }

  int64 height = -1;
  if (tokens.size() == 3) {
    if (!absl::SimpleAtoi(tokens[2], &height)) {
      std::cerr << "Can't parse '" << tokens[2] << "' as an integer."
                << std::endl;
      return;
    }
  }

  auto extracted_module = ExtractModule(instr, height);
  std::cout << extracted_module->ToString(
                   HloPrintOptions::ShortParsable().set_print_backend_config(
                       show_backend_config))
            << std::endl;
}

// Checks if there is a use-def path from `from` to `to`.
bool ExistsPathFromTo(const HloInstruction* from, const HloInstruction* to) {
  std::unordered_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> to_visit = {from};
  while (!to_visit.empty()) {
    auto* n = to_visit.back();
    if (n == to) {
      return true;
    }
    to_visit.pop_back();
    visited.insert(n);
    for (auto* user : n->users()) {
      if (!visited.count(user)) {
        to_visit.push_back(user);
      }
    }
  }
  return false;
}

void DisplayGraphHandle(const Options &opts, const string& handle) {
  std::cout << handle << std::endl;

  // If it is a url, try to open it up in the user's browser too.
  if (absl::StartsWithIgnoreCase(handle, "http://") ||
      absl::StartsWithIgnoreCase(handle, "https://") ||
      absl::StartsWithIgnoreCase(handle, "file://")) {
    const char* browser_bin = opts.browser.empty() ? "/usr/bin/sensible-browser"
                                                   : opts.browser.c_str();
    tensorflow::SubProcess p;
    p.SetProgram(browser_bin, {browser_bin, handle});
    p.Start();
  } else if (handle.empty()) {
    std::cerr << "Unable to render graph, perhaps due to graphviz server "
                 "timeout.  Run with --logtostderr to see."
              << std::endl;
  } else {
    std::cerr << "\nExpected a URL, but got strange graph result (dumped "
                 "above).  If this isn't what you expected, maybe file a bug?"
              << std::endl;
  }
}

void DoAllPathsCommand(const Options& opts, const HloModule& module,
                       const std::vector<string>& tokens) {
  if (tokens.size() > 4) {
    std::cerr << R"(Illegal input.  Enter e.g. "allpaths %add.4 %subtract.2" or
"allpaths add.4 subtract.2 42.)"
              << std::endl;
    return;
  }

  int64 max_nodes = kDefaultMaxNumNodesInAllPaths;
  if (tokens.size() == 4 && !absl::SimpleAtoi(tokens[3], &max_nodes)) {
    std::cerr << "Can't parse '" << tokens[3] << "' as an integer."
              << std::endl;
    return;
  }

  const HloInstruction* n1 = FindInstruction(module, tokens[1]);
  if (!n1) {
    std::cerr << "Couldn't find HloInstruction named " << tokens[1];
    return;
  }
  const HloInstruction* n2 = FindInstruction(module, tokens[2]);
  if (!n2) {
    std::cerr << "Couldn't find HloInstruction named " << tokens[2];
    return;
  }

  // Is there a path from n1 to n2, or vice versa?
  const HloInstruction* from;
  const HloInstruction* to;
  if (ExistsPathFromTo(n1, n2)) {
    from = n1;
    to = n2;
  } else if (ExistsPathFromTo(n2, n1)) {
    from = n2;
    to = n1;
  } else {
    std::cerr << "No path from/to " << tokens[1] << " to/from " << tokens[2];
    return;
  }
  DisplayGraphHandle(opts, hlo_graph_dumper::DumpAllPathsFromTo(
      *from, *to, max_nodes, /*show_backend_config=*/show_backend_config));
}

// Plot a given instruction neighborhood or computation with graphviz.
void DoPlotCommand(const Options& opts, const HloModule& module,
                   const std::vector<string>& tokens) {
  string node_name = tokens[0];

  // Find the node with the given name.
  const HloInstruction* instr = FindInstruction(module, node_name);
  const HloComputation* comp = FindComputation(module, node_name);
  if (!instr && !comp) {
    std::cerr << "Couldn't find HloInstruction or HloComputation named "
              << node_name << "." << std::endl;
    return;
  }

  uint64 graph_width = kDefaultWidth;
  std::set<const HloInstruction*> boundary;
  if (tokens.size() >= 2) {
    if (comp) {
      std::cerr << "Can only use graph-size parameter with instructions, but "
                << node_name << " is a computation." << std::endl;
      return;
    }

    int bound_index = tokens.size();
    if (!absl::SimpleAtoi(tokens[1], &graph_width)) {
      if (tokens[1] != "/") {
        std::cerr << "Can't parse '" << tokens[1] << "' as an integer."
                  << std::endl;
        return;
      }
      graph_width = kDefaultWidth;
      bound_index = 2;
    } else {
      if (tokens.size() > 2) {
        if (tokens[2] != "/") {
          std::cerr << "Expect a /, but get a '" << tokens[1] << "'."
                    << std::endl;
          return;
        }
        bound_index = 3;
      }
    }
    while (bound_index < tokens.size()) {
      string bnode_name = tokens[bound_index];
      const HloInstruction* binstr = FindInstruction(module, bnode_name);
      if (!binstr) {
        std::cerr << "Couldn't find HloInstruction named " << node_name << "."
                  << std::endl;
        return;
      }
      boundary.insert(binstr);
      bound_index++;
    }
  }

  // Generate the graph and print the resulting string, which should be a
  // graphviz url.
  if (comp) {
    DisplayGraphHandle(opts, hlo_graph_dumper::DumpGraph(
        *comp, "", comp->parent()->config().debug_options(), nullptr,
        /*show_backend_config=*/show_backend_config));
  } else {
    DisplayGraphHandle(opts, hlo_graph_dumper::DumpNeighborhoodAround(
                                 *instr, graph_width,
                                 /*boundary=*/boundary,
                                 /*show_backend_config=*/show_backend_config));
  }
}

// Run the main event loop, reading user commands and processing them.
void InteractiveDumpGraphs(const Options& opts, const HloModule& module) {
  // This is an interactive tool, but some may use `extract` in non-tty
  // environment anyway. Give them a clean hlo dump.
  if (isatty(fileno(stdin))) {
    std::cout << "\n\nLoaded module " << module.name() << "." << std::endl;
    DoHelpCommand();
  }
  for (string line; ReadLine("\ncommand: ", &line);) {
    if (line.empty()) {
      std::cout << R"(Enter e.g. "fusion.1 3" or "add.8".)" << std::endl
                << R"(Enter "help" for help; ^D, "quit", or "exit" to exit.)"
                << std::endl;
      continue;
    }
    std::vector<string> tokens = absl::StrSplit(line, ' ');
    if (tokens[0] == "quit" || tokens[0] == "exit") {
      break;
    } else if (tokens[0] == "help") {
      DoHelpCommand();
    } else if (tokens[0] == "backend_config") {
      DoBackendConfigCommand(tokens);
    } else if (tokens[0] == "list") {
      if (tokens.size() > 1 && tokens[1] == "computations") {
        DoListComputationsCommand(module, tokens);
      } else {
        DoListCommand(module, tokens);
      }
    } else if (tokens[0] == "info") {
      DoInfoCommand(module, tokens);
    } else if (tokens[0] == "extract") {
      DoExtractCommand(module, tokens);
    } else if (tokens[0] == "allpaths") {
      DoAllPathsCommand(opts, module, tokens);
    } else {
      DoPlotCommand(opts, module, tokens);
    }
  }
}

void CheckFlags(const Options &opts) {
  std::vector<string> nonempty_proto_flags;
  if (!opts.hlo_proto.empty()) {
    nonempty_proto_flags.push_back("--hlo_proto");
  }
  if (!opts.hlo_snapshot.empty()) {
    nonempty_proto_flags.push_back("--hlo_snapshot");
  }
  if (!opts.hlo_text.empty()) {
    nonempty_proto_flags.push_back("--hlo_text");
  }
  switch (nonempty_proto_flags.size()) {
    case 1:
      // We're good to go.
      break;
    case 0:
      LOG(FATAL) << "Need one of the following options: "
                 << absl::StrJoin(nonempty_proto_flags, ", ");
    default:
      LOG(FATAL) << "Can only specify one of "
                 << absl::StrJoin(nonempty_proto_flags, ", ");
  }
}

void RealMain(const Options& opts) {
  if (!isatty(fileno(stdin))) {
    LOG(ERROR) << "\n\n*****************************************\n"
               << "This is an interactive tool, but stdin is not a tty.\n"
               << "*****************************************\n\n";
  }

  CheckFlags(opts);

  std::unique_ptr<HloModule> module;
  if (!opts.hlo_snapshot.empty()) {
    HloSnapshot snapshot;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                            opts.hlo_snapshot, &snapshot))
        << "Can't open, read, or parse HloSnapshot proto at "
        << opts.hlo_snapshot;
    auto config =
        HloModule::CreateModuleConfigFromProto(snapshot.hlo().hlo_module(),
                                               xla::GetDebugOptionsFromFlags())
            .ValueOrDie();
    module = HloModule::CreateFromProto(snapshot.hlo().hlo_module(), config)
                 .ValueOrDie();
  } else if (!opts.hlo_proto.empty()) {
    module = HloRunner::ReadModuleFromBinaryProtoFile(
                 opts.hlo_proto, xla::GetDebugOptionsFromFlags())
                 .ValueOrDie();
  } else if (!opts.hlo_text.empty()) {
    module = HloRunner::ReadModuleFromHloTextFile(
                 opts.hlo_text, xla::GetDebugOptionsFromFlags())
                 .ValueOrDie();
  }

  // If a platform was specified, compile the module for that platform.
  if (!opts.platform.empty()) {
    se::Platform* platform =
        PlatformUtil::GetPlatform(opts.platform).ValueOrDie();
    LOG(INFO) << "Compiling module for " << platform->Name();

    se::StreamExecutor* executor =
        platform->ExecutorForDevice(/*ordinal=*/0).ValueOrDie();
    auto compiler = Compiler::GetForPlatform(platform).ValueOrDie();
    module = compiler
                 ->RunHloPasses(std::move(module), executor,
                                /*device_allocator=*/nullptr)
                 .ValueOrDie();
    auto executable = compiler
                          ->RunBackend(std::move(module), executor,
                                       /*device_allocator=*/nullptr)
                          .ValueOrDie();
    InteractiveDumpGraphs(opts, executable->module());
  } else {
    InteractiveDumpGraphs(opts, *module);
  }
}

}  // namespace
}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
  xla::tools::Options opts;
  opts.browser = "/usr/bin/sensible-browser";
  bool need_help = false;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("hlo_snapshot", &opts.hlo_snapshot,
                       "HloSnapshot proto to interactively dump to graphviz"),
      tensorflow::Flag("hlo_proto", &opts.hlo_proto,
                       "XLA hlo proto to interactively dump to graphviz"),
      tensorflow::Flag("hlo_text", &opts.hlo_text,
                       "XLA hlo proto to interactively dump to graphviz"),
      tensorflow::Flag("platform", &opts.platform,
                       "Platform to compile for: CPU, CUDA, etc"),
      tensorflow::Flag("browser", &opts.browser,
                       "Path to web browser used to display produced graphs."),
      tensorflow::Flag("help", &need_help,
                       "Prints this help message"),
  };
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 1 || !parse_ok || need_help) {
    LOG(QFATAL) << usage;
  }
  xla::tools::RealMain(opts);
  return 0;
}
