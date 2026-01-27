/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

// Utility to embed arbitrary files into C++ binaries.
//
// This utility encapsulates the given files as binary blobs in a .o file or
// as char arrays in a .cc file.
//
// Usage: %s [options] name files...
//
// By default, the following files are generated:
//     <name>.h       The header file describing the data.
//     <name>.cc      Table of contents initialization.
//
// Besides the names of the output files, the <name> argument also determines
// the names of initialization routines and data structures, as described
// below.
//
// The header and table of contents files are vanilla C.
//
// The header file defines the structure that stores the table of contents:
//     struct FileToc {
//       char* name;
//       char* data;
//       size_t size;
//       unsigned char md5digest[16];
//     };
//
// (This structure is also defined in file_toc.h, and these definitions must be
// kept consistent!)
//
// The <name>.cc defines the functions:
//     const struct FileToc* <name>_create();
//     size_t <name>_size();
//
// The create function initializes and returns a static array containing the
// table of contents, while the size function returns the size of that array.
//
// The original implementation was very low-dependency to allow embedding files
// low in the stack and should be modernized over time now that it is no longer
// part of the toolchain implementation.
#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MD5.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

ABSL_FLAG(std::string, align, "16", "Align embedded data to this value.");
ABSL_FLAG(bool, allow_dir, false,
          "Recursively expand directories to their contents.");
ABSL_FLAG(bool, create_header, true, "Whether to create the .h file.");
ABSL_FLAG(bool, create_impl, true, "Whether to create the .cc file.");
ABSL_FLAG(bool, eliminate_duplication, true,
          "Whether to share a single copy of the data between TOC entries that "
          "have the same size and md5digest.");
ABSL_FLAG(bool, flatten, false, "Strip all directories from file names.");
ABSL_FLAG(bool, redact_filename, false, "Whether to strip filenames.");
ABSL_FLAG(bool, sort_toc, false, "Whether to sort the TOC entries by name.");

ABSL_FLAG(std::string, include_path, "", "Path to use when writing #include.");
ABSL_FLAG(std::string, namespace, "",
          "C++ namespace to wrap the TOC entries in.");
ABSL_FLAG(std::string, out_cc, "", "Filename for the .cc file.");
ABSL_FLAG(std::string, out_h, "", "Filename for the .h file.");
ABSL_FLAG(std::string, toc_section_name, "filewrapper_toc",
          "Put generated TOC into the given ELF section.");
ABSL_FLAG(std::string, strip, "", "Leading prefix to strip off file names.");

namespace {

void FatalError(const char* fmt, ...) {
  fprintf(stderr, "%s: ", "filewrapper");
  std::va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  exit(1);
}

// Determine the "runfiles" directory from argv[0].
std::string RunFiles(int argc, char** argv) {
  if (argc <= 0) return std::string();

  std::string runfiles = argv[0];
  const std::string suffix = ".runfiles/";
  std::string::size_type pos = runfiles.rfind(suffix);
  if (pos != std::string::npos) {
    return runfiles.substr(0, pos + suffix.size());
  }
  pos = runfiles.find_last_of('.');
  if (pos != std::string::npos) {
    if (runfiles.find_first_of('/', pos) == std::string::npos) {
      runfiles.erase(pos);
    }
  }
  runfiles += suffix;
  return runfiles;
}

// Expands any directories in the input list to the files they contain.
absl::StatusOr<std::vector<std::string>> ExpandDirs(
    tsl::Env& env, const std::string& runfiles,
    const std::vector<std::string>& infiles) {
  std::vector<std::string> allfiles;
  std::vector<std::string> to_process = infiles;
  while (!to_process.empty()) {
    std::string filename = to_process.back();
    const absl::Status s = env.IsDirectory(filename);
    if (s.ok() && !absl::GetFlag(FLAGS_allow_dir)) {
      LOG(FATAL) << "filewrapper: refusing to process dir '" << filename << "'";
    } else if (s.ok()) {
      TF_RETURN_IF_ERROR(env.GetChildren(filename, &to_process));
    } else if (absl::IsFailedPrecondition(s)) {
      allfiles.push_back(filename);
    }
    // Other errors are intentionally swallowed since one of the inputs here is
    // the name, not a directory.

    to_process.pop_back();
  }

  return allfiles;
}

// A short, escaped representation of a character.  We choose octal
// escapes as they always end after three characters.
std::string Escape(unsigned char c) {
  static const char kDigits[] = "01234567";
  std::string::value_type buf[sizeof "\\377"];
  std::string::value_type* ep = buf + sizeof buf;
  std::string::value_type* p = ep;
  switch (c) {
    case '"':
    case '?':
    case '\\':
      *--p = c;
      break;
    case '\a':
      *--p = 'a';
      break;
    case '\b':
      *--p = 'b';
      break;
    case '\f':
      *--p = 'f';
      break;
    case '\n':
      *--p = 'n';
      break;
    case '\r':
      *--p = 'r';
      break;
    case '\v':
      *--p = 'v';
      break;
    case '\t':
      *--p = 't';
      break;
    default:
      *--p = kDigits[c & 7];
      if ((c >>= 3) != 0) {
        *--p = kDigits[c & 7];
        if ((c >>= 3) != 0) {
          *--p = kDigits[c & 3];
        }
      }
      break;
  }
  *--p = '\\';
  return std::string(p, ep - p);
}

std::string TrimFront(const std::string& s, char c) {
  std::string::size_type pos = 0;
  while (pos < s.size() && s[pos] == c) ++pos;
  return s.substr(pos);
}

std::string TrimBack(const std::string& s, char c) {
  std::string::size_type pos = s.size();
  while (pos > 0 && s[pos - 1] == c) --pos;
  return s.substr(0, pos);
}

// Strip out non-alphanumeric characters, replacing them with underscore.
std::string ToCIdentifier(const std::string& s) {
  std::string symbol = s;
  for (auto& c : symbol)
    if (!isalnum(c)) c = '_';
  return symbol;
}

std::vector<std::string> Split(const std::string& str, const std::string& sep) {
  std::vector<std::string> result;
  std::string::size_type pos = 0;
  for (;;) {
    std::string::size_type end = str.find_first_of(sep, pos);
    std::string::size_type len = (end == std::string::npos ? end : end - pos);
    result.push_back(str.substr(pos, len));
    if (end == std::string::npos) break;
    pos = end + sep.size();
  }
  return result;
}

// Generates (possibly nested) namespace wrapping.
std::pair<std::string, std::string> GetNamespaces() {
  std::string intro;
  std::string outro;
  std::string ns = absl::GetFlag(FLAGS_namespace);
  if (!ns.empty()) {
    for (const auto& ns : Split(ns, "::")) {
      intro = intro + "namespace " + ns + " {\n";
      outro = "}  // namespace " + ns + "\n" + outro;
    }
    intro = intro + "\n";
    outro = "\n" + outro;
  }
  return std::make_pair(intro, outro);
}

// Information about an encapsulated file.
struct Initializer {
  Initializer(std::string f, std::string s, std::streamoff sz, llvm::MD5 digest)
      : filename(absl::GetFlag(FLAGS_redact_filename) ? "" : std::move(f)),
        sym(std::move(s)),
        size(sz) {
    llvm::MD5::MD5Result result = digest.final();
    CHECK_EQ(result.max_size(), 16)
        << "MD5 digest size must be 16 bytes (not hex)";

    memcpy(md5digest, result.data(), result.size());
  }
  std::string filename;
  std::string sym;
  std::streamoff size;
  unsigned char md5digest[16];
};

absl::string_view Md5DigestAsSV(const Initializer& initializer) {
  return absl::string_view(reinterpret_cast<const char*>(initializer.md5digest),
                           sizeof(initializer.md5digest));
}

// Checks if there is a previous symbol with the same size/md5digest.
// Returns a pointer to the symbol of a previously found equivalent
// initializer, or nullptr if none is found.
const std::string* PreviousEquivalentSymbol(
    const Initializer& new_initializer,
    const absl::flat_hash_map<absl::string_view, const Initializer*>&
        md5_to_initializer) {
  if (absl::GetFlag(FLAGS_eliminate_duplication)) {
    if (auto it = md5_to_initializer.find(Md5DigestAsSV(new_initializer));
        it != md5_to_initializer.end() &&
        new_initializer.size == it->second->size) {
      return &it->second->sym;
    }
  }
  return nullptr;  // no match
}

// "what-was-done" comment written at the top of each file.
std::string Comment(const std::string& base) {
  std::string comment;
  comment = "//  Automatically generated by filewrapper\n";
  comment += "//    " + base + "\n";
  if (absl::GetFlag(FLAGS_flatten)) {
    comment += "//    --flatten\n";
  }
  if (const std::string strip = absl::GetFlag(FLAGS_strip); !strip.empty()) {
    comment += "//    --strip " + strip + "\n";
  }
  return comment;
}

// Generates a header guard for the TOC factory.
std::string GetHeaderGuard(const std::string& base) {
  std::string guard;
  std::string ns = absl::GetFlag(FLAGS_namespace);
  if (!ns.empty()) {
    for (const auto& ns : Split(ns, "::")) {
      guard += ns;
      guard += "_";
    }
  }
  guard += base;
  return guard;
}

// The comments here are load-bearing for Google <-> OSS transforms; do not
// remove or edit them unless you have access to the Google transformation
// configurations. They must differ from each other, including not being
// prefixes, to ensure the transformations are reversible.
static constexpr absl::string_view kSourceRootPath =
    "xla/tsl/util/file_toc.h";  // copybara substituted

static constexpr absl::string_view kIncludePath =
    "xla/tsl/util/file_toc.h";  // copybara replaced

// Writes the table of contents .h file.
void WriteHeader(const std::string& filename, const std::string& comment,
                 const std::pair<std::string, std::string>& namespaces,
                 const std::string& base, const std::string& runfiles) {
  std::ofstream hdr(filename, std::ios_base::out | std::ios_base::trunc);
  if (!hdr.is_open()) {
    FatalError("Unable to open header file '%s' for writing", filename.c_str());
  }

  hdr << comment << "//  Output: " << filename << "\n\n";

  std::ifstream toc(tsl::io::JoinPath(runfiles, kSourceRootPath));
  if (toc.is_open()) {
    std::string line;
    while (std::getline(toc, line)) {
      hdr << line << "\n";
    }
    toc.close();
  } else {
    hdr << "#include \"" << kIncludePath << "\"";
  }
  hdr << "\n";

  const std::string guard = "__STRUCT_FILE_TOC_" + GetHeaderGuard(base) + "_";
  hdr << "#ifndef " << guard << "\n";
  hdr << "#define " << guard << "\n";
  hdr << "\n" << namespaces.first;
  hdr << "const struct FileToc* " << base << "_create();\n";
  hdr << "size_t " << base << "_size();\n";
  hdr << namespaces.second << "\n";
  hdr << "#endif  // " << guard << "\n";
  hdr.close();
  if (hdr.fail()) {
    FatalError("Error during header creation");
  }
}

// Embeds each file into the .cc file as string literal.
std::vector<Initializer> EmbedFiles(const std::vector<std::string>& infiles,
                                    const std::string& cc_name,
                                    std::ofstream& f_cc) {
  // For each input file we create an array named dataX (where X is a sequence
  // number starting at 0) in an anonymous namespace.
  //
  // Although the worst-case expansion of the data is 316% (for a sequence of
  // entirely high-bit chars), the size expansion for random data is a more
  // modest 162%, and text (or mostly text) will be almost unexpanded.
  std::vector<Initializer> initializers;
  std::size_t seq = 0;

  // This map will point into `initializers`.
  absl::flat_hash_map<absl::string_view, const Initializer*> md5_to_initializer;

  // We store pointers into `initializers` in `md5_to_initializer` map.
  // Prevent reallocation.
  initializers.reserve(infiles.size());

  // Copied from base/port.h and renamed.
  f_cc << "#if defined(COMPILER_MSVC)\n";
  f_cc << "#define ALIGN_ATTRIBUTE(X) __declspec(align(X))\n";
  f_cc << "#elif defined(__GNUC__) || defined(COMPILER_ICC)\n";
  f_cc << "#define ALIGN_ATTRIBUTE(X) __attribute__((aligned(X)))\n";
  f_cc << "#endif\n";

  f_cc << "namespace {\n";

  for (const auto& filename : infiles) {
    // Embed the input file into the .cc file.
    std::streamoff size = 0;
    std::ifstream f_in(filename, std::ios_base::in | std::ios_base::binary);
    if (!f_in.is_open()) {
      FatalError("Unable to open input file '%s'", filename.c_str());
    }

    const std::streampos offset = f_cc.tellp();

    const std::string seq_str = std::to_string(seq++);
    const std::string symbol =
        ToCIdentifier("filewrapper_" + seq_str + "_" + filename);

    f_cc << "ALIGN_ATTRIBUTE(" << absl::GetFlag(FLAGS_align) << ") "
         << "const char " << symbol << "[] =\n";
    std::string pending_line = "\"";
    bool esc_digit = false;

    llvm::MD5 digest;

    const std::size_t kBufSize = 4096;
    std::unique_ptr<unsigned char[]> buf(new unsigned char[kBufSize]);
    auto rbuf = reinterpret_cast<char*>(buf.get());
    for (;;) {
      f_in.read(rbuf, kBufSize);
      const std::streamsize cc = f_in.gcount();
      if (cc == 0) break;
      for (std::streamsize i = 0; i < cc; ++i) {
        unsigned char c = rbuf[i];
        std::string rep(1, c);  // default to self
        if (!isprint(c) || c == '"' || c == '?' || c == '\\' ||
            (isdigit(c) && esc_digit)) {
          rep = Escape(c);  // "\0" through "\377"
          esc_digit = (rep.size() < 4 && isdigit(rep.back()));
        }
        if (pending_line.size() + rep.size() > 79) {
          f_cc << pending_line << "\"\n";
          pending_line = "\"";
        }
        pending_line += rep;
      }
      llvm::ArrayRef<unsigned char> ref(buf.get(), cc);
      digest.update(ref);
      size += cc;
    }
    f_cc << pending_line << "\";\n";

    f_in.close();
    if (!f_in.eof()) {
      FatalError("Unable to read input file '%s'", filename.c_str());
    }

    Initializer initializer(filename, symbol, size, digest);

    // Drop this symbol in favor of any previous equivalent one.
    const std::string* prev =
        PreviousEquivalentSymbol(initializer, md5_to_initializer);
    if (prev != nullptr) {
      f_cc.seekp(offset);
      initializer.sym = *prev;
    }

    initializers.push_back(std::move(initializer));

    const Initializer& last_initializer = initializers.back();
    md5_to_initializer[Md5DigestAsSV(last_initializer)] = &last_initializer;
  }

  f_cc << "}  // namespace\n";
  return initializers;
}

// Writes the .cc file.
absl::Status WriteCpp(tsl::Env* env, const std::string& cc_filename,
                      const std::string& comment,
                      const std::pair<std::string, std::string>& namespaces,
                      const std::string& base,
                      const std::vector<std::string>& externs,
                      std::vector<Initializer>& initializers,
                      const std::vector<std::string>& infiles) {
  std::ofstream toc(cc_filename, std::ios_base::out | std::ios_base::trunc);
  if (!toc.is_open()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "filewrapper: unable to open file '%s' for writing", cc_filename));
  }

  toc << comment << "//  Output: " << cc_filename << "\n\n";
  toc << "#include \"";
  std::string include_path = absl::GetFlag(FLAGS_include_path);
  if (!include_path.empty() && (base.empty() || base[0] != '/')) {
    toc << include_path;
    if (include_path.back() != '/') {
      toc << '/';
    }
  }
  toc << base << ".h\"\n\n";

  initializers = EmbedFiles(infiles, cc_filename, toc);
  toc << "\n";

  if (absl::GetFlag(FLAGS_sort_toc)) {
    std::sort(initializers.begin(), initializers.end(),
              [](const Initializer& a, const Initializer& b) {
                return a.filename < b.filename;
              });
  }

  // Ensure that the prefix ends, but does not start, with a slash.
  // This allows both "//third_party/some/path" and "subdir/" to work.
  std::vector<std::string> prefixes;
  std::string strip = absl::GetFlag(FLAGS_strip);
  if (!strip.empty()) {
    prefixes.push_back(TrimFront(TrimBack(strip, '/'), '/') + "/");
  }

  toc << "static const struct FileToc toc[";
  toc << initializers.size() + 1;  // one more for sentinel
  toc << "] = {\n";
  for (const auto& initializer : initializers) {
    std::string filename = initializer.filename;
    if (absl::GetFlag(FLAGS_flatten)) {
      filename = tsl::io::Basename(filename);
    } else {
      for (const auto& prefix : prefixes) {
        if (prefix.size() <= filename.size()) {
          if (filename.compare(0, prefix.size(), prefix) == 0) {
            filename = filename.substr(prefix.size());
          }
        }
      }
    }
    toc << "  { ";
    toc << "\"" << filename << "\", ";
    toc << (initializer.size ? initializer.sym : "\"\"") << ", ";
    toc << initializer.size << ", {";
    const std::ios_base::fmtflags ff = toc.flags(std::ios_base::hex);
    const char fill = toc.fill('0');
    for (std::size_t i = 0; i < sizeof initializer.md5digest; ++i) {
      if (i != 0) toc << ",";
      toc << " 0x" << std::setw(2)
          << static_cast<int>(initializer.md5digest[i]);
    }
    toc.fill(fill);
    toc.flags(ff);
    toc << " } },\n";
  }
  toc << "  { (const char*) 0, (const char*) 0, 0, {} }\n";
  toc << "};\n\n";

  std::string toc_section_name = absl::GetFlag(FLAGS_toc_section_name);
  if (!toc_section_name.empty()) {
    // The named section is only supported on ELF platforms, with GCC or Clang
    // and if not explicitly disabled via the DISABLE_FILEWRAPPER_TOC_SECTION.
    toc << "#if defined(__ELF__) && defined(__GNUC__) && "
        << " !defined(DISABLE_FILEWRAPPER_TOC_SECTION)\n"
        << "  #define ATTRIBUTE_SECTION(name) "
        << "    __attribute__((section(#name), used))\n"
        << "#endif\n"
        << "#ifndef ATTRIBUTE_SECTION\n"
        << "  #define ATTRIBUTE_SECTION(name) /**/\n"
        << "#endif\n\n";
  }
  toc << "static const struct FileToc* toc_ptr ";
  if (!toc_section_name.empty()) {
    toc << "ATTRIBUTE_SECTION(" << toc_section_name << ") ";
  }
  toc << "= toc;\n\n";

  toc << namespaces.first;
  toc << "const struct FileToc* " << base << "_create() {\n";
  toc << "  return toc_ptr;\n";
  toc << "}\n";
  toc << "\n";
  toc << "size_t " << base << "_size() {\n";
  toc << "  return " << initializers.size() << ";\n";
  toc << "}\n";
  toc << namespaces.second;

  const std::streampos end_pos = toc.tellp();
  toc.close();
  if (toc.fail()) {
    return absl::InvalidArgumentError("filewrapper: Error during cc creation");
  }

  if (absl::GetFlag(FLAGS_eliminate_duplication)) {
    // If we did a backwards seek in EmbedFiles() we may have written data past
    // end_pos, so we truncate now just in case we did. Unfortunately there's no
    // simple cross-platform way to truncate files, so we just read and write
    // again.
    std::string contents;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, cc_filename, &contents));
    contents.resize(end_pos);
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, cc_filename, contents));
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  // Internally InitMain will remove leave us with positional args, but
  // externally it won't so we need to explicitly parse the command line.
  tsl::port::InitMain("", &argc, &argv);
  const std::string runfiles = RunFiles(argc, argv);

  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  // positional_args[0] is still the binary name.
  if (positional_args.size() < 2) {
    fprintf(stderr, "%s\n", "filewrapper: At least one file is required!");
    return 1;
  }

  // The base name for the files, functions and data structures.
  const std::string base = positional_args[1];
  auto it = positional_args.begin();
  it++;

  // The files to encapsulate.
  std::vector<std::string> infiles;
  for (; it != positional_args.end(); ++it) {
    infiles.push_back(std::string(*it));
  }

  // Compute the final destinations for the files.
  std::string hdr_name = absl::GetFlag(FLAGS_out_h);
  if (hdr_name.empty()) {
    hdr_name = base + ".h";
  }

  std::string src_name = absl::GetFlag(FLAGS_out_cc);
  if (src_name.empty()) {
    src_name = base + ".cc";
  }

  std::vector<Initializer> initializers;  // info for encapsulated files
  std::vector<std::string> externs;       // extern declarations

  if (absl::GetFlag(FLAGS_create_impl)) {
    absl::StatusOr<std::vector<std::string>> expanded =
        ExpandDirs(*tsl::Env::Default(), runfiles, infiles);
    QCHECK_OK(expanded);
    infiles = *std::move(expanded);
  }

  const std::string comment = Comment(base);
  const std::pair<std::string, std::string> namespaces = GetNamespaces();

  if (absl::GetFlag(FLAGS_create_header)) {
    WriteHeader(hdr_name, comment, namespaces, base, runfiles);
  }
  if (absl::GetFlag(FLAGS_create_impl)) {
    QCHECK_OK(WriteCpp(tsl::Env::Default(), src_name, comment, namespaces, base,
                       externs, initializers, infiles));
  }

  return 0;
}
