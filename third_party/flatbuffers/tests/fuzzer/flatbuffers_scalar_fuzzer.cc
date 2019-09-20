#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <clocale>
#include <memory>
#include <regex>
#include <string>

#include "flatbuffers/idl.h"
#include "test_init.h"

static constexpr uint8_t flags_scalar_type = 0x0F;  // type of scalar value
static constexpr uint8_t flags_quotes_kind = 0x10;  // quote " or '
// reserved for future: json {named} or [unnamed]
// static constexpr uint8_t flags_json_bracer = 0x20;

// Find all 'subj' sub-strings and replace first character of sub-string.
// BreakSequence("testest","tes", 'X') -> "XesXest".
// BreakSequence("xxx","xx", 'Y') -> "YYx".
static void BreakSequence(std::string &s, const char *subj, char repl) {
  size_t pos = 0;
  while (pos = s.find(subj, pos), pos != std::string::npos) {
    s.at(pos) = repl;
    pos++;
  }
}

// Remove all leading and trailing symbols matched with pattern set.
// StripString("xy{xy}y", "xy") -> "{xy}"
static std::string StripString(const std::string &s, const char *pattern,
                               size_t *pos = nullptr) {
  if (pos) *pos = 0;
  // leading
  auto first = s.find_first_not_of(pattern);
  if (std::string::npos == first) return "";
  if (pos) *pos = first;
  // trailing
  auto last = s.find_last_not_of(pattern);
  assert(last < s.length());
  assert(first <= last);
  return s.substr(first, last - first + 1);
}

class RegexMatcher {
 protected:
  virtual bool MatchNumber(const std::string &input) const = 0;

 public:
  virtual ~RegexMatcher() = default;

  struct MatchResult {
    size_t pos{ 0 };
    size_t len{ 0 };
    bool res{ false };
    bool quoted{ false };
  };

  MatchResult Match(const std::string &input) const {
    MatchResult r;
    // strip leading and trailing "spaces" accepted by flatbuffer
    auto test = StripString(input, "\t\r\n ", &r.pos);
    r.len = test.size();
    // check quotes
    if (test.size() >= 2) {
      auto fch = test.front();
      auto lch = test.back();
      r.quoted = (fch == lch) && (fch == '\'' || fch == '\"');
      if (r.quoted) {
        // remove quotes for regex test
        test = test.substr(1, test.size() - 2);
      }
    }
    // Fast check:
    if (test.empty()) return r;
    // A string with a valid scalar shouldn't have non-ascii or non-printable
    // symbols.
    for (auto c : test) {
      if ((c < ' ') || (c > '~')) return r;
    }
    // Check with regex
    r.res = MatchNumber(test);
    return r;
  }

  bool MatchRegexList(const std::string &input,
                      const std::vector<std::regex> &re_list) const {
    auto str = StripString(input, " ");
    if (str.empty()) return false;
    for (auto &re : re_list) {
      std::smatch match;
      if (std::regex_match(str, match, re)) return true;
    }
    return false;
  }
};

class IntegerRegex : public RegexMatcher {
 protected:
  bool MatchNumber(const std::string &input) const override {
    static const std::vector<std::regex> re_list = {
      std::regex{ R"(^[-+]?[0-9]+$)", std::regex_constants::optimize },

      std::regex{
          R"(^[-+]?0[xX][0-9a-fA-F]+$)", std::regex_constants::optimize }
    };
    return MatchRegexList(input, re_list);
  }

 public:
  IntegerRegex() = default;
  virtual ~IntegerRegex() = default;
};

class UIntegerRegex : public RegexMatcher {
 protected:
  bool MatchNumber(const std::string &input) const override {
    static const std::vector<std::regex> re_list = {
      std::regex{ R"(^[+]?[0-9]+$)", std::regex_constants::optimize },
      std::regex{
          R"(^[+]?0[xX][0-9a-fA-F]+$)", std::regex_constants::optimize },
      // accept -0 number
      std::regex{ R"(^[-](?:0[xX])?0+$)", std::regex_constants::optimize }
    };
    return MatchRegexList(input, re_list);
  }

 public:
  UIntegerRegex() = default;
  virtual ~UIntegerRegex() = default;
};

class BooleanRegex : public IntegerRegex {
 protected:
  bool MatchNumber(const std::string &input) const override {
    if (input == "true" || input == "false") return true;
    return IntegerRegex::MatchNumber(input);
  }

 public:
  BooleanRegex() = default;
  virtual ~BooleanRegex() = default;
};

class FloatRegex : public RegexMatcher {
 protected:
  bool MatchNumber(const std::string &input) const override {
    static const std::vector<std::regex> re_list = {
      // hex-float
      std::regex{
          R"(^[-+]?0[xX](?:(?:[.][0-9a-fA-F]+)|(?:[0-9a-fA-F]+[.][0-9a-fA-F]*)|(?:[0-9a-fA-F]+))[pP][-+]?[0-9]+$)",
          std::regex_constants::optimize },
      // dec-float
      std::regex{
          R"(^[-+]?(?:(?:[.][0-9]+)|(?:[0-9]+[.][0-9]*)|(?:[0-9]+))(?:[eE][-+]?[0-9]+)?$)",
          std::regex_constants::optimize },

      std::regex{ R"(^[-+]?(?:nan|inf|infinity)$)",
                  std::regex_constants::optimize | std::regex_constants::icase }
    };
    return MatchRegexList(input, re_list);
  }

 public:
  FloatRegex() = default;
  virtual ~FloatRegex() = default;
};

class ScalarReferenceResult {
 private:
  ScalarReferenceResult(const char *_type, RegexMatcher::MatchResult _matched)
      : type(_type), matched(_matched) {}

 public:
  // Decode scalar type and check if the input string satisfies the scalar type.
  static ScalarReferenceResult Check(uint8_t code, const std::string &input) {
    switch (code) {
      case 0x0: return { "double", FloatRegex().Match(input) };
      case 0x1: return { "float", FloatRegex().Match(input) };
      case 0x2: return { "int8", IntegerRegex().Match(input) };
      case 0x3: return { "int16", IntegerRegex().Match(input) };
      case 0x4: return { "int32", IntegerRegex().Match(input) };
      case 0x5: return { "int64", IntegerRegex().Match(input) };
      case 0x6: return { "uint8", UIntegerRegex().Match(input) };
      case 0x7: return { "uint16", UIntegerRegex().Match(input) };
      case 0x8: return { "uint32", UIntegerRegex().Match(input) };
      case 0x9: return { "uint64", UIntegerRegex().Match(input) };
      case 0xA: return { "bool", BooleanRegex().Match(input) };
      default: return { "float", FloatRegex().Match(input) };
    };
  }

  const char *type;
  const RegexMatcher::MatchResult matched;
};

bool Parse(flatbuffers::Parser &parser, const std::string &json,
           std::string *_text) {
  auto done = parser.Parse(json.c_str());
  if (done) {
    TEST_EQ(GenerateText(parser, parser.builder_.GetBufferPointer(), _text),
            true);
  } else {
    *_text = parser.error_;
  }
  return done;
}

// Utility for test run.
OneTimeTestInit OneTimeTestInit::one_time_init_;

// llvm std::regex have problem with stack overflow, limit maximum length.
// ./scalar_fuzzer -max_len=3000
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Reserve one byte for Parser flags and one byte for repetition counter.
  if (size < 3) return 0;
  const uint8_t flags = data[0];
  // normalize to ascii alphabet
  const int extra_rep_number = data[1] >= '0' ? (data[1] - '0') : 0;
  data += 2;
  size -= 2;  // bypass

  // Guarantee 0-termination.
  const std::string original(reinterpret_cast<const char *>(data), size);
  auto input = std::string(original.c_str());  // until '\0'
  if (input.empty()) return 0;

  // Break comments in json to avoid complexity with regex matcher.
  // The string " 12345 /* text */" will be accepted if insert it to string
  // expression: "table X { Y: " + " 12345 /* text */" + "; }.
  // But strings like this will complicate regex matcher.
  // We reject this by transform "/* text */ 12345" to "@* text */ 12345".
  BreakSequence(input, "//", '@');  // "//" -> "@/"
  BreakSequence(input, "/*", '@');  // "/*" -> "@*"
  // Break all known scalar functions (todo: add them to regex?):
  for (auto f : { "deg", "rad", "sin", "cos", "tan", "asin", "acos", "atan" }) {
    BreakSequence(input, f, '_');  // ident -> ident
  }

  // Extract type of scalar from 'flags' and check if the input string satisfies
  // the scalar type.
  const auto ref_res =
      ScalarReferenceResult::Check(flags & flags_scalar_type, input);
  auto &recheck = ref_res.matched;

  // Create parser
  flatbuffers::IDLOptions opts;
  opts.force_defaults = true;
  opts.output_default_scalars_in_json = true;
  opts.indent_step = -1;
  opts.strict_json = true;

  flatbuffers::Parser parser(opts);
  auto schema =
      "table X { Y: " + std::string(ref_res.type) + "; } root_type X;";
  TEST_EQ_FUNC(parser.Parse(schema.c_str()), true);

  // The fuzzer can adjust the number repetition if a side-effects have found.
  // Each test should pass at least two times to ensure that the parser doesn't
  // have any hidden-states or locale-depended effects.
  for (auto cnt = 0; cnt < (extra_rep_number + 2); cnt++) {
    // Each even run (0,2,4..) will test locale independed code.
    auto use_locale = !!OneTimeTestInit::test_locale() && (0 == (cnt % 2));
    // Set new locale.
    if (use_locale) {
      FLATBUFFERS_ASSERT(setlocale(LC_ALL, OneTimeTestInit::test_locale()));
    }

    // Parse original input as-is.
    auto orig_scalar = "{ \"Y\" : " + input + " }";
    std::string orig_back;
    auto orig_done = Parse(parser, orig_scalar, &orig_back);

    if (recheck.res != orig_done) {
      // look for "does not fit" or "doesn't fit" or "out of range"
      auto not_fit =
          (true == recheck.res)
              ? ((orig_back.find("does not fit") != std::string::npos) ||
                 (orig_back.find("out of range") != std::string::npos))
              : false;

      if (false == not_fit) {
        TEST_OUTPUT_LINE("Stage 1 failed: Parser(%d) != Regex(%d)", orig_done,
                         recheck.res);
        TEST_EQ_STR(orig_back.c_str(),
                    input.substr(recheck.pos, recheck.len).c_str());
        TEST_EQ_FUNC(orig_done, recheck.res);
      }
    }

    // Try to make quoted string and test it.
    std::string qouted_input;
    if (true == recheck.quoted) {
      // we can't simply remove quotes, they may be nested "'12'".
      // Original string "\'12\'" converted to "'12'".
      // The string can be an invalid string by JSON rules, but after quotes
      // removed can transform to valid.
      assert(recheck.len >= 2);
    } else {
      const auto quote = (flags & flags_quotes_kind) ? '\"' : '\'';
      qouted_input = input;  // copy
      qouted_input.insert(recheck.pos + recheck.len, 1, quote);
      qouted_input.insert(recheck.pos, 1, quote);
    }

    // Test quoted version of the string
    if (!qouted_input.empty()) {
      auto fix_scalar = "{ \"Y\" : " + qouted_input + " }";
      std::string fix_back;
      auto fix_done = Parse(parser, fix_scalar, &fix_back);

      if (orig_done != fix_done) {
        TEST_OUTPUT_LINE("Stage 2 failed: Parser(%d) != Regex(%d)", fix_done,
                         orig_done);
        TEST_EQ_STR(fix_back.c_str(), orig_back.c_str());
      }
      if (orig_done) { TEST_EQ_STR(fix_back.c_str(), orig_back.c_str()); }
      TEST_EQ_FUNC(fix_done, orig_done);
    }

    // Create new parser and test default value
    if (true == orig_done) {
      flatbuffers::Parser def_parser(opts);  // re-use options
      auto def_schema = "table X { Y: " + std::string(ref_res.type) + " = " +
                        input + "; } root_type X;" +
                        "{}";  // <- with empty json {}!

      auto def_done = def_parser.Parse(def_schema.c_str());
      if (false == def_done) {
        TEST_OUTPUT_LINE("Stage 3.1 failed with _error = %s",
                         def_parser.error_.c_str());
        FLATBUFFERS_ASSERT(false);
      }
      // Compare with print.
      std::string ref_string, def_string;
      FLATBUFFERS_ASSERT(GenerateText(
          parser, parser.builder_.GetBufferPointer(), &ref_string));
      FLATBUFFERS_ASSERT(GenerateText(
          def_parser, def_parser.builder_.GetBufferPointer(), &def_string));
      if (ref_string != def_string) {
        TEST_OUTPUT_LINE("Stage 3.2 failed: '%s' != '%s'", def_string.c_str(),
                         ref_string.c_str());
        FLATBUFFERS_ASSERT(false);
      }
    }

    // Restore locale.
    if (use_locale) { FLATBUFFERS_ASSERT(setlocale(LC_ALL, "C")); }
  }
  return 0;
}
