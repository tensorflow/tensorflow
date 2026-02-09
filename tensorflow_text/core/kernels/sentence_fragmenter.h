// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A class to split up a document into sentence fragments. A sentence
// fragment is a token sequence whose end is potentially an end-of-sentence.
//
// Example:
//
//   Document text:
//     John said, "I.B.M. went up 5 points today."
//
//   SentenceFragments:
//     (1) John said, "I.B.M.
//     (2) went up 5 points today."
//
// Fragment boundaries are induced by punctuation and paragraph breaks.

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_text/core/kernels/sentence_breaking_utils.h"

namespace tensorflow {
namespace text {

class Token {
 public:
  enum BreakLevel {
    NO_BREAK = 0,         // No separation between tokens.
    SPACE_BREAK = 1,      // Tokens separated by space.
    LINE_BREAK = 2,       // Tokens separated by line break.
    SENTENCE_BREAK = 3,   // Tokens separated by sentence break.
    PARAGRAPH_BREAK = 4,  // Tokens separated by paragraph break.
    SECTION_BREAK = 10,   // Tokens separated by section break.
    CHAPTER_BREAK = 20,   // Tokens separated by chapter break.
  };

  // Bitmask for properties of the token text.
  enum TextProperty {
    NONE = 0x00,

    // Token is ill-formed if:
    //
    // All tokens in a paragraph are marked as ill-formed if it has too few
    // non-punctuation tokens in a paragraph (currently, a heading must have
    // at least 2 tokens, and a non-heading must have at least 8).
    //
    // All tokens in a paragraph are marked as ill-formed if it lacks terminal
    // sentence ending punctuation(e.g.: . ! ? …) or an emoticon (e.g.: ':)',
    // ':D').
    // Exception: If a paragraph ends in an introductory punctuation
    // character (','':' ';'), we say that it is an introductory paragraph.
    // If it is followed by a "simple" HTML list (one whose list items have
    // no substructure, such as embedded tables), then we keep both the
    // introductory paragraph and the entire list. If not, we keep the
    // introductory paragraph if it is followed by a well-formed paragraph.
    //
    // All tokens in a paragraph are marked as ill-formed if it contains the
    // copyright sign (C in a circle) as this usually indicates a copyright
    // notice, and is therefore effectively boilerplate.
    ILL_FORMED = 0x01,

    // Indicates that the token is a part of the page title (<title> tag) or
    // a heading (<hN> tag).
    TITLE = 0x40,
    HEADING = 0x02,

    // Text style. Determined from HTML tags only (<b>, etc), not from CSS.
    BOLD = 0x04,
    ITALIC = 0x08,
    UNDERLINED = 0x10,

    // Indicates that the token is a part of a list. Currently set only for
    // "simple" HTML lists (have no embedded paragraph boundaries) that are
    // preceded by an introductory paragraph (ends in colon or a few other
    // characters).
    LIST = 0x20,

    // Token is an emoticon.
    EMOTICON = 0x80,

    // Token was identified by Lexer as an acronym.  Lexer identifies period-,
    // hyphen-, and space-separated acronyms: "U.S.", "U-S", and "U S".
    // Lexer normalizes all three to "US", but the token.word field
    // normalizes only space-separated acronyms.
    ACRONYM = 0x100,

    // Indicates that the token (or part of the token) is a covered by at
    // least one hyperlink. More information of the hyperlink is stored in the
    // first token covered by the hyperlink.
    HYPERLINK = 0x200,
  };

  Token(const tstring &word, uint32 start, uint32 end, BreakLevel break_level,
        TextProperty text_properties)
      : word_(word),
        start_(start),
        end_(end),
        break_level_(break_level),
        text_properties_(text_properties) {}

  const tstring &word() const { return word_; }
  const uint32 start() const { return start_; }
  const uint32 end() const { return end_; }
  const BreakLevel break_level() const { return break_level_; }
  const TextProperty text_properties() const { return text_properties_; }

 private:
  const tstring &word_;
  uint32 start_;
  uint32 end_;
  BreakLevel break_level_;
  TextProperty text_properties_;
};

class Document {
 public:
  // Does NOT take ownership of 'tokens'.
  Document(std::vector<Token> *tokens) : tokens_(tokens) {}

  void AddToken(const tstring &word, uint32 start, uint32 end,
                Token::BreakLevel break_level,
                Token::TextProperty text_properties) {
    tokens_->emplace_back(word, start, end, break_level, text_properties);
  }

  const std::vector<Token> &tokens() const { return *tokens_; }

 private:
  // not owned
  std::vector<Token> *tokens_;
};

struct SentenceFragment {
  int start;
  int limit;

  enum Property {
    TERMINAL_PUNC = 0x0001,               // ends with terminal punctuation
    MULTIPLE_TERMINAL_PUNC = 0x0002,      // e.g.: She said what?!
    HAS_CLOSE_PAREN = 0x0004,             // e.g.: Mushrooms (they're fungi!!)
    HAS_SENTENTIAL_CLOSE_PAREN = 0x0008,  // e.g.: (Mushrooms are fungi!)
  };
  // A mask of the above listed properties.
  uint32 properties = 0;
  int terminal_punc_token = -1;
};

// Utility class for splitting documents into a list of sentence fragments.
class SentenceFragmenter {
 public:
  // Constructs a fragmenter to process a specific part of a document.
  SentenceFragmenter(const Document *document, UnicodeUtil *util)
      : document_(document), util_(util) {}

  // Finds sentence fragments in the [start_, limit_) range of the associated
  // document.
  absl::Status FindFragments(std::vector<SentenceFragment> *result);

 private:
  // State for matching a fragment-boundary regexp against a token sequence.
  // The regexp is: terminal_punc+ close_punc*.
  class FragmentBoundaryMatch;

  // Matches a fragment-boundary regexp against the tokens starting at
  // 'i_start'. Returns the longest match found; will be non-empty as long as
  // 'i_start' was not already at the end of the associated token range.
  absl::Status FindNextFragmentBoundary(int i_start,
                                        FragmentBoundaryMatch *result) const;

  // Updates 'latest_open_paren_is_sentential_' for the tokens in the given
  // fragment.
  absl::Status UpdateLatestOpenParenForFragment(int i_start, int i_end);

  // Populates a sentence fragment with the tokens from 'i_start' to the end
  // of the given FragmentBoundaryMatch.
  absl::Status FillInFragmentFields(int i_start,
                                    const FragmentBoundaryMatch &match,
                                    SentenceFragment *fragment) const;

  // Returns the adjusted first terminal punctuation index in a
  // FragmentBoundaryMatch.
  absl::Status GetAdjustedFirstTerminalPuncIndex(
      const FragmentBoundaryMatch &match, int *result) const;

  // Returns true iff a FragmentBoundaryMatch has an "unattachable" terminal
  // punctuation mark.
  absl::Status HasUnattachableTerminalPunc(const FragmentBoundaryMatch &match,
                                           bool *result) const;

  // Returns true iff a FragmentBoundaryMatch has a close paren in its closing
  // punctuation.
  absl::Status HasCloseParen(const FragmentBoundaryMatch &match,
                             bool *result) const;

  // Whether the latest open paren seen so far appears to be sentence-initial.
  // See UpdateLatestOpenParenForFragment() in the .cc file for details.
  bool latest_open_paren_is_sentential_ = false;

  const Document *document_ = nullptr;  // not owned
  UnicodeUtil *util_ = nullptr;         // not owned

  // TODO(thuang513): DISALLOW_COPY_AND_ASSIGN(SentenceFragmenter);
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_H_
