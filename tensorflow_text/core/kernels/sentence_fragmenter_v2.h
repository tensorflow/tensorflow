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

// Updated version of sentence fragmenter and util functions to split up a
// document into sentence fragments. A sentence fragment is a string whose end
// is potentially an end-of-sentence. The original version of
// sentence_fragmenter operates on tokens and defines the start and end of
// fragments using token indices, while sentence_fragmenter_v2 operates on a
// string_view sliding window of the text and defines the start and end of a
// fragment based on the character offset.
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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

// A class of utils for identifying certain classes and properties of unicode
// characters. These utils are included in the header for use in tests.

// Returns true iff a string is terminal punctuation.
bool IsTerminalPunc(const absl::string_view& input, int* offset);

// Returns true iff a string is close punctuation (close quote or close
// paren).
bool IsClosePunc(const absl::string_view& input, int* offset);

// Returns true iff a string is an open paren.
bool IsOpenParen(const absl::string_view& input);

// Returns true iff a string is a close paren.
bool IsCloseParen(const absl::string_view& input);

// Returns true iff a word is made of punctuation characters only.
bool IsPunctuationWord(const absl::string_view& input);

// Returns true iff a string is an ellipsis ("...").
bool IsEllipsis(const absl::string_view& input, int* offset);

// Returns true iff a string is a period separated acronym (ex: "A.B.C.").
bool IsPeriodSeparatedAcronym(const absl::string_view& input, int* offset);

// Returns true iff a string is an emoticon (ex: ":-)").
bool IsEmoticon(const absl::string_view& input, int* offset);

bool SpaceAllowedBeforeChar(const absl::string_view& input);

void ConsumeOneUChar(const absl::string_view& input, UChar32* result,
                     int* offset);

// Returns true iff a string is white space.
bool IsWhiteSpace(const absl::string_view& input);

class FragmentBoundaryMatch {
 public:
  FragmentBoundaryMatch() {}

  // Goes to initial state.
  void Reset() {
    state_ = INITIAL_STATE;
    first_terminal_punc_index_ = -1;
    first_close_punc_index_ = -1;
    limit_index_ = -1;
  }

  // Follows the state transition for the slice at
  // the given index. Returns true for success, or
  // false if there was no valid transition.
  bool Advance(int index, absl::string_view slice);

  // Returns true iff we have matched at least one terminal punctuation
  // character.
  bool GotTerminalPunc() const { return first_terminal_punc_index_ >= 0; }

  // Field accessors.
  int first_terminal_punc_index() const { return first_terminal_punc_index_; }
  int first_close_punc_index() const { return first_close_punc_index_; }
  int limit_index() const { return limit_index_; }

  // Match state.
  enum MatchState {
    INITIAL_STATE = 0,
    COLLECTING_TERMINAL_PUNC,
    COLLECTING_CLOSE_PUNC
  };

  MatchState state() const { return state_; }

 private:
  MatchState state_ = INITIAL_STATE;

  // First terminal punctuation mark matched; may be an acronym.
  // -1 for not found.
  int first_terminal_punc_index_ = -1;

  // First closing punctuation mark matched. -1 for not found.
  int first_close_punc_index_ = -1;

  // First character after the terminal sequence.
  int limit_index_ = -1;
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
class SentenceFragmenterV2 {
 public:
  // Constructs a fragmenter to process a specific part of a document.
  SentenceFragmenterV2(absl::string_view document) : document_(document) {}

  // Finds sentence fragments in the [start_, limit_) range of the associated
  // document.
  absl::Status FindFragments(std::vector<SentenceFragment>* result);

 private:
  // State for matching a fragment-boundary regexp against a character sequence.
  // The regexp is: terminal_punc+ close_punc*.

  // Matches a fragment-boundary regexp against a slice of the document starting
  // at 'doc_index'. Returns the longest match found; will be non-empty as long
  // as 'doc_index' was not already at the end of the associated document.
  FragmentBoundaryMatch FindNextFragmentBoundary(int doc_index) const;

  // Updates 'latest_open_paren_is_sentential_' for the given
  // fragment.
  void UpdateLatestOpenParenForFragment(int i_start, int i_end);

  // Populates a sentence fragment with the text from 'i_start' to the end
  // of the given FragmentBoundaryMatch.
  void FillInFragmentFields(int i_start, const FragmentBoundaryMatch& match,
                            SentenceFragment* fragment) const;

  // Returns the adjusted first terminal punctuation index in a
  // FragmentBoundaryMatch.
  int GetAdjustedFirstTerminalPuncIndex(
      const FragmentBoundaryMatch& match) const;

  // Returns true iff a FragmentBoundaryMatch has an "unattachable" terminal
  // punctuation mark.
  bool HasUnattachableTerminalPunc(const FragmentBoundaryMatch& match) const;

  // Returns true iff a FragmentBoundaryMatch has a close paren in its closing
  // punctuation.
  bool HasCloseParen(const FragmentBoundaryMatch& match) const;

  // Whether the latest open paren seen so far appears to be sentence-initial.
  // See UpdateLatestOpenParenForFragment() in the .cc file for details.
  bool latest_open_paren_is_sentential_ = false;

  absl::string_view document_ = {};  // not owned

  // TODO(thuang513): DISALLOW_COPY_AND_ASSIGN(SentenceFragmenter);
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_H_
