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

#include "tensorflow_text/core/kernels/sentence_fragmenter.h"
#include <string>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_text/core/kernels/sentence_breaking_utils.h"

using ::tensorflow::Status;

namespace tensorflow {
namespace text {
namespace {

// Sets a property of a sentence fragment.
void SetFragmentProperty(SentenceFragment::Property property,
                         SentenceFragment *fragment) {
  fragment->properties = fragment->properties | property;
}

// Returns true iff a token has any of the given properties.
bool TokenHasProperty(uint32 properties, const Token &token) {
  return token.text_properties() & properties;
}

// Returns true iff a token has the ACRONYM text property and token.word()
// ends with a period.
bool IsPeriodSeparatedAcronym(const Token &token) {
  return TokenHasProperty(Token::ACRONYM, token) &&
         (!token.word().empty() && token.word().back() == '.');
}

// Returns true iff the token can appear after a space in a sentence-terminal
// token sequence.
absl::Status SpaceAllowedBeforeToken(const UnicodeUtil *util,
                                     const Token &token, bool *result) {
  const tstring &word = token.word();
  bool is_ellipsis = false;
  TF_RETURN_IF_ERROR(util->IsEllipsis(word, &is_ellipsis));

  bool is_terminal_punc = false;
  TF_RETURN_IF_ERROR(util->IsTerminalPunc(word, &is_terminal_punc));

  bool is_close_paren = false;
  TF_RETURN_IF_ERROR(util->IsCloseParen(word, &is_close_paren));

  *result = (TokenHasProperty(Token::EMOTICON, token) ||
             (is_ellipsis || is_terminal_punc || is_close_paren));
  return absl::OkStatus();
}
}  // namespace

class SentenceFragmenter::FragmentBoundaryMatch {
 public:
  FragmentBoundaryMatch() {
    Reset();
  }

  // Goes to initial state.
  void Reset() {
    state_ = INITIAL_STATE;
    first_terminal_punc_index_ = -1;
    first_close_punc_index_ = -1;
    limit_index_ = -1;
  }

  // Follows the state transition for the token at the given index. Returns
  // true for success, or false if there was no valid transition.
  absl::Status Advance(const UnicodeUtil *util, const Document &document,
                       int index, bool *result) {
    const Token &token = document.tokens()[index];
    const tstring &word = token.word();
    bool no_transition = false;

    bool is_terminal_punc = false;
    TF_RETURN_IF_ERROR(util->IsTerminalPunc(word, &is_terminal_punc));

    bool is_ellipsis = false;
    TF_RETURN_IF_ERROR(util->IsEllipsis(word, &is_ellipsis));

    bool is_close_punc = false;
    TF_RETURN_IF_ERROR(util->IsClosePunc(word, &is_close_punc));

    switch (state_) {
      case INITIAL_STATE:
        if (is_terminal_punc || is_ellipsis ||
            IsPeriodSeparatedAcronym(token) ||
            TokenHasProperty(Token::EMOTICON, token)) {
          first_terminal_punc_index_ = index;
          state_ = COLLECTING_TERMINAL_PUNC;
        }
        break;
      case COLLECTING_TERMINAL_PUNC:

        if (is_terminal_punc || is_ellipsis ||
            TokenHasProperty(Token::EMOTICON, token)) {
          // Stay in COLLECTING_TERMINAL_PUNC state.
        } else if (is_close_punc) {
          first_close_punc_index_ = index;
          state_ = COLLECTING_CLOSE_PUNC;
        } else {
          no_transition = true;
        }
        break;
      case COLLECTING_CLOSE_PUNC:
        if (is_close_punc || is_ellipsis ||
            TokenHasProperty(Token::EMOTICON, token)) {
          // Stay in COLLECTING_CLOSE_PUNC state. We effectively ignore
          // emoticons and ellipses and continue to accept closing punctuation
          // after them.
        } else {
          no_transition = true;
        }
        break;
    }

    if (no_transition) {
      *result = false;
      return absl::OkStatus();
    } else {
      limit_index_ = index + 1;
      if (state_ == COLLECTING_TERMINAL_PUNC) {
        // We've gotten terminal punctuation, but no close punctuation yet.
        first_close_punc_index_ = limit_index_;
      }
      *result = true;
      return absl::OkStatus();
    }
  }

  // Returns true iff we have matched at least one terminal punctuation
  // character.
  bool GotTerminalPunc() const {
    return first_terminal_punc_index_ >= 0;
  }

  // Field accessors.
  int first_terminal_punc_index() const {
    return first_terminal_punc_index_;
  }
  int first_close_punc_index() const {
    return first_close_punc_index_;
  }
  int limit_index() const {
    return limit_index_;
  }

 private:
  // Match state.
  enum MatchState {
    INITIAL_STATE = 0,
    COLLECTING_TERMINAL_PUNC,
    COLLECTING_CLOSE_PUNC
  };
  MatchState state_ = INITIAL_STATE;

  // First terminal punctuation mark matched; may be an acronym.
  // -1 for not found.
  int first_terminal_punc_index_ = -1;

  // First closing punctuation mark matched. -1 for not found.
  int first_close_punc_index_ = -1;

  // First token after the terminal sequence.
  int limit_index_ = -1;
};

absl::Status SentenceFragmenter::FindFragments(
    std::vector<SentenceFragment> *result) {
  // Partition tokens into sentence fragments.
  for (int i_start = 0; i_start < document_->tokens().size();) {
    SentenceFragment fragment;

    // Match regexp for fragment boundary.
    FragmentBoundaryMatch match;
    TF_RETURN_IF_ERROR(FindNextFragmentBoundary(i_start, &match));

    // Update 'latest_open_paren_is_sentential_' for the tokens in this
    // fragment.
    TF_RETURN_IF_ERROR(
        UpdateLatestOpenParenForFragment(i_start, match.limit_index()));

    // Add a new sentence fragment up to this boundary.
    TF_RETURN_IF_ERROR(FillInFragmentFields(i_start, match, &fragment));

    result->push_back(std::move(fragment));
    i_start = match.limit_index();
  }
  return absl::OkStatus();
}

// This method is essentially a control layer on top of a simple state machine
// that matches an end-of-fragment regexp. This method finds the next token to
// feed to the state machine, and handles embedded whitespace. The main
// complexity is that a space may delimit end-of-match, or be embedded in the
// termination sequence. When we encounter a space, we record the match found so
// far, but also continue matching. We return the longer match if it succeeds,
// else fall back to the earlier one. Note that the lookahead can incur at most
// 2n cost.
//
// E.g., suppose we're given: x? !!!y. We encounter the space after "x?" and
// have to look ahead all the way to "y" before realizing that the longer match
// fails. We put a fragment boundary after "x?", and next time around, we again
// scan "!!!" looking for a fragment boundary. Since we failed to find one last
// time, we'll fail again this time and therefore continue past "y" to find the
// next boundary. We will not try to scan "!!!" a third time.
absl::Status SentenceFragmenter::FindNextFragmentBoundary(
    int i_start, SentenceFragmenter::FragmentBoundaryMatch *result) const {
  FragmentBoundaryMatch current_match;
  FragmentBoundaryMatch previous_match;

  for (int i = i_start; i < static_cast<int>(document_->tokens().size()); ++i) {
    const auto &token = document_->tokens()[i];
    if (current_match.GotTerminalPunc() && i > i_start &&
        token.break_level() >= Token::SPACE_BREAK) {
      // Got terminal punctuation and a space delimiter, so match is valid.
      bool space_allowed_before_token = false;
      TF_RETURN_IF_ERROR(
          SpaceAllowedBeforeToken(util_, token, &space_allowed_before_token));
      if (space_allowed_before_token) {
        // Remember this match. Try to extend it.
        previous_match = current_match;
      } else {
        // Stop here. We're not allowed to extend the match in this case.
        break;
      }
    }
    bool got_transition = false;
    TF_RETURN_IF_ERROR(
        current_match.Advance(util_, *document_, i, &got_transition));
    if (!got_transition) {
      if (previous_match.GotTerminalPunc()) {
        // Extension failed. Return previous match.
        *result = previous_match;
        return absl::OkStatus();
      } else {
        // Start matching again from scratch.
        current_match.Reset();

        // Reprocess current token since it might be terminal punctuation. No
        // infinite loop, because can't be "no transition" from INITIAL_STATE.
        --i;
      }
    }
  }
  *result = current_match;
  return absl::OkStatus();
}

// Keep track of whether the latest open parenthesis seen so far appears to be
// sentence-initial. This is useful because if it is *non-sentence-initial*,
// then any terminal punctuation before the corresponding close paren is
// probably not a sentence boundary. Example:
//
//   Mushrooms (they're fungi!!) are delicious.
//   (Mushrooms are fungi!!)
//
// In the first case, the open paren is non-sentence-initial, and therefore
// the "!!)" is not a sentence boundary. In the second case, the open paren *is*
// sentence-initial, and so the "!!)" is a sentence boundary.
//
// Of course, we don't know true sentence boundaries, so we make the
// approximation that an open paren is sentence-initial iff it is
// fragment-initial. This will be wrong if the open paren occurs after terminal
// punctuation that turns out not to be a sentence boundary, e.g.,
// "Yahoo! (known for search, etc.) blah", but this is not expected to happen
// often.
absl::Status SentenceFragmenter::UpdateLatestOpenParenForFragment(int i_start,
                                                                  int i_end) {
  for (int i = i_end; i > i_start; --i) {
    const auto &token = document_->tokens()[i - 1];
    bool is_open_paren = false;
    TF_RETURN_IF_ERROR(util_->IsOpenParen(token.word(), &is_open_paren));
    if (is_open_paren) {
      // Make the approximation that this open paren is sentence-initial iff it
      // is fragment-initial.
      latest_open_paren_is_sentential_ = (i - 1 == i_start);
      break;
    }
  }

  return absl::OkStatus();
}

absl::Status SentenceFragmenter::FillInFragmentFields(
    int i_start, const FragmentBoundaryMatch &match,
    SentenceFragment *fragment) const {
  // Set the fragment's boundaries.
  fragment->start = i_start;
  fragment->limit = match.limit_index();

  // Set the fragment's properties.
  if (match.GotTerminalPunc()) {
    // TERMINAL_PUNC.
    SetFragmentProperty(SentenceFragment::TERMINAL_PUNC, fragment);
    int terminal_punc_index = -1;
    TF_RETURN_IF_ERROR(
        GetAdjustedFirstTerminalPuncIndex(match, &terminal_punc_index));
    bool has_unattachable_terminal_punc = false;
    TF_RETURN_IF_ERROR(
        HasUnattachableTerminalPunc(match, &has_unattachable_terminal_punc));
    bool has_close_paren = false;
    TF_RETURN_IF_ERROR(HasCloseParen(match, &has_close_paren));

    fragment->terminal_punc_token = terminal_punc_index;
    // MULTIPLE_TERMINAL_PUNC.
    if (has_unattachable_terminal_punc) {
      SetFragmentProperty(SentenceFragment::MULTIPLE_TERMINAL_PUNC, fragment);
    }

    // HAS_CLOSE_PAREN & HAS_SENTENTIAL_CLOSE_PAREN.
    if (has_close_paren) {
      SetFragmentProperty(SentenceFragment::HAS_CLOSE_PAREN, fragment);

      if (latest_open_paren_is_sentential_) {
        SetFragmentProperty(SentenceFragment::HAS_SENTENTIAL_CLOSE_PAREN,
                            fragment);
      }
    }
  }

  return absl::OkStatus();
}

// The standard first terminal punctuation index is just
// match.first_terminal_punc_index(). But if there is an ambiguous terminal
// punctuation mark (ellipsis) followed by an unambiguous one (.!?), then we
// treat the ellipsis as part of the sentence, and return the index of the first
// unambiguous punctuation mark after it. Example:
//
//   He agreed...!
//
// We treat "!" as the first terminal punctuation mark; the ellipsis acts as
// left context.
absl::Status SentenceFragmenter::GetAdjustedFirstTerminalPuncIndex(
    const FragmentBoundaryMatch &match, int *result) const {
  // Get terminal punctuation span.
  int i1 = match.first_terminal_punc_index();
  if (i1 < 0) {
    *result = i1;
    return absl::OkStatus();
  }
  int i2 = match.first_close_punc_index();

  for (int i = i2; i > i1; --i) {
    const auto &token = document_->tokens()[i - 1];
    bool is_ellipsis = false;
    TF_RETURN_IF_ERROR(util_->IsEllipsis(token.word(), &is_ellipsis));
    if (is_ellipsis || TokenHasProperty(Token::EMOTICON, token)) {
      if (i == i2) {
        // Ellipsis is last terminal punctuation mark. No adjustment.
        *result = i1;
        return absl::OkStatus();
      } else {
        // Ellipsis is not the last terminal punctuation mark. Return the index
        // of the terminal punctuation mark after it.
        *result = i;  // current token = i - 1
        return absl::OkStatus();
      }
    }
  }

  // No ellipsis.
  *result = i1;
  return absl::OkStatus();
}

// Example of an an "unattachable" terminal punctuation mark:
//
//   He agreed!?
//
// The "?" is "unattachable" in that it can't be part of the word "agreed"
// because of the intervening "!", and therefore strongly suggests this is a
// true sentence boundary. The terminal punctuation mark must be unambiguous
// (.!?), as ambiguous ones (ellipsis/emoticon) do not necessarily imply a
// sentence boundary.
absl::Status SentenceFragmenter::HasUnattachableTerminalPunc(
    const FragmentBoundaryMatch &match, bool *result) const {
  *result = false;
  // Get terminal punctuation span.
  int i1 = match.first_terminal_punc_index();
  if (i1 < 0) {
    *result = false;
    return absl::OkStatus();
  }
  int i2 = match.first_close_punc_index();

  // Iterate over the second and later punctuation marks.
  for (int i = i1 + 1; i < i2; ++i) {
    const auto &token = document_->tokens()[i];
    bool is_punctuation = false;
    TF_RETURN_IF_ERROR(util_->IsPunctuationWord(token.word(), &is_punctuation));
    bool is_ellipsis = false;
    TF_RETURN_IF_ERROR(util_->IsEllipsis(token.word(), &is_ellipsis));
    if (is_punctuation && !is_ellipsis &&
        !TokenHasProperty(Token::EMOTICON, token)) {
      // Found an unattachable, unambiguous terminal punctuation mark.
      *result = true;
      return absl::OkStatus();
    }
  }

  *result = false;
  return absl::OkStatus();
}

absl::Status SentenceFragmenter::HasCloseParen(
    const FragmentBoundaryMatch &match, bool *result) const {
  *result = false;
  // Get close punctuation span.
  int i1 = match.first_close_punc_index();
  if (i1 < 0) {
    *result = false;
    return absl::OkStatus();
  }
  int i2 = match.limit_index();

  for (int i = i1; i < i2; ++i) {
    const auto &token = document_->tokens()[i];
    bool is_close_paren = false;
    TF_RETURN_IF_ERROR(util_->IsCloseParen(token.word(), &is_close_paren));
    if (is_close_paren) {
      *result = true;
      return absl::OkStatus();
    }
  }
  *result = false;
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
