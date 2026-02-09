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

#include "tensorflow_text/core/kernels/sentence_fragmenter_v2.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

void ConsumeOneUChar(const absl::string_view& input, UChar32* result,
                     int* offset) {
  const char* source = input.data();

  int input_length = input.length();
  U8_NEXT_OR_FFFD(source, *offset, input_length, *result);
}

bool IsTerminalPunc(const absl::string_view& input, int* offset) {
  *offset = 0;
  bool is_ellipsis = IsEllipsis(input, offset);
  if (is_ellipsis) return true;

  *offset = 0;
  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, offset);

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case 0x055C:  // Armenian exclamation mark
    case 0x055E:  // Armenian question mark
    case 0x17d4:  // Khmer sign khan
    case 0x037E:  // Greek question mark
    case 0x2026:  // ellipsis
      return true;
  }

  USentenceBreak sb_property = static_cast<USentenceBreak>(
      u_getIntPropertyValue(char_value, UCHAR_SENTENCE_BREAK));
  return sb_property == U_SB_ATERM || sb_property == U_SB_STERM;
}

bool IsClosePunc(const absl::string_view& input, int* offset) {
  *offset = 0;

  if (absl::StartsWith(input, "''")) {
    *offset += absl::string_view("''").length();
    return true;
  }

  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, offset);

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '>':
    case ']':
    case '`':
    case 64831:  // Ornate right parenthesis
    case 65282:  // fullwidth quotation mark
    case 65287:  // fullwidth apostrophe
      return true;
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));

  return lb_property == U_LB_CLOSE_PUNCTUATION ||
         lb_property == U_LB_CLOSE_PARENTHESIS || lb_property == U_LB_QUOTATION;
}

bool IsOpenParen(const absl::string_view& input) {
  int offset = 0;
  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, &offset);

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '<':
    case 64830:  // Ornate left parenthesis
      return true;
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));
  return lb_property == U_LB_OPEN_PUNCTUATION;
}

bool IsCloseParen(const absl::string_view& input) {
  int offset = 0;

  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, &offset);

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '>':
    case 64831:  // Ornate right parenthesis
      return true;
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));
  return lb_property == U_LB_CLOSE_PUNCTUATION ||
         lb_property == U_LB_CLOSE_PARENTHESIS;
}

bool IsPunctuationWord(const absl::string_view& input) {
  int offset = 0;
  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, &offset);

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '`':
    case '<':
    case '>':
    case '~':
    case 5741:
      return true;
  }

  return u_ispunct(char_value) || u_hasBinaryProperty(char_value, UCHAR_DASH) ||
         u_hasBinaryProperty(char_value, UCHAR_HYPHEN);
}

bool IsEllipsis(const absl::string_view& input, int* offset) {
  *offset = 0;
  if (absl::StartsWith(input, "...")) {
    *offset += absl::string_view("...").length();
    return true;
  }

  const UChar32 kEllipsisCharValue = 0x2026;
  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, offset);

  return char_value == kEllipsisCharValue;
}

inline bool IsAcronymComponent(const absl::string_view& input, int index) {
  return (input.data()[index] >= 'A' && input.data()[index] <= 'Z') &&
         input.data()[index + 1] == '.';
}

bool IsPeriodSeparatedAcronym(const absl::string_view& input, int* offset) {
  bool result = false;

  for (int i = 0; i < static_cast<int>(input.length()) - 1; i += 2) {
    if (IsAcronymComponent(input, i)) {
      *offset = i + 2;
      if (*offset > 2) {
        result = true;
      }
    } else {
      break;
    }
  }
  return result;
}

bool IsEmoticon(const absl::string_view& input, int* offset) {
  *offset = 0;
  static std::vector<std::string> emoticon_list = {":(:)",
                                                   ":)",
                                                   ":(",
                                                   ":o)",
                                                   ":]",
                                                   ":3",
                                                   ":>",
                                                   "=]",
                                                   "=)",
                                                   ":}",
                                                   ":^)",
                                                   ":-D",
                                                   ":-)))))",
                                                   ":-))))",
                                                   ":-)))",
                                                   ":-))",
                                                   ":-)",
                                                   ">:[",
                                                   ":-(",
                                                   ":(",
                                                   ":-c",
                                                   ":c",
                                                   ":-<",
                                                   ":<",
                                                   ":-[",
                                                   ":[",
                                                   ":{",
                                                   ";(",
                                                   ":-||",
                                                   ":@",
                                                   ">:(",
                                                   ":'-(",
                                                   ":'(",
                                                   ":'-)",
                                                   ":')",
                                                   "D:<",
                                                   ">:O",
                                                   ":-O",
                                                   ":-o",
                                                   ":*",
                                                   ":-*",
                                                   ":^*",
                                                   ";-)",
                                                   ";)",
                                                   "*-)",
                                                   "*)",
                                                   ";-]",
                                                   ";]",
                                                   ";^)",
                                                   ":-,",
                                                   ">:P",
                                                   ":-P",
                                                   ":p",
                                                   "=p",
                                                   ":-p",
                                                   "=p",
                                                   ":P",
                                                   "=P",
                                                   ";p",
                                                   ";-p",
                                                   ";P",
                                                   ";-P",
                                                   ">:\\",
                                                   ">:/",
                                                   ":-/",
                                                   ":-.",
                                                   ":/",
                                                   ":\\",
                                                   "=/",
                                                   "=\\",
                                                   ":|",
                                                   ":-|",
                                                   ":$",
                                                   ":-#",
                                                   ":#",
                                                   "O:-)",
                                                   "0:-)",
                                                   "0:)",
                                                   "0;^)",
                                                   ">:)",
                                                   ">;)",
                                                   ">:-)",
                                                   "}:-)",
                                                   "}:)",
                                                   "3:-)",
                                                   ">_>^",
                                                   "^<_<",
                                                   "|;-)",
                                                   "|-O",
                                                   ":-J",
                                                   ":-&",
                                                   ":&",
                                                   "#-)",
                                                   "<3",
                                                   "8-)",
                                                   "^_^",
                                                   ":D",
                                                   ":-D",
                                                   "=D",
                                                   "^_^;;",
                                                   "O=)",
                                                   "}=)",
                                                   "B)",
                                                   "B-)",
                                                   "=|",
                                                   "-_-",
                                                   "o_o;",
                                                   "u_u",
                                                   ":-\\",
                                                   ":s",
                                                   ":S",
                                                   ":-s",
                                                   ":-S",
                                                   ";*",
                                                   ";-*"
                                                   "=(",
                                                   ">.<",
                                                   ">:-(",
                                                   ">:(",
                                                   ">=(",
                                                   ";_;",
                                                   "T_T",
                                                   "='(",
                                                   ">_<",
                                                   "D:",
                                                   ":o",
                                                   ":-o",
                                                   "=o",
                                                   "o.o",
                                                   ":O",
                                                   ":-O",
                                                   "=O",
                                                   "O.O",
                                                   "x_x",
                                                   "X-(",
                                                   "X(",
                                                   "X-o",
                                                   "X-O",
                                                   ":X)",
                                                   "(=^.^=)",
                                                   "(=^..^=)",
                                                   "=^_^=",
                                                   "-<@%",
                                                   ":(|)",
                                                   "(]:{",
                                                   "<\\3",
                                                   "~@~",
                                                   "8'(",
                                                   "XD",
                                                   "DX"};

  for (int i = 0; i < static_cast<int>(emoticon_list.size()); ++i) {
    if (absl::StartsWith(input, emoticon_list[i])) {
      *offset = emoticon_list[i].length();
      return true;
    }
  }
  return false;
}

// Returns true iff the punctuation input can appear after a space in a
// sentence-terminal punctuation sequence.
bool SpaceAllowedBeforeChar(const absl::string_view& input) {
  int offset = 0;
  bool is_terminal_punc = IsTerminalPunc(input, &offset);
  bool is_close_paren = IsCloseParen(input);
  bool is_emoticon = IsEmoticon(input, &offset);
  return is_terminal_punc || is_close_paren || is_emoticon;
}

bool IsWhiteSpace(const absl::string_view& input) {
  int offset = 0;

  if (absl::StartsWith(input, " ")) {
    return true;
  } else if (absl::StartsWith(input, "\n")) {
    return true;
  } else if (absl::StartsWith(input, "  ")) {
    return true;
  }

  UChar32 char_value;
  ConsumeOneUChar(input, &char_value, &offset);

  return u_isUWhiteSpace(char_value);
}

// Follows the state transition for the slice at the given index. Returns true
// for success, or false if there was no valid transition.
bool FragmentBoundaryMatch::Advance(int index, absl::string_view slice) {
  int temp_offset;
  // By defualt offset is the next character.
  int offset = 1;
  bool no_transition = false;
  bool is_terminal_punc = IsTerminalPunc(slice, &temp_offset);
  if (is_terminal_punc) {
    offset = temp_offset;
  }

  bool is_ellipsis = IsEllipsis(slice, &temp_offset);
  if (is_ellipsis) {
    offset = temp_offset;
  }
  bool is_close_punc = IsClosePunc(slice, &temp_offset);
  if (is_close_punc) {
    offset = temp_offset;
  }
  bool is_acronym = IsPeriodSeparatedAcronym(slice, &temp_offset);
  if (is_acronym) {
    is_terminal_punc = false;
    offset = temp_offset;
  }
  bool is_emoticon = IsEmoticon(slice, &temp_offset);
  if (is_emoticon) {
    is_terminal_punc = false;
    offset = temp_offset;
  }

  switch (state_) {
    case INITIAL_STATE:
      if (is_terminal_punc || is_acronym || is_emoticon) {
        first_terminal_punc_index_ = index;
        state_ = COLLECTING_TERMINAL_PUNC;
      }
      break;
    case COLLECTING_TERMINAL_PUNC:
      if (is_terminal_punc || is_emoticon) {
        // Stay in COLLECTING_TERMINAL_PUNC state.
      } else if (is_close_punc) {
        first_close_punc_index_ = index;
        state_ = COLLECTING_CLOSE_PUNC;
      } else {
        no_transition = true;
      }
      break;
    case COLLECTING_CLOSE_PUNC:
      if (is_close_punc || is_ellipsis || is_emoticon) {
        // Stay in COLLECTING_CLOSE_PUNC state. We effectively ignore
        // emoticons and ellipses and continue to accept closing punctuation
        // after them.
      } else {
        no_transition = true;
      }
      break;
  }

  if (no_transition) {
    return false;
  } else {
    limit_index_ = index + offset;
    if (state_ == COLLECTING_TERMINAL_PUNC) {
      // We've gotten terminal punctuation, but no close punctuation yet.
      first_close_punc_index_ = limit_index_;
    }
    return true;
  }
}

// Sets a property of a sentence fragment.
void SetFragmentProperty(SentenceFragment::Property property,
                         SentenceFragment* fragment) {
  fragment->properties = fragment->properties | property;
}

absl::Status SentenceFragmenterV2::FindFragments(
    std::vector<SentenceFragment>* result) {
  // Partition document into sentence fragments.
  for (int i_start = 0; i_start < static_cast<int>(document_.size());) {
    bool is_white_space = IsWhiteSpace(document_.substr(i_start));
    if (is_white_space) {
      ++i_start;
      continue;
    }

    SentenceFragment fragment;

    // Match regexp for fragment boundary.
    FragmentBoundaryMatch match = FindNextFragmentBoundary(i_start);

    // Update 'latest_open_paren_is_sentential_' for this fragment.
    UpdateLatestOpenParenForFragment(i_start, match.limit_index());

    // Add a new sentence fragment up to this boundary.
    FillInFragmentFields(i_start, match, &fragment);

    result->push_back(std::move(fragment));
    i_start = match.limit_index();
  }
  return absl::OkStatus();
}

// This method is essentially a control layer on top of a simple state machine
// that matches an end-of-fragment regexp. This method finds the next slice of
// text to feed to the state machine, and handles embedded whitespace. The main
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

FragmentBoundaryMatch SentenceFragmenterV2::FindNextFragmentBoundary(
    int doc_index) const {
  FragmentBoundaryMatch current_match;
  FragmentBoundaryMatch previous_match;

  for (int i = doc_index; i < static_cast<int>(document_.size()); ++i) {
    absl::string_view slice = document_.substr(i);
    if (current_match.GotTerminalPunc() && i > doc_index) {
      // Got terminal punctuation and a space delimiter, so match is valid.
      bool space_allowed_before_char = SpaceAllowedBeforeChar(slice);
      if (space_allowed_before_char) {
        // Remember this match. Try to extend it.
        previous_match = current_match;
      } else {
        // Stop here. We're not allowed to extend the match in this case.
        break;
      }
    }
    bool got_transition = current_match.Advance(i, slice);
    if (!got_transition) {
      if (previous_match.GotTerminalPunc()) {
        // Extension failed. Return previous match.
        return previous_match;
      } else {
        // Start matching again from scratch.
        current_match.Reset();

        // Reprocess current character since it might be terminal punctuation.
        // No infinite loop, because can't be "no transition" from
        // INITIAL_STATE.
        --i;
      }
    } else {
      i = current_match.limit_index() - 1;
    }
  }
  return current_match;
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
void SentenceFragmenterV2::UpdateLatestOpenParenForFragment(int i_start,
                                                            int i_end) {
  for (int i = i_end; i > i_start; --i) {
    absl::string_view slice = document_.substr(i);
    if (slice.length() > 0 && IsOpenParen(slice)) {
      // Make the approximation that this open paren is sentence-initial iff it
      // is fragment-initial.
      latest_open_paren_is_sentential_ = (i == i_start);
      break;
    }
  }
}

void SentenceFragmenterV2::FillInFragmentFields(
    int i_start, const FragmentBoundaryMatch& match,
    SentenceFragment* fragment) const {
  // Set the fragment's boundaries.
  fragment->start = i_start;
  fragment->limit = match.limit_index();

  // Set the fragment's properties.
  if (match.GotTerminalPunc()) {
    // TERMINAL_PUNC.
    SetFragmentProperty(SentenceFragment::TERMINAL_PUNC, fragment);
    int terminal_punc_index = GetAdjustedFirstTerminalPuncIndex(match);

    bool has_unattachable_terminal_punc = HasUnattachableTerminalPunc(match);
    bool has_close_paren = HasCloseParen(match);

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
int SentenceFragmenterV2::GetAdjustedFirstTerminalPuncIndex(
    const FragmentBoundaryMatch& match) const {
  // Get terminal punctuation span.
  int i1 = match.first_terminal_punc_index();
  if (i1 < 0) {
    return i1;
  }
  int i2 = match.first_close_punc_index();

  for (int i = i2; i > i1; --i) {
    absl::string_view slice = document_.substr(i);
    int temp_offset = 0;
    bool is_ellipsis = IsEllipsis(slice, &temp_offset);
    bool is_emoticon = IsEmoticon(slice, &temp_offset);
    if (is_ellipsis || is_emoticon) {
      if (i == i2) {
        // Ellipsis is last terminal punctuation mark. No adjustment.
        return i1;
      } else {
        // Ellipsis is not the last terminal punctuation mark. Return the index
        // of the terminal punctuation mark after it.
        return i;  // current character = i - 1
      }
    }
  }
  // No ellipsis.
  return i1;
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
bool SentenceFragmenterV2::HasUnattachableTerminalPunc(
    const FragmentBoundaryMatch& match) const {
  // Get terminal punctuation span.
  int i1 = match.first_terminal_punc_index();
  if (i1 < 0) {
    return false;
  }
  // Check where second and later punctuation marks start
  absl::string_view start_slice = document_.substr(i1);
  int temp_offset = 0;
  bool is_ellipsis = IsEllipsis(start_slice, &temp_offset);
  if (is_ellipsis) {
    i1 += temp_offset - 1;
  }
  bool is_emoticon = IsEmoticon(start_slice, &temp_offset);
  if (is_emoticon) {
    i1 += temp_offset - 1;
  }

  int i2 = match.first_close_punc_index();

  // Iterate over the second and later punctuation marks.
  for (int i = i1 + 1; i < i2; ++i) {
    absl::string_view slice = document_.substr(i);
    bool is_punctuation = IsPunctuationWord(slice);
    is_ellipsis = IsEllipsis(slice, &temp_offset);
    if (is_ellipsis) {
      i += temp_offset - 1;
    }
    is_emoticon = IsEmoticon(slice, &temp_offset);
    if (is_emoticon) {
      i += temp_offset - 1;
    }
    if (is_punctuation && !is_ellipsis && !is_emoticon) {
      // Found an unattachable, unambiguous terminal punctuation mark.
      return true;
    }
  }
  return false;
}

bool SentenceFragmenterV2::HasCloseParen(
    const FragmentBoundaryMatch& match) const {
  // Get close punctuation span.
  int i1 = match.first_close_punc_index();
  if (i1 < 0) {
    return false;
  }
  int i2 = match.limit_index();

  for (int i = i1; i < i2; ++i) {
    absl::string_view slice = document_.substr(i);
    if (IsCloseParen(slice)) {
      return true;
    }
  }
  return false;
}

}  // namespace text
}  // namespace tensorflow
