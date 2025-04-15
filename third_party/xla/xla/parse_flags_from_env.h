/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PARSE_FLAGS_FROM_ENV_H_
#define XLA_PARSE_FLAGS_FROM_ENV_H_

// This module exports ParseFlagsFromEnvAndDieIfUnknown(), which allows other
// modules to parse flags from an environment variable, or (if the first
// non-whitespace in the variable value is not '-'), a file named by that
// environment variable.
//
// The accepted syntax is that flags arguments are of the form --flag=value or
// (for boolean flags) --flag, and are whitespace separated.  The <value> may be
// one of:
//
//  - <non-whitespace, non-nul not starting with single-quote or double-quote>
//    in which case the effective value is the string itself
//  - <single-quote><characters string not containing nul or
//    single-quote><single_quote> in which case the effective value is the
//    string with the single-quotes removed
//  - <double-quote><character string not containing nul or unescaped
//    double-quote><double_quote> in which case the effective value if the
//    string with the double-quotes removed, and escaped sequences of
//    <backslash><char> replaced by <char>.
//
// Flags values inconsistent with the type of the flag will be rejected by the
// flag parser.
//
// Examples:
//
//  - TF_XLA_FLAGS="--foo=bar  --wombat='value with a space'"
//  - TF_XLA_FLAGS=/tmp/flagfile
//
// where /tmp/flagfile might contain
//
//  --some_flag="This is a string containing a \" and a '."
//  --another_flag=wombats

#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/types.h"

namespace xla {

// Calls tsl::Flags::Parse(argc, argv, flag_list) against any as yet
// unrecognized flags passed in the environment variable `envvar`.
//
// Raises a fatal error if any flags in `envvar` were not recognized, or if flag
// parsing failed.
void ParseFlagsFromEnvAndDieIfUnknown(absl::string_view envvar,
                                      const std::vector<tsl::Flag>& flag_list,
                                      bool reset_envvar = false);

// Calls tsl::Flags::Parse(argc, argv, flag_list) against any as yet
// unrecognized flags passed in the environment variable `envvar`, and returns
// its return value.
//
// Ignores unknown flags, raises a fatal if flag parsing failed.
void ParseFlagsFromEnvAndIgnoreUnknown(absl::string_view envvar,
                                       const std::vector<tsl::Flag>& flag_list,
                                       bool reset_envvar = false);

// Used only for testing.  Not to be used by clients.
void ResetFlagsFromEnvForTesting(absl::string_view envvar, int** pargc,
                                 std::vector<char*>** pargv);

}  // namespace xla

#endif  // XLA_PARSE_FLAGS_FROM_ENV_H_
