/*
 * Copyright 2015 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include "flatbuffers/hash.h"

enum OutputFormat { kDecimal, kHexadecimal, kHexadecimal0x };

int main(int argc, char *argv[]) {
  const char *name = argv[0];
  if (argc <= 1) {
    printf("%s HASH [OPTION]... [--] STRING...\n", name);
    printf("Available hashing algorithms:\n");
    printf("  16 bit:\n");
    size_t size = sizeof(flatbuffers::kHashFunctions16) /
                  sizeof(flatbuffers::kHashFunctions16[0]);
    for (size_t i = 0; i < size; ++i) {
      printf("    * %s\n", flatbuffers::kHashFunctions16[i].name);
    }
    printf("  32 bit:\n");
    size = sizeof(flatbuffers::kHashFunctions32) /
                  sizeof(flatbuffers::kHashFunctions32[0]);
    for (size_t i = 0; i < size; ++i) {
      printf("    * %s\n", flatbuffers::kHashFunctions32[i].name);
    }
    printf("  64 bit:\n");
    size = sizeof(flatbuffers::kHashFunctions64) /
           sizeof(flatbuffers::kHashFunctions64[0]);
    for (size_t i = 0; i < size; ++i) {
      printf("    * %s\n", flatbuffers::kHashFunctions64[i].name);
    }
    printf(
        "  -d         Output hash in decimal.\n"
        "  -x         Output hash in hexadecimal.\n"
        "  -0x        Output hash in hexadecimal and prefix with 0x.\n"
        "  -c         Append the string to the output in a c-style comment.\n");
    return 1;
  }

  const char *hash_algorithm = argv[1];

  flatbuffers::NamedHashFunction<uint16_t>::HashFunction hash_function16 =
      flatbuffers::FindHashFunction16(hash_algorithm);
  flatbuffers::NamedHashFunction<uint32_t>::HashFunction hash_function32 =
      flatbuffers::FindHashFunction32(hash_algorithm);
  flatbuffers::NamedHashFunction<uint64_t>::HashFunction hash_function64 =
      flatbuffers::FindHashFunction64(hash_algorithm);

  if (!hash_function16 && !hash_function32 && !hash_function64) {
    printf("\"%s\" is not a known hash algorithm.\n", hash_algorithm);
    return 1;
  }

  OutputFormat output_format = kHexadecimal;
  bool annotate = false;
  bool escape_dash = false;
  for (int i = 2; i < argc; i++) {
    const char *arg = argv[i];
    if (!escape_dash && arg[0] == '-') {
      std::string opt = arg;
      if (opt == "-d")
        output_format = kDecimal;
      else if (opt == "-x")
        output_format = kHexadecimal;
      else if (opt == "-0x")
        output_format = kHexadecimal0x;
      else if (opt == "-c")
        annotate = true;
      else if (opt == "--")
        escape_dash = true;
      else
        printf("Unrecognized argument: \"%s\"\n", arg);
    } else {
      std::stringstream ss;
      if (output_format == kDecimal) {
        ss << std::dec;
      } else if (output_format == kHexadecimal) {
        ss << std::hex;
      } else if (output_format == kHexadecimal0x) {
        ss << std::hex;
        ss << "0x";
      }
      if (hash_function16)
        ss << hash_function16(arg);
      else if (hash_function32)
        ss << hash_function32(arg);
      else if (hash_function64)
        ss << hash_function64(arg);

      if (annotate) ss << " /* \"" << arg << "\" */";

      ss << "\n";

      std::cout << ss.str();
    }
  }
  return 0;
}
