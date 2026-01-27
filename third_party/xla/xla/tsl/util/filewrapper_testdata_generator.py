# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This script generates carefully constructed testdata.

The data is designed to probe a number of edge cases with embedding data in a
C string literal.
"""

import itertools
import random

from absl import app
from absl import flags


_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "File to write output to."
)


def main(unused_argv):
  with open(_OUTPUT_FILE.value, "w", encoding="utf-8") as f:
    # Nuls followed by every character
    f.write("\0" + "\0".join([chr(x) for x in range(256)]))

    # Double-question-mark (i.e. potential trigraph) followed by every
    # character.
    f.write("??" + "??".join([chr(x) for x in range(256)]))

    # Triple-question-mark followed by every 2-byte sequence from the
    # "interesting" set chr(0:48) + chr(127:129). This covers a wide range of
    # trigraph-like things, as well as combinations of characters that might
    # cause issues.
    interesting_chars = [
        chr(x) for x in itertools.chain(range(48), range(127, 129))
    ]
    f.write(
        "???"
        + "???".join(
            x + y for x, y in zip(interesting_chars, interesting_chars)
        )
    )

    # Random characters from the "difficult" set "\0\n\r\\\"=/'()!<>-?01234567".
    # These are the only characters that would ever (reasonably) have
    # special-case handling, not counting the high-bit characters. With 200k of
    # these 23 characters, we should cover every 3-character sequence at least
    # once, as well as most 4-character sequences.
    rand = random.Random(0)
    f.write(
        "".join(
            rand.choice("\0\n\r\\\"=/'()!<>-?01234567") for _ in range(200000)
        )
    )

    # Finally, 1.5M of totally random data. This should cover all two-byte
    # sequences at least once, and many 3-byte sequences.
    f.write("".join(chr(rand.getrandbits(8)) for _ in range(1500000)))


if __name__ == "__main__":
  flags.mark_flag_as_required("output_file")
  app.run(main)
