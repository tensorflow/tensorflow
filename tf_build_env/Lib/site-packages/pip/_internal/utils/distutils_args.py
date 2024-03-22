from getopt import GetoptError, getopt
from typing import Dict, List

_options = [
    "exec-prefix=",
    "home=",
    "install-base=",
    "install-data=",
    "install-headers=",
    "install-lib=",
    "install-platlib=",
    "install-purelib=",
    "install-scripts=",
    "prefix=",
    "root=",
    "user",
]


def parse_distutils_args(args: List[str]) -> Dict[str, str]:
    """Parse provided arguments, returning an object that has the matched arguments.

    Any unknown arguments are ignored.
    """
    result = {}
    for arg in args:
        try:
            parsed_opt, _ = getopt(args=[arg], shortopts="", longopts=_options)
        except GetoptError:
            # We don't care about any other options, which here may be
            # considered unrecognized since our option list is not
            # exhaustive.
            continue

        if not parsed_opt:
            continue

        option = parsed_opt[0]
        name_from_parsed = option[0][2:].replace("-", "_")
        value_from_parsed = option[1] or "true"
        result[name_from_parsed] = value_from_parsed

    return result
