# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import absolute_import, division, print_function

import re


_canonicalize_regex = re.compile(r"[-_.]+")


def canonicalize_name(name):
    # This is taken from PEP 503.
    return _canonicalize_regex.sub("-", name).lower()
