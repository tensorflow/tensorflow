# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0

"""CacheControl import Interface.

Make it easy to import from cachecontrol without long namespaces.
"""
__author__ = "Eric Larson"
__email__ = "eric@ionrock.org"
__version__ = "0.12.11"

from .wrapper import CacheControl
from .adapter import CacheControlAdapter
from .controller import CacheController

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
