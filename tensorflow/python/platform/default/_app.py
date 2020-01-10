"""Generic entry point script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.platform import flags


def run():
  f = flags.FLAGS
  f._parse_flags()
  main = sys.modules['__main__'].main
  sys.exit(main(sys.argv))
