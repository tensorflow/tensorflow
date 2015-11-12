"""Start a simple interactive console with TensorFlow available."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import code
import sys


def main(_):
  """Run an interactive console."""
  code.interact()
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
