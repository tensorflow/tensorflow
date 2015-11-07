"""Generic entry point script."""
import sys

from tensorflow.python.platform import flags


def run():
    f = flags.FLAGS
    f._parse_flags()
    main = sys.modules['__main__'].main
    sys.exit(main(sys.argv))
