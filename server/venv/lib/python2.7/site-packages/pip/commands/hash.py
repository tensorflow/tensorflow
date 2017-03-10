from __future__ import absolute_import

import hashlib
import logging
import sys

from pip.basecommand import Command
from pip.status_codes import ERROR
from pip.utils import read_chunks
from pip.utils.hashes import FAVORITE_HASH, STRONG_HASHES


logger = logging.getLogger(__name__)


class HashCommand(Command):
    """
    Compute a hash of a local package archive.

    These can be used with --hash in a requirements file to do repeatable
    installs.

    """
    name = 'hash'
    usage = '%prog [options] <file> ...'
    summary = 'Compute hashes of package archives.'

    def __init__(self, *args, **kw):
        super(HashCommand, self).__init__(*args, **kw)
        self.cmd_opts.add_option(
            '-a', '--algorithm',
            dest='algorithm',
            choices=STRONG_HASHES,
            action='store',
            default=FAVORITE_HASH,
            help='The hash algorithm to use: one of %s' %
                 ', '.join(STRONG_HASHES))
        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        if not args:
            self.parser.print_usage(sys.stderr)
            return ERROR

        algorithm = options.algorithm
        for path in args:
            logger.info('%s:\n--hash=%s:%s',
                        path, algorithm, _hash_of_file(path, algorithm))


def _hash_of_file(path, algorithm):
    """Return the hash digest of a file."""
    with open(path, 'rb') as archive:
        hash = hashlib.new(algorithm)
        for chunk in read_chunks(archive):
            hash.update(chunk)
    return hash.hexdigest()
