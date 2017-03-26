from __future__ import absolute_import

import sys

import pip
from pip.compat import stdlib_pkgs
from pip.basecommand import Command
from pip.operations.freeze import freeze
from pip.wheel import WheelCache


DEV_PKGS = ('pip', 'setuptools', 'distribute', 'wheel')


class FreezeCommand(Command):
    """
    Output installed packages in requirements format.

    packages are listed in a case-insensitive sorted order.
    """
    name = 'freeze'
    usage = """
      %prog [options]"""
    summary = 'Output installed packages in requirements format.'
    log_streams = ("ext://sys.stderr", "ext://sys.stderr")

    def __init__(self, *args, **kw):
        super(FreezeCommand, self).__init__(*args, **kw)

        self.cmd_opts.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='file',
            help="Use the order in the given requirements file and its "
                 "comments when generating output. This option can be "
                 "used multiple times.")
        self.cmd_opts.add_option(
            '-f', '--find-links',
            dest='find_links',
            action='append',
            default=[],
            metavar='URL',
            help='URL for finding packages, which will be added to the '
                 'output.')
        self.cmd_opts.add_option(
            '-l', '--local',
            dest='local',
            action='store_true',
            default=False,
            help='If in a virtualenv that has global access, do not output '
                 'globally-installed packages.')
        self.cmd_opts.add_option(
            '--user',
            dest='user',
            action='store_true',
            default=False,
            help='Only output packages installed in user-site.')
        self.cmd_opts.add_option(
            '--all',
            dest='freeze_all',
            action='store_true',
            help='Do not skip these packages in the output:'
                 ' %s' % ', '.join(DEV_PKGS))

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        format_control = pip.index.FormatControl(set(), set())
        wheel_cache = WheelCache(options.cache_dir, format_control)
        skip = set(stdlib_pkgs)
        if not options.freeze_all:
            skip.update(DEV_PKGS)

        freeze_kwargs = dict(
            requirement=options.requirements,
            find_links=options.find_links,
            local_only=options.local,
            user_only=options.user,
            skip_regex=options.skip_requirements_regex,
            isolated=options.isolated_mode,
            wheel_cache=wheel_cache,
            skip=skip)

        for line in freeze(**freeze_kwargs):
            sys.stdout.write(line + '\n')
