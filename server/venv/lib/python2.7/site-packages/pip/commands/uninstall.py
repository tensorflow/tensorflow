from __future__ import absolute_import

import pip
from pip.wheel import WheelCache
from pip.req import InstallRequirement, RequirementSet, parse_requirements
from pip.basecommand import Command
from pip.exceptions import InstallationError


class UninstallCommand(Command):
    """
    Uninstall packages.

    pip is able to uninstall most installed packages. Known exceptions are:

    - Pure distutils packages installed with ``python setup.py install``, which
      leave behind no metadata to determine what files were installed.
    - Script wrappers installed by ``python setup.py develop``.
    """
    name = 'uninstall'
    usage = """
      %prog [options] <package> ...
      %prog [options] -r <requirements file> ..."""
    summary = 'Uninstall packages.'

    def __init__(self, *args, **kw):
        super(UninstallCommand, self).__init__(*args, **kw)
        self.cmd_opts.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='file',
            help='Uninstall all the packages listed in the given requirements '
                 'file.  This option can be used multiple times.',
        )
        self.cmd_opts.add_option(
            '-y', '--yes',
            dest='yes',
            action='store_true',
            help="Don't ask for confirmation of uninstall deletions.")

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        with self._build_session(options) as session:
            format_control = pip.index.FormatControl(set(), set())
            wheel_cache = WheelCache(options.cache_dir, format_control)
            requirement_set = RequirementSet(
                build_dir=None,
                src_dir=None,
                download_dir=None,
                isolated=options.isolated_mode,
                session=session,
                wheel_cache=wheel_cache,
            )
            for name in args:
                requirement_set.add_requirement(
                    InstallRequirement.from_line(
                        name, isolated=options.isolated_mode,
                        wheel_cache=wheel_cache
                    )
                )
            for filename in options.requirements:
                for req in parse_requirements(
                        filename,
                        options=options,
                        session=session,
                        wheel_cache=wheel_cache):
                    requirement_set.add_requirement(req)
            if not requirement_set.has_requirements:
                raise InstallationError(
                    'You must give at least one requirement to %(name)s (see '
                    '"pip help %(name)s")' % dict(name=self.name)
                )
            requirement_set.uninstall(auto_confirm=options.yes)
