from __future__ import absolute_import

import logging
import os

from pip.exceptions import CommandError
from pip.index import FormatControl
from pip.req import RequirementSet
from pip.basecommand import RequirementCommand
from pip import cmdoptions
from pip.utils import ensure_dir, normalize_path
from pip.utils.build import BuildDirectory
from pip.utils.filesystem import check_path_owner


logger = logging.getLogger(__name__)


class DownloadCommand(RequirementCommand):
    """
    Download packages from:

    - PyPI (and other indexes) using requirement specifiers.
    - VCS project urls.
    - Local project directories.
    - Local or remote source archives.

    pip also supports downloading from "requirements files", which provide
    an easy way to specify a whole environment to be downloaded.
    """
    name = 'download'

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] [-e] <vcs project url> ...
      %prog [options] [-e] <local project path> ...
      %prog [options] <archive url/path> ..."""

    summary = 'Download packages.'

    def __init__(self, *args, **kw):
        super(DownloadCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.editable())
        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.build_dir())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.global_options())
        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(cmdoptions.src())
        cmd_opts.add_option(cmdoptions.pre())
        cmd_opts.add_option(cmdoptions.no_clean())
        cmd_opts.add_option(cmdoptions.require_hashes())

        cmd_opts.add_option(
            '-d', '--dest', '--destination-dir', '--destination-directory',
            dest='download_dir',
            metavar='dir',
            default=os.curdir,
            help=("Download packages into <dir>."),
        )

        cmd_opts.add_option(
            '--platform',
            dest='platform',
            metavar='platform',
            default=None,
            help=("Only download wheels compatible with <platform>. "
                  "Defaults to the platform of the running system."),
        )

        cmd_opts.add_option(
            '--python-version',
            dest='python_version',
            metavar='python_version',
            default=None,
            help=("Only download wheels compatible with Python "
                  "interpreter version <version>. If not specified, then the "
                  "current system interpreter minor version is used. A major "
                  "version (e.g. '2') can be specified to match all "
                  "minor revs of that major version.  A minor version "
                  "(e.g. '34') can also be specified."),
        )

        cmd_opts.add_option(
            '--implementation',
            dest='implementation',
            metavar='implementation',
            default=None,
            help=("Only download wheels compatible with Python "
                  "implementation <implementation>, e.g. 'pp', 'jy', 'cp', "
                  " or 'ip'. If not specified, then the current "
                  "interpreter implementation is used.  Use 'py' to force "
                  "implementation-agnostic wheels."),
        )

        cmd_opts.add_option(
            '--abi',
            dest='abi',
            metavar='abi',
            default=None,
            help=("Only download wheels compatible with Python "
                  "abi <abi>, e.g. 'pypy_41'.  If not specified, then the "
                  "current interpreter abi tag is used.  Generally "
                  "you will need to specify --implementation, "
                  "--platform, and --python-version when using "
                  "this option."),
        )

        index_opts = cmdoptions.make_option_group(
            cmdoptions.non_deprecated_index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    def run(self, options, args):
        options.ignore_installed = True

        if options.python_version:
            python_versions = [options.python_version]
        else:
            python_versions = None

        dist_restriction_set = any([
            options.python_version,
            options.platform,
            options.abi,
            options.implementation,
        ])
        binary_only = FormatControl(set(), set([':all:']))
        if dist_restriction_set and options.format_control != binary_only:
            raise CommandError(
                "--only-binary=:all: must be set and --no-binary must not "
                "be set (or must be set to :none:) when restricting platform "
                "and interpreter constraints using --python-version, "
                "--platform, --abi, or --implementation."
            )

        options.src_dir = os.path.abspath(options.src_dir)
        options.download_dir = normalize_path(options.download_dir)

        ensure_dir(options.download_dir)

        with self._build_session(options) as session:
            finder = self._build_package_finder(
                options=options,
                session=session,
                platform=options.platform,
                python_versions=python_versions,
                abi=options.abi,
                implementation=options.implementation,
            )
            build_delete = (not (options.no_clean or options.build_dir))
            if options.cache_dir and not check_path_owner(options.cache_dir):
                logger.warning(
                    "The directory '%s' or its parent directory is not owned "
                    "by the current user and caching wheels has been "
                    "disabled. check the permissions and owner of that "
                    "directory. If executing pip with sudo, you may want "
                    "sudo's -H flag.",
                    options.cache_dir,
                )
                options.cache_dir = None

            with BuildDirectory(options.build_dir,
                                delete=build_delete) as build_dir:

                requirement_set = RequirementSet(
                    build_dir=build_dir,
                    src_dir=options.src_dir,
                    download_dir=options.download_dir,
                    ignore_installed=True,
                    ignore_dependencies=options.ignore_dependencies,
                    session=session,
                    isolated=options.isolated_mode,
                    require_hashes=options.require_hashes
                )
                self.populate_requirement_set(
                    requirement_set,
                    args,
                    options,
                    finder,
                    session,
                    self.name,
                    None
                )

                if not requirement_set.has_requirements:
                    return

                requirement_set.prepare_files(finder)

                downloaded = ' '.join([
                    req.name for req in requirement_set.successfully_downloaded
                ])
                if downloaded:
                    logger.info(
                        'Successfully downloaded %s', downloaded
                    )

                # Clean up
                if not options.no_clean:
                    requirement_set.cleanup_files()

        return requirement_set
