"""
Requirements file parsing
"""

from __future__ import absolute_import

import os
import re
import shlex
import sys
import optparse
import warnings

from pip._vendor.six.moves.urllib import parse as urllib_parse
from pip._vendor.six.moves import filterfalse

import pip
from pip.download import get_file_content
from pip.req.req_install import InstallRequirement
from pip.exceptions import (RequirementsFileParseError)
from pip.utils.deprecation import RemovedInPip10Warning
from pip import cmdoptions

__all__ = ['parse_requirements']

SCHEME_RE = re.compile(r'^(http|https|file):', re.I)
COMMENT_RE = re.compile(r'(^|\s)+#.*$')

SUPPORTED_OPTIONS = [
    cmdoptions.constraints,
    cmdoptions.editable,
    cmdoptions.requirements,
    cmdoptions.no_index,
    cmdoptions.index_url,
    cmdoptions.find_links,
    cmdoptions.extra_index_url,
    cmdoptions.allow_external,
    cmdoptions.allow_all_external,
    cmdoptions.no_allow_external,
    cmdoptions.allow_unsafe,
    cmdoptions.no_allow_unsafe,
    cmdoptions.use_wheel,
    cmdoptions.no_use_wheel,
    cmdoptions.always_unzip,
    cmdoptions.no_binary,
    cmdoptions.only_binary,
    cmdoptions.pre,
    cmdoptions.process_dependency_links,
    cmdoptions.trusted_host,
    cmdoptions.require_hashes,
]

# options to be passed to requirements
SUPPORTED_OPTIONS_REQ = [
    cmdoptions.install_options,
    cmdoptions.global_options,
    cmdoptions.hash,
]

# the 'dest' string values
SUPPORTED_OPTIONS_REQ_DEST = [o().dest for o in SUPPORTED_OPTIONS_REQ]


def parse_requirements(filename, finder=None, comes_from=None, options=None,
                       session=None, constraint=False, wheel_cache=None):
    """Parse a requirements file and yield InstallRequirement instances.

    :param filename:    Path or url of requirements file.
    :param finder:      Instance of pip.index.PackageFinder.
    :param comes_from:  Origin description of requirements.
    :param options:     cli options.
    :param session:     Instance of pip.download.PipSession.
    :param constraint:  If true, parsing a constraint file rather than
        requirements file.
    :param wheel_cache: Instance of pip.wheel.WheelCache
    """
    if session is None:
        raise TypeError(
            "parse_requirements() missing 1 required keyword argument: "
            "'session'"
        )

    _, content = get_file_content(
        filename, comes_from=comes_from, session=session
    )

    lines_enum = preprocess(content, options)

    for line_number, line in lines_enum:
        req_iter = process_line(line, filename, line_number, finder,
                                comes_from, options, session, wheel_cache,
                                constraint=constraint)
        for req in req_iter:
            yield req


def preprocess(content, options):
    """Split, filter, and join lines, and return a line iterator

    :param content: the content of the requirements file
    :param options: cli options
    """
    lines_enum = enumerate(content.splitlines(), start=1)
    lines_enum = join_lines(lines_enum)
    lines_enum = ignore_comments(lines_enum)
    lines_enum = skip_regex(lines_enum, options)
    return lines_enum


def process_line(line, filename, line_number, finder=None, comes_from=None,
                 options=None, session=None, wheel_cache=None,
                 constraint=False):
    """Process a single requirements line; This can result in creating/yielding
    requirements, or updating the finder.

    For lines that contain requirements, the only options that have an effect
    are from SUPPORTED_OPTIONS_REQ, and they are scoped to the
    requirement. Other options from SUPPORTED_OPTIONS may be present, but are
    ignored.

    For lines that do not contain requirements, the only options that have an
    effect are from SUPPORTED_OPTIONS. Options from SUPPORTED_OPTIONS_REQ may
    be present, but are ignored. These lines may contain multiple options
    (although our docs imply only one is supported), and all our parsed and
    affect the finder.

    :param constraint: If True, parsing a constraints file.
    :param options: OptionParser options that we may update
    """
    parser = build_parser()
    defaults = parser.get_default_values()
    defaults.index_url = None
    if finder:
        # `finder.format_control` will be updated during parsing
        defaults.format_control = finder.format_control
    args_str, options_str = break_args_options(line)
    if sys.version_info < (2, 7, 3):
        # Prior to 2.7.3, shlex cannot deal with unicode entries
        options_str = options_str.encode('utf8')
    opts, _ = parser.parse_args(shlex.split(options_str), defaults)

    # preserve for the nested code path
    line_comes_from = '%s %s (line %s)' % (
        '-c' if constraint else '-r', filename, line_number)

    # yield a line requirement
    if args_str:
        isolated = options.isolated_mode if options else False
        if options:
            cmdoptions.check_install_build_global(options, opts)
        # get the options that apply to requirements
        req_options = {}
        for dest in SUPPORTED_OPTIONS_REQ_DEST:
            if dest in opts.__dict__ and opts.__dict__[dest]:
                req_options[dest] = opts.__dict__[dest]
        yield InstallRequirement.from_line(
            args_str, line_comes_from, constraint=constraint,
            isolated=isolated, options=req_options, wheel_cache=wheel_cache
        )

    # yield an editable requirement
    elif opts.editables:
        isolated = options.isolated_mode if options else False
        default_vcs = options.default_vcs if options else None
        yield InstallRequirement.from_editable(
            opts.editables[0], comes_from=line_comes_from,
            constraint=constraint, default_vcs=default_vcs, isolated=isolated,
            wheel_cache=wheel_cache
        )

    # parse a nested requirements file
    elif opts.requirements or opts.constraints:
        if opts.requirements:
            req_path = opts.requirements[0]
            nested_constraint = False
        else:
            req_path = opts.constraints[0]
            nested_constraint = True
        # original file is over http
        if SCHEME_RE.search(filename):
            # do a url join so relative paths work
            req_path = urllib_parse.urljoin(filename, req_path)
        # original file and nested file are paths
        elif not SCHEME_RE.search(req_path):
            # do a join so relative paths work
            req_path = os.path.join(os.path.dirname(filename), req_path)
        # TODO: Why not use `comes_from='-r {} (line {})'` here as well?
        parser = parse_requirements(
            req_path, finder, comes_from, options, session,
            constraint=nested_constraint, wheel_cache=wheel_cache
        )
        for req in parser:
            yield req

    # percolate hash-checking option upward
    elif opts.require_hashes:
        options.require_hashes = opts.require_hashes

    # set finder options
    elif finder:
        if opts.allow_external:
            warnings.warn(
                "--allow-external has been deprecated and will be removed in "
                "the future. Due to changes in the repository protocol, it no "
                "longer has any effect.",
                RemovedInPip10Warning,
            )

        if opts.allow_all_external:
            warnings.warn(
                "--allow-all-external has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        if opts.allow_unverified:
            warnings.warn(
                "--allow-unverified has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        if opts.index_url:
            finder.index_urls = [opts.index_url]
        if opts.use_wheel is False:
            finder.use_wheel = False
            pip.index.fmt_ctl_no_use_wheel(finder.format_control)
        if opts.no_index is True:
            finder.index_urls = []
        if opts.extra_index_urls:
            finder.index_urls.extend(opts.extra_index_urls)
        if opts.find_links:
            # FIXME: it would be nice to keep track of the source
            # of the find_links: support a find-links local path
            # relative to a requirements file.
            value = opts.find_links[0]
            req_dir = os.path.dirname(os.path.abspath(filename))
            relative_to_reqs_file = os.path.join(req_dir, value)
            if os.path.exists(relative_to_reqs_file):
                value = relative_to_reqs_file
            finder.find_links.append(value)
        if opts.pre:
            finder.allow_all_prereleases = True
        if opts.process_dependency_links:
            finder.process_dependency_links = True
        if opts.trusted_hosts:
            finder.secure_origins.extend(
                ("*", host, "*") for host in opts.trusted_hosts)


def break_args_options(line):
    """Break up the line into an args and options string.  We only want to shlex
    (and then optparse) the options, not the args.  args can contain markers
    which are corrupted by shlex.
    """
    tokens = line.split(' ')
    args = []
    options = tokens[:]
    for token in tokens:
        if token.startswith('-') or token.startswith('--'):
            break
        else:
            args.append(token)
            options.pop(0)
    return ' '.join(args), ' '.join(options)


def build_parser():
    """
    Return a parser for parsing requirement lines
    """
    parser = optparse.OptionParser(add_help_option=False)

    option_factories = SUPPORTED_OPTIONS + SUPPORTED_OPTIONS_REQ
    for option_factory in option_factories:
        option = option_factory()
        parser.add_option(option)

    # By default optparse sys.exits on parsing errors. We want to wrap
    # that in our own exception.
    def parser_exit(self, msg):
        raise RequirementsFileParseError(msg)
    parser.exit = parser_exit

    return parser


def join_lines(lines_enum):
    """Joins a line ending in '\' with the previous line (except when following
    comments).  The joined line takes on the index of the first line.
    """
    primary_line_number = None
    new_line = []
    for line_number, line in lines_enum:
        if not line.endswith('\\') or COMMENT_RE.match(line):
            if COMMENT_RE.match(line):
                # this ensures comments are always matched later
                line = ' ' + line
            if new_line:
                new_line.append(line)
                yield primary_line_number, ''.join(new_line)
                new_line = []
            else:
                yield line_number, line
        else:
            if not new_line:
                primary_line_number = line_number
            new_line.append(line.strip('\\'))

    # last line contains \
    if new_line:
        yield primary_line_number, ''.join(new_line)

    # TODO: handle space after '\'.


def ignore_comments(lines_enum):
    """
    Strips comments and filter empty lines.
    """
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line_number, line


def skip_regex(lines_enum, options):
    """
    Skip lines that match '--skip-requirements-regex' pattern

    Note: the regex pattern is only built once
    """
    skip_regex = options.skip_requirements_regex if options else None
    if skip_regex:
        pattern = re.compile(skip_regex)
        lines_enum = filterfalse(
            lambda e: pattern.search(e[1]),
            lines_enum)
    return lines_enum
