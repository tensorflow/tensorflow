#! /usr/bin/env python
# encoding: utf-8
#
# written by Sylvain Rouquette, 2014

'''

This is an extra tool, not bundled with the default waf binary.
To add the cpplint tool to the waf file:
$ ./waf-light --tools=compat15,cpplint

this tool also requires cpplint for python.
If you have PIP, you can install it like this: pip install cpplint

When using this tool, the wscript will look like:

    def options(opt):
        opt.load('compiler_cxx cpplint')

    def configure(conf):
        conf.load('compiler_cxx cpplint')
        # optional, you can also specify them on the command line
        conf.env.CPPLINT_FILTERS = ','.join((
            '-whitespace/newline',      # c++11 lambda
            '-readability/braces',      # c++11 constructor
            '-whitespace/braces',       # c++11 constructor
            '-build/storage_class',     # c++11 for-range
            '-whitespace/blank_line',   # user pref
            '-whitespace/labels'        # user pref
            ))

    def build(bld):
        bld(features='cpplint', source='main.cpp', target='app')
        # add include files, because they aren't usually built
        bld(features='cpplint', source=bld.path.ant_glob('**/*.hpp'))
'''

from __future__ import absolute_import
import sys, re
import logging
from waflib import Errors, Task, TaskGen, Logs, Options, Node, Utils


critical_errors = 0
CPPLINT_FORMAT = '[CPPLINT] %(filename)s:\nline %(linenum)s, severity %(confidence)s, category: %(category)s\n%(message)s\n'
RE_EMACS = re.compile(r'(?P<filename>.*):(?P<linenum>\d+):  (?P<message>.*)  \[(?P<category>.*)\] \[(?P<confidence>\d+)\]')
CPPLINT_RE = {
    'waf': RE_EMACS,
    'emacs': RE_EMACS,
    'vs7': re.compile(r'(?P<filename>.*)\((?P<linenum>\d+)\):  (?P<message>.*)  \[(?P<category>.*)\] \[(?P<confidence>\d+)\]'),
    'eclipse': re.compile(r'(?P<filename>.*):(?P<linenum>\d+): warning: (?P<message>.*)  \[(?P<category>.*)\] \[(?P<confidence>\d+)\]'),
}
CPPLINT_STR = ('${CPPLINT} '
               '--verbose=${CPPLINT_LEVEL} '
               '--output=${CPPLINT_OUTPUT} '
               '--filter=${CPPLINT_FILTERS} '
               '--root=${CPPLINT_ROOT} '
               '--linelength=${CPPLINT_LINE_LENGTH} ')


def options(opt):
    opt.add_option('--cpplint-filters', type='string',
                   default='', dest='CPPLINT_FILTERS',
                   help='add filters to cpplint')
    opt.add_option('--cpplint-length', type='int',
                   default=80, dest='CPPLINT_LINE_LENGTH',
                   help='specify the line length (default: 80)')
    opt.add_option('--cpplint-level', default=1, type='int', dest='CPPLINT_LEVEL',
                   help='specify the log level (default: 1)')
    opt.add_option('--cpplint-break', default=5, type='int', dest='CPPLINT_BREAK',
                   help='break the build if error >= level (default: 5)')
    opt.add_option('--cpplint-root', type='string',
                   default='', dest='CPPLINT_ROOT',
                   help='root directory used to derive header guard')
    opt.add_option('--cpplint-skip', action='store_true',
                   default=False, dest='CPPLINT_SKIP',
                   help='skip cpplint during build')
    opt.add_option('--cpplint-output', type='string',
                   default='waf', dest='CPPLINT_OUTPUT',
                   help='select output format (waf, emacs, vs7, eclipse)')


def configure(conf):
    try:
        conf.find_program('cpplint', var='CPPLINT')
    except Errors.ConfigurationError:
        conf.env.CPPLINT_SKIP = True


class cpplint_formatter(Logs.formatter, object):
    def __init__(self, fmt):
        logging.Formatter.__init__(self, CPPLINT_FORMAT)
        self.fmt = fmt

    def format(self, rec):
        if self.fmt == 'waf':
            result = CPPLINT_RE[self.fmt].match(rec.msg).groupdict()
            rec.msg = CPPLINT_FORMAT % result
        if rec.levelno <= logging.INFO:
            rec.c1 = Logs.colors.CYAN
        return super(cpplint_formatter, self).format(rec)


class cpplint_handler(Logs.log_handler, object):
    def __init__(self, stream=sys.stderr, **kw):
        super(cpplint_handler, self).__init__(stream, **kw)
        self.stream = stream

    def emit(self, rec):
        rec.stream = self.stream
        self.emit_override(rec)
        self.flush()


class cpplint_wrapper(object):
    def __init__(self, logger, threshold, fmt):
        self.logger = logger
        self.threshold = threshold
        self.fmt = fmt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_value, Utils.subprocess.CalledProcessError):
            messages = [m for m in exc_value.output.splitlines() 
                        if 'Done processing' not in m 
                        and 'Total errors found' not in m]
            for message in messages:
                self.write(message)
            return True

    def write(self, message):
        global critical_errors
        result = CPPLINT_RE[self.fmt].match(message)
        if not result:
            return
        level = int(result.groupdict()['confidence'])
        if level >= self.threshold:
            critical_errors += 1
        if level <= 2:
            self.logger.info(message)
        elif level <= 4:
            self.logger.warning(message)
        else:
            self.logger.error(message)


cpplint_logger = None
def get_cpplint_logger(fmt):
    global cpplint_logger
    if cpplint_logger:
        return cpplint_logger
    cpplint_logger = logging.getLogger('cpplint')
    hdlr = cpplint_handler()
    hdlr.setFormatter(cpplint_formatter(fmt))
    cpplint_logger.addHandler(hdlr)
    cpplint_logger.setLevel(logging.DEBUG)
    return cpplint_logger


class cpplint(Task.Task):
    color = 'PINK'

    def __init__(self, *k, **kw):
        super(cpplint, self).__init__(*k, **kw)

    def run(self):
        global critical_errors
        with cpplint_wrapper(get_cpplint_logger(self.env.CPPLINT_OUTPUT), self.env.CPPLINT_BREAK, self.env.CPPLINT_OUTPUT):
            params = {key: str(self.env[key]) for key in self.env if 'CPPLINT_' in key}
            if params['CPPLINT_OUTPUT'] is 'waf':
                params['CPPLINT_OUTPUT'] = 'emacs'
            params['CPPLINT'] = self.env.get_flat('CPPLINT')
            cmd = Utils.subst_vars(CPPLINT_STR, params)
            env = self.env.env or None
            Utils.subprocess.check_output(cmd + self.inputs[0].abspath(),
                                          stderr=Utils.subprocess.STDOUT,
                                          env=env, shell=True)
        return critical_errors

@TaskGen.extension('.h', '.hh', '.hpp', '.hxx')
def cpplint_includes(self, node):
    pass

@TaskGen.feature('cpplint')
@TaskGen.before_method('process_source')
def post_cpplint(self):
    if not self.env.CPPLINT_INITIALIZED:
        for key, value in Options.options.__dict__.items():
            if not key.startswith('CPPLINT_') or self.env[key]:
                continue
            self.env[key] = value
        self.env.CPPLINT_INITIALIZED = True

    if self.env.CPPLINT_SKIP:
        return

    if not self.env.CPPLINT_OUTPUT in CPPLINT_RE:
        return

    for src in self.to_list(getattr(self, 'source', [])):
        if isinstance(src, Node.Node):
            node = src
        else:
            node = self.path.find_or_declare(src)
        if not node:
            self.bld.fatal('Could not find %r' % src)
        self.create_task('cpplint', node)
