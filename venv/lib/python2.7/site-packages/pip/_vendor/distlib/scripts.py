# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2015 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
from io import BytesIO
import logging
import os
import re
import struct
import sys

from .compat import sysconfig, detect_encoding, ZipFile
from .resources import finder
from .util import (FileOperator, get_export_entry, convert_path,
                   get_executable, in_venv)

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST = '''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
 <assemblyIdentity version="1.0.0.0"
 processorArchitecture="X86"
 name="%s"
 type="win32"/>

 <!-- Identify the application security requirements. -->
 <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
 <security>
 <requestedPrivileges>
 <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
 </requestedPrivileges>
 </security>
 </trustInfo>
</assembly>'''.strip()

# check if Python is called on the first line with this expression
FIRST_LINE_RE = re.compile(b'^#!.*pythonw?[0-9.]*([ \t].*)?$')
SCRIPT_TEMPLATE = '''# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import sys, re

    def _resolve(module, func):
        __import__(module)
        mod = sys.modules[module]
        parts = func.split('.')
        result = getattr(mod, parts.pop(0))
        for p in parts:
            result = getattr(result, p)
        return result

    try:
        sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])

        func = _resolve('%(module)s', '%(func)s')
        rc = func() # None interpreted as 0
    except Exception as e:  # only supporting Python >= 2.6
        sys.stderr.write('%%s\\n' %% e)
        rc = 1
    sys.exit(rc)
'''


def _enquote_executable(executable):
    if ' ' in executable:
        # make sure we quote only the executable in case of env
        # for example /usr/bin/env "/dir with spaces/bin/jython"
        # instead of "/usr/bin/env /dir with spaces/bin/jython"
        # otherwise whole
        if executable.startswith('/usr/bin/env '):
            env, _executable = executable.split(' ', 1)
            if ' ' in _executable and not _executable.startswith('"'):
                executable = '%s "%s"' % (env, _executable)
        else:
            if not executable.startswith('"'):
                executable = '"%s"' % executable
    return executable


class ScriptMaker(object):
    """
    A class to copy or create scripts from source scripts or callable
    specifications.
    """
    script_template = SCRIPT_TEMPLATE

    executable = None  # for shebangs

    def __init__(self, source_dir, target_dir, add_launchers=True,
                 dry_run=False, fileop=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.add_launchers = add_launchers
        self.force = False
        self.clobber = False
        # It only makes sense to set mode bits on POSIX.
        self.set_mode = (os.name == 'posix') or (os.name == 'java' and
                                                 os._name == 'posix')
        self.variants = set(('', 'X.Y'))
        self._fileop = fileop or FileOperator(dry_run)

        self._is_nt = os.name == 'nt' or (
            os.name == 'java' and os._name == 'nt')

    def _get_alternate_executable(self, executable, options):
        if options.get('gui', False) and self._is_nt:  # pragma: no cover
            dn, fn = os.path.split(executable)
            fn = fn.replace('python', 'pythonw')
            executable = os.path.join(dn, fn)
        return executable

    if sys.platform.startswith('java'):  # pragma: no cover
        def _is_shell(self, executable):
            """
            Determine if the specified executable is a script
            (contains a #! line)
            """
            try:
                with open(executable) as fp:
                    return fp.read(2) == '#!'
            except (OSError, IOError):
                logger.warning('Failed to open %s', executable)
                return False

        def _fix_jython_executable(self, executable):
            if self._is_shell(executable):
                # Workaround for Jython is not needed on Linux systems.
                import java

                if java.lang.System.getProperty('os.name') == 'Linux':
                    return executable
            elif executable.lower().endswith('jython.exe'):
                # Use wrapper exe for Jython on Windows
                return executable
            return '/usr/bin/env %s' % executable

    def _get_shebang(self, encoding, post_interp=b'', options=None):
        enquote = True
        if self.executable:
            executable = self.executable
            enquote = False     # assume this will be taken care of
        elif not sysconfig.is_python_build():
            executable = get_executable()
        elif in_venv():  # pragma: no cover
            executable = os.path.join(sysconfig.get_path('scripts'),
                            'python%s' % sysconfig.get_config_var('EXE'))
        else:  # pragma: no cover
            executable = os.path.join(
                sysconfig.get_config_var('BINDIR'),
               'python%s%s' % (sysconfig.get_config_var('VERSION'),
                               sysconfig.get_config_var('EXE')))
        if options:
            executable = self._get_alternate_executable(executable, options)

        if sys.platform.startswith('java'):  # pragma: no cover
            executable = self._fix_jython_executable(executable)
        # Normalise case for Windows
        executable = os.path.normcase(executable)
        # If the user didn't specify an executable, it may be necessary to
        # cater for executable paths with spaces (not uncommon on Windows)
        if enquote:
            executable = _enquote_executable(executable)
        # Issue #51: don't use fsencode, since we later try to
        # check that the shebang is decodable using utf-8.
        executable = executable.encode('utf-8')
        # in case of IronPython, play safe and enable frames support
        if (sys.platform == 'cli' and '-X:Frames' not in post_interp
            and '-X:FullFrames' not in post_interp):  # pragma: no cover
            post_interp += b' -X:Frames'
        shebang = b'#!' + executable + post_interp + b'\n'
        # Python parser starts to read a script using UTF-8 until
        # it gets a #coding:xxx cookie. The shebang has to be the
        # first line of a file, the #coding:xxx cookie cannot be
        # written before. So the shebang has to be decodable from
        # UTF-8.
        try:
            shebang.decode('utf-8')
        except UnicodeDecodeError:  # pragma: no cover
            raise ValueError(
                'The shebang (%r) is not decodable from utf-8' % shebang)
        # If the script is encoded to a custom encoding (use a
        # #coding:xxx cookie), the shebang has to be decodable from
        # the script encoding too.
        if encoding != 'utf-8':
            try:
                shebang.decode(encoding)
            except UnicodeDecodeError:  # pragma: no cover
                raise ValueError(
                    'The shebang (%r) is not decodable '
                    'from the script encoding (%r)' % (shebang, encoding))
        return shebang

    def _get_script_text(self, entry):
        return self.script_template % dict(module=entry.prefix,
                                           func=entry.suffix)

    manifest = _DEFAULT_MANIFEST

    def get_manifest(self, exename):
        base = os.path.basename(exename)
        return self.manifest % base

    def _write_script(self, names, shebang, script_bytes, filenames, ext):
        use_launcher = self.add_launchers and self._is_nt
        linesep = os.linesep.encode('utf-8')
        if not use_launcher:
            script_bytes = shebang + linesep + script_bytes
        else:  # pragma: no cover
            if ext == 'py':
                launcher = self._get_launcher('t')
            else:
                launcher = self._get_launcher('w')
            stream = BytesIO()
            with ZipFile(stream, 'w') as zf:
                zf.writestr('__main__.py', script_bytes)
            zip_data = stream.getvalue()
            script_bytes = launcher + shebang + linesep + zip_data
        for name in names:
            outname = os.path.join(self.target_dir, name)
            if use_launcher:  # pragma: no cover
                n, e = os.path.splitext(outname)
                if e.startswith('.py'):
                    outname = n
                outname = '%s.exe' % outname
                try:
                    self._fileop.write_binary_file(outname, script_bytes)
                except Exception:
                    # Failed writing an executable - it might be in use.
                    logger.warning('Failed to write executable - trying to '
                                   'use .deleteme logic')
                    dfname = '%s.deleteme' % outname
                    if os.path.exists(dfname):
                        os.remove(dfname)       # Not allowed to fail here
                    os.rename(outname, dfname)  # nor here
                    self._fileop.write_binary_file(outname, script_bytes)
                    logger.debug('Able to replace executable using '
                                 '.deleteme logic')
                    try:
                        os.remove(dfname)
                    except Exception:
                        pass    # still in use - ignore error
            else:
                if self._is_nt and not outname.endswith('.' + ext):  # pragma: no cover
                    outname = '%s.%s' % (outname, ext)
                if os.path.exists(outname) and not self.clobber:
                    logger.warning('Skipping existing file %s', outname)
                    continue
                self._fileop.write_binary_file(outname, script_bytes)
                if self.set_mode:
                    self._fileop.set_executable_mode([outname])
            filenames.append(outname)

    def _make_script(self, entry, filenames, options=None):
        post_interp = b''
        if options:
            args = options.get('interpreter_args', [])
            if args:
                args = ' %s' % ' '.join(args)
                post_interp = args.encode('utf-8')
        shebang = self._get_shebang('utf-8', post_interp, options=options)
        script = self._get_script_text(entry).encode('utf-8')
        name = entry.name
        scriptnames = set()
        if '' in self.variants:
            scriptnames.add(name)
        if 'X' in self.variants:
            scriptnames.add('%s%s' % (name, sys.version[0]))
        if 'X.Y' in self.variants:
            scriptnames.add('%s-%s' % (name, sys.version[:3]))
        if options and options.get('gui', False):
            ext = 'pyw'
        else:
            ext = 'py'
        self._write_script(scriptnames, shebang, script, filenames, ext)

    def _copy_script(self, script, filenames):
        adjust = False
        script = os.path.join(self.source_dir, convert_path(script))
        outname = os.path.join(self.target_dir, os.path.basename(script))
        if not self.force and not self._fileop.newer(script, outname):
            logger.debug('not copying %s (up-to-date)', script)
            return

        # Always open the file, but ignore failures in dry-run mode --
        # that way, we'll get accurate feedback if we can read the
        # script.
        try:
            f = open(script, 'rb')
        except IOError:  # pragma: no cover
            if not self.dry_run:
                raise
            f = None
        else:
            first_line = f.readline()
            if not first_line:  # pragma: no cover
                logger.warning('%s: %s is an empty file (skipping)',
                               self.get_command_name(),  script)
                return

            match = FIRST_LINE_RE.match(first_line.replace(b'\r\n', b'\n'))
            if match:
                adjust = True
                post_interp = match.group(1) or b''

        if not adjust:
            if f:
                f.close()
            self._fileop.copy_file(script, outname)
            if self.set_mode:
                self._fileop.set_executable_mode([outname])
            filenames.append(outname)
        else:
            logger.info('copying and adjusting %s -> %s', script,
                        self.target_dir)
            if not self._fileop.dry_run:
                encoding, lines = detect_encoding(f.readline)
                f.seek(0)
                shebang = self._get_shebang(encoding, post_interp)
                if b'pythonw' in first_line:  # pragma: no cover
                    ext = 'pyw'
                else:
                    ext = 'py'
                n = os.path.basename(outname)
                self._write_script([n], shebang, f.read(), filenames, ext)
            if f:
                f.close()

    @property
    def dry_run(self):
        return self._fileop.dry_run

    @dry_run.setter
    def dry_run(self, value):
        self._fileop.dry_run = value

    if os.name == 'nt' or (os.name == 'java' and os._name == 'nt'):  # pragma: no cover
        # Executable launcher support.
        # Launchers are from https://bitbucket.org/vinay.sajip/simple_launcher/

        def _get_launcher(self, kind):
            if struct.calcsize('P') == 8:   # 64-bit
                bits = '64'
            else:
                bits = '32'
            name = '%s%s.exe' % (kind, bits)
            # Issue 31: don't hardcode an absolute package name, but
            # determine it relative to the current package
            distlib_package = __name__.rsplit('.', 1)[0]
            result = finder(distlib_package).find(name).bytes
            return result

    # Public API follows

    def make(self, specification, options=None):
        """
        Make a script.

        :param specification: The specification, which is either a valid export
                              entry specification (to make a script from a
                              callable) or a filename (to make a script by
                              copying from a source location).
        :param options: A dictionary of options controlling script generation.
        :return: A list of all absolute pathnames written to.
        """
        filenames = []
        entry = get_export_entry(specification)
        if entry is None:
            self._copy_script(specification, filenames)
        else:
            self._make_script(entry, filenames, options=options)
        return filenames

    def make_multiple(self, specifications, options=None):
        """
        Take a list of specifications and make scripts from them,
        :param specifications: A list of specifications.
        :return: A list of all absolute pathnames written to,
        """
        filenames = []
        for specification in specifications:
            filenames.extend(self.make(specification, options))
        return filenames
