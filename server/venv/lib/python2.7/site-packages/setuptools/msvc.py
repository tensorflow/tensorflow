"""
Improved support for Microsoft Visual C++ compilers.

Known supported compilers:
--------------------------
Microsoft Visual C++ 9.0:
    Microsoft Visual C++ Compiler for Python 2.7 (x86, amd64);
    Microsoft Windows SDK 7.0 (x86, x64, ia64);
    Microsoft Windows SDK 6.1 (x86, x64, ia64)

Microsoft Visual C++ 10.0:
    Microsoft Windows SDK 7.1 (x86, x64, ia64)

Microsoft Visual C++ 14.0:
    Microsoft Visual C++ Build Tools 2015 (x86, x64, arm)
"""

import os
import sys
import platform
import itertools
import distutils.errors
from packaging.version import LegacyVersion

from six.moves import filterfalse

from .monkey import get_unpatched

if platform.system() == 'Windows':
    from six.moves import winreg
    safe_env = os.environ
else:
    """
    Mock winreg and environ so the module can be imported
    on this platform.
    """

    class winreg:
        HKEY_USERS = None
        HKEY_CURRENT_USER = None
        HKEY_LOCAL_MACHINE = None
        HKEY_CLASSES_ROOT = None

    safe_env = dict()

try:
    from distutils.msvc9compiler import Reg
except ImportError:
    pass


def msvc9_find_vcvarsall(version):
    """
    Patched "distutils.msvc9compiler.find_vcvarsall" to use the standalone
    compiler build for Python (VCForPython). Fall back to original behavior
    when the standalone compiler is not available.

    Redirect the path of "vcvarsall.bat".

    Known supported compilers
    -------------------------
    Microsoft Visual C++ 9.0:
        Microsoft Visual C++ Compiler for Python 2.7 (x86, amd64)

    Parameters
    ----------
    version: float
        Required Microsoft Visual C++ version.

    Return
    ------
    vcvarsall.bat path: str
    """
    VC_BASE = r'Software\%sMicrosoft\DevDiv\VCForPython\%0.1f'
    key = VC_BASE % ('', version)
    try:
        # Per-user installs register the compiler path here
        productdir = Reg.get_value(key, "installdir")
    except KeyError:
        try:
            # All-user installs on a 64-bit system register here
            key = VC_BASE % ('Wow6432Node\\', version)
            productdir = Reg.get_value(key, "installdir")
        except KeyError:
            productdir = None

    if productdir:
        vcvarsall = os.path.os.path.join(productdir, "vcvarsall.bat")
        if os.path.isfile(vcvarsall):
            return vcvarsall

    return get_unpatched(msvc9_find_vcvarsall)(version)


def msvc9_query_vcvarsall(ver, arch='x86', *args, **kwargs):
    """
    Patched "distutils.msvc9compiler.query_vcvarsall" for support standalones
    compilers.

    Set environment without use of "vcvarsall.bat".

    Known supported compilers
    -------------------------
    Microsoft Visual C++ 9.0:
        Microsoft Visual C++ Compiler for Python 2.7 (x86, amd64);
        Microsoft Windows SDK 7.0 (x86, x64, ia64);
        Microsoft Windows SDK 6.1 (x86, x64, ia64)

    Microsoft Visual C++ 10.0:
        Microsoft Windows SDK 7.1 (x86, x64, ia64)

    Parameters
    ----------
    ver: float
        Required Microsoft Visual C++ version.
    arch: str
        Target architecture.

    Return
    ------
    environment: dict
    """
    # Try to get environement from vcvarsall.bat (Classical way)
    try:
        orig = get_unpatched(msvc9_query_vcvarsall)
        return orig(ver, arch, *args, **kwargs)
    except distutils.errors.DistutilsPlatformError:
        # Pass error if Vcvarsall.bat is missing
        pass
    except ValueError:
        # Pass error if environment not set after executing vcvarsall.bat
        pass

    # If error, try to set environment directly
    try:
        return EnvironmentInfo(arch, ver).return_env()
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, ver, arch)
        raise


def msvc14_get_vc_env(plat_spec):
    """
    Patched "distutils._msvccompiler._get_vc_env" for support standalones
    compilers.

    Set environment without use of "vcvarsall.bat".

    Known supported compilers
    -------------------------
    Microsoft Visual C++ 14.0:
        Microsoft Visual C++ Build Tools 2015 (x86, x64, arm)

    Parameters
    ----------
    plat_spec: str
        Target architecture.

    Return
    ------
    environment: dict
    """
    # Try to get environment from vcvarsall.bat (Classical way)
    try:
        return get_unpatched(msvc14_get_vc_env)(plat_spec)
    except distutils.errors.DistutilsPlatformError:
        # Pass error Vcvarsall.bat is missing
        pass

    # If error, try to set environment directly
    try:
        return EnvironmentInfo(plat_spec, vc_min_ver=14.0).return_env()
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, 14.0)
        raise


def msvc14_gen_lib_options(*args, **kwargs):
    """
    Patched "distutils._msvccompiler.gen_lib_options" for fix
    compatibility between "numpy.distutils" and "distutils._msvccompiler"
    (for Numpy < 1.11.2)
    """
    if "numpy.distutils" in sys.modules:
        import numpy as np
        if LegacyVersion(np.__version__) < LegacyVersion('1.11.2'):
            return np.distutils.ccompiler.gen_lib_options(*args, **kwargs)
    return get_unpatched(msvc14_gen_lib_options)(*args, **kwargs)


def _augment_exception(exc, version, arch=''):
    """
    Add details to the exception message to help guide the user
    as to what action will resolve it.
    """
    # Error if MSVC++ directory not found or environment not set
    message = exc.args[0]

    if "vcvarsall" in message.lower() or "visual c" in message.lower():
        # Special error message if MSVC++ not installed
        tmpl = 'Microsoft Visual C++ {version:0.1f} is required.'
        message = tmpl.format(**locals())
        msdownload = 'www.microsoft.com/download/details.aspx?id=%d'
        if version == 9.0:
            if arch.lower().find('ia64') > -1:
                # For VC++ 9.0, if IA64 support is needed, redirect user
                # to Windows SDK 7.0
                message += ' Get it with "Microsoft Windows SDK 7.0": '
                message += msdownload % 3138
            else:
                # For VC++ 9.0 redirect user to Vc++ for Python 2.7 :
                # This redirection link is maintained by Microsoft.
                # Contact vspython@microsoft.com if it needs updating.
                message += ' Get it from http://aka.ms/vcpython27'
        elif version == 10.0:
            # For VC++ 10.0 Redirect user to Windows SDK 7.1
            message += ' Get it with "Microsoft Windows SDK 7.1": '
            message += msdownload % 8279
        elif version >= 14.0:
            # For VC++ 14.0 Redirect user to Visual C++ Build Tools
            message += (' Get it with "Microsoft Visual C++ Build Tools": '
                        r'http://landinghub.visualstudio.com/'
                        'visual-cpp-build-tools')

    exc.args = (message, )


class PlatformInfo:
    """
    Current and Target Architectures informations.

    Parameters
    ----------
    arch: str
        Target architecture.
    """
    current_cpu = safe_env.get('processor_architecture', '').lower()

    def __init__(self, arch):
        self.arch = arch.lower().replace('x64', 'amd64')

    @property
    def target_cpu(self):
        return self.arch[self.arch.find('_') + 1:]

    def target_is_x86(self):
        return self.target_cpu == 'x86'

    def current_is_x86(self):
        return self.current_cpu == 'x86'

    def current_dir(self, hidex86=False, x64=False):
        """
        Current platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        subfolder: str
            '\target', or '' (see hidex86 parameter)
        """
        return (
            '' if (self.current_cpu == 'x86' and hidex86) else
            r'\x64' if (self.current_cpu == 'amd64' and x64) else
            r'\%s' % self.current_cpu
        )

    def target_dir(self, hidex86=False, x64=False):
        r"""
        Target platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        subfolder: str
            '\current', or '' (see hidex86 parameter)
        """
        return (
            '' if (self.target_cpu == 'x86' and hidex86) else
            r'\x64' if (self.target_cpu == 'amd64' and x64) else
            r'\%s' % self.target_cpu
        )

    def cross_dir(self, forcex86=False):
        r"""
        Cross platform specific subfolder.

        Parameters
        ----------
        forcex86: bool
            Use 'x86' as current architecture even if current acritecture is
            not x86.

        Return
        ------
        subfolder: str
            '' if target architecture is current architecture,
            '\current_target' if not.
        """
        current = 'x86' if forcex86 else self.current_cpu
        return (
            '' if self.target_cpu == current else
            self.target_dir().replace('\\', '\\%s_' % current)
        )


class RegistryInfo:
    """
    Microsoft Visual Studio related registry informations.

    Parameters
    ----------
    platform_info: PlatformInfo
        "PlatformInfo" instance.
    """
    HKEYS = (winreg.HKEY_USERS,
             winreg.HKEY_CURRENT_USER,
             winreg.HKEY_LOCAL_MACHINE,
             winreg.HKEY_CLASSES_ROOT)

    def __init__(self, platform_info):
        self.pi = platform_info

    @property
    def visualstudio(self):
        """
        Microsoft Visual Studio root registry key.
        """
        return 'VisualStudio'

    @property
    def sxs(self):
        """
        Microsoft Visual Studio SxS registry key.
        """
        return os.path.join(self.visualstudio, 'SxS')

    @property
    def vc(self):
        """
        Microsoft Visual C++ VC7 registry key.
        """
        return os.path.join(self.sxs, 'VC7')

    @property
    def vs(self):
        """
        Microsoft Visual Studio VS7 registry key.
        """
        return os.path.join(self.sxs, 'VS7')

    @property
    def vc_for_python(self):
        """
        Microsoft Visual C++ for Python registry key.
        """
        return r'DevDiv\VCForPython'

    @property
    def microsoft_sdk(self):
        """
        Microsoft SDK registry key.
        """
        return 'Microsoft SDKs'

    @property
    def windows_sdk(self):
        """
        Microsoft Windows/Platform SDK registry key.
        """
        return os.path.join(self.microsoft_sdk, 'Windows')

    @property
    def netfx_sdk(self):
        """
        Microsoft .NET Framework SDK registry key.
        """
        return os.path.join(self.microsoft_sdk, 'NETFXSDK')

    @property
    def windows_kits_roots(self):
        """
        Microsoft Windows Kits Roots registry key.
        """
        return r'Windows Kits\Installed Roots'

    def microsoft(self, key, x86=False):
        """
        Return key in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        x86: str
            Force x86 software registry.

        Return
        ------
        str: value
        """
        node64 = '' if self.pi.current_is_x86() or x86 else r'\Wow6432Node'
        return os.path.join('Software', node64, 'Microsoft', key)

    def lookup(self, key, name):
        """
        Look for values in registry in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        name: str
            Value name to find.

        Return
        ------
        str: value
        """
        KEY_READ = winreg.KEY_READ
        openkey = winreg.OpenKey
        ms = self.microsoft
        for hkey in self.HKEYS:
            try:
                bkey = openkey(hkey, ms(key), 0, KEY_READ)
            except (OSError, IOError):
                if not self.pi.current_is_x86():
                    try:
                        bkey = openkey(hkey, ms(key, True), 0, KEY_READ)
                    except (OSError, IOError):
                        continue
                else:
                    continue
            try:
                return winreg.QueryValueEx(bkey, name)[0]
            except (OSError, IOError):
                pass


class SystemInfo:
    """
    Microsoft Windows and Visual Studio related system inormations.

    Parameters
    ----------
    registry_info: RegistryInfo
        "RegistryInfo" instance.
    vc_ver: float
        Required Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparaison.
    WinDir = safe_env.get('WinDir', '')
    ProgramFiles = safe_env.get('ProgramFiles', '')
    ProgramFilesx86 = safe_env.get('ProgramFiles(x86)', ProgramFiles)

    def __init__(self, registry_info, vc_ver=None):
        self.ri = registry_info
        self.pi = self.ri.pi
        if vc_ver:
            self.vc_ver = vc_ver
        else:
            try:
                self.vc_ver = self.find_available_vc_vers()[-1]
            except IndexError:
                err = 'No Microsoft Visual C++ version found'
                raise distutils.errors.DistutilsPlatformError(err)

    def find_available_vc_vers(self):
        """
        Find all available Microsoft Visual C++ versions.
        """
        vckeys = (self.ri.vc, self.ri.vc_for_python)
        vc_vers = []
        for hkey in self.ri.HKEYS:
            for key in vckeys:
                try:
                    bkey = winreg.OpenKey(hkey, key, 0, winreg.KEY_READ)
                except (OSError, IOError):
                    continue
                subkeys, values, _ = winreg.QueryInfoKey(bkey)
                for i in range(values):
                    try:
                        ver = float(winreg.EnumValue(bkey, i)[0])
                        if ver not in vc_vers:
                            vc_vers.append(ver)
                    except ValueError:
                        pass
                for i in range(subkeys):
                    try:
                        ver = float(winreg.EnumKey(bkey, i))
                        if ver not in vc_vers:
                            vc_vers.append(ver)
                    except ValueError:
                        pass
        return sorted(vc_vers)

    @property
    def VSInstallDir(self):
        """
        Microsoft Visual Studio directory.
        """
        # Default path
        name = 'Microsoft Visual Studio %0.1f' % self.vc_ver
        default = os.path.join(self.ProgramFilesx86, name)

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vs, '%0.1f' % self.vc_ver) or default

    @property
    def VCInstallDir(self):
        """
        Microsoft Visual C++ directory.
        """
        # Default path
        default = r'Microsoft Visual Studio %0.1f\VC' % self.vc_ver
        guess_vc = os.path.join(self.ProgramFilesx86, default)

        # Try to get "VC++ for Python" path from registry as default path
        reg_path = os.path.join(self.ri.vc_for_python, '%0.1f' % self.vc_ver)
        python_vc = self.ri.lookup(reg_path, 'installdir')
        default_vc = os.path.join(python_vc, 'VC') if python_vc else guess_vc

        # Try to get path from registry, if fail use default path
        path = self.ri.lookup(self.ri.vc, '%0.1f' % self.vc_ver) or default_vc

        if not os.path.isdir(path):
            msg = 'Microsoft Visual C++ directory not found'
            raise distutils.errors.DistutilsPlatformError(msg)

        return path

    @property
    def WindowsSdkVersion(self):
        """
        Microsoft Windows SDK versions.
        """
        # Set Windows SDK versions for specified MSVC++ version
        if self.vc_ver <= 9.0:
            return ('7.0', '6.1', '6.0a')
        elif self.vc_ver == 10.0:
            return ('7.1', '7.0a')
        elif self.vc_ver == 11.0:
            return ('8.0', '8.0a')
        elif self.vc_ver == 12.0:
            return ('8.1', '8.1a')
        elif self.vc_ver >= 14.0:
            return ('10.0', '8.1')

    @property
    def WindowsSdkDir(self):
        """
        Microsoft Windows SDK directory.
        """
        sdkdir = ''
        for ver in self.WindowsSdkVersion:
            # Try to get it from registry
            loc = os.path.join(self.ri.windows_sdk, 'v%s' % ver)
            sdkdir = self.ri.lookup(loc, 'installationfolder')
            if sdkdir:
                break
        if not sdkdir or not os.path.isdir(sdkdir):
            # Try to get "VC++ for Python" version from registry
            path = os.path.join(self.ri.vc_for_python, '%0.1f' % self.vc_ver)
            install_base = self.ri.lookup(path, 'installdir')
            if install_base:
                sdkdir = os.path.join(install_base, 'WinSDK')
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default new path
            for ver in self.WindowsSdkVersion:
                intver = ver[:ver.rfind('.')]
                path = r'Microsoft SDKs\Windows Kits\%s' % (intver)
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default old path
            for ver in self.WindowsSdkVersion:
                path = r'Microsoft SDKs\Windows\v%s' % ver
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir:
            # If fail, use Platform SDK
            sdkdir = os.path.join(self.VCInstallDir, 'PlatformSDK')
        return sdkdir

    @property
    def WindowsSDKExecutablePath(self):
        """
        Microsoft Windows SDK executable directory.
        """
        # Find WinSDK NetFx Tools registry dir name
        if self.vc_ver <= 11.0:
            netfxver = 35
            arch = ''
        else:
            netfxver = 40
            hidex86 = True if self.vc_ver <= 12.0 else False
            arch = self.pi.current_dir(x64=True, hidex86=hidex86)
        fx = 'WinSDK-NetFx%dTools%s' % (netfxver, arch.replace('\\', '-'))

        # liste all possibles registry paths
        regpaths = []
        if self.vc_ver >= 14.0:
            for ver in self.NetFxSdkVersion:
                regpaths += [os.path.join(self.ri.netfx_sdk, ver, fx)]

        for ver in self.WindowsSdkVersion:
            regpaths += [os.path.join(self.ri.windows_sdk, 'v%sA' % ver, fx)]

        # Return installation folder from the more recent path
        for path in regpaths:
            execpath = self.ri.lookup(path, 'installationfolder')
            if execpath:
                break
        return execpath

    @property
    def FSharpInstallDir(self):
        """
        Microsoft Visual F# directory.
        """
        path = r'%0.1f\Setup\F#' % self.vc_ver
        path = os.path.join(self.ri.visualstudio, path)
        return self.ri.lookup(path, 'productdir') or ''

    @property
    def UniversalCRTSdkDir(self):
        """
        Microsoft Universal CRT SDK directory.
        """
        # Set Kit Roots versions for specified MSVC++ version
        if self.vc_ver >= 14.0:
            vers = ('10', '81')
        else:
            vers = ()

        # Find path of the more recent Kit
        for ver in vers:
            sdkdir = self.ri.lookup(self.ri.windows_kits_roots,
                                    'kitsroot%s' % ver)
            if sdkdir:
                break
        return sdkdir or ''

    @property
    def NetFxSdkVersion(self):
        """
        Microsoft .NET Framework SDK versions.
        """
        # Set FxSdk versions for specified MSVC++ version
        if self.vc_ver >= 14.0:
            return ('4.6.1', '4.6')
        else:
            return ()

    @property
    def NetFxSdkDir(self):
        """
        Microsoft .NET Framework SDK directory.
        """
        for ver in self.NetFxSdkVersion:
            loc = os.path.join(self.ri.netfx_sdk, ver)
            sdkdir = self.ri.lookup(loc, 'kitsinstallationfolder')
            if sdkdir:
                break
        return sdkdir or ''

    @property
    def FrameworkDir32(self):
        """
        Microsoft .NET Framework 32bit directory.
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir32') or guess_fw

    @property
    def FrameworkDir64(self):
        """
        Microsoft .NET Framework 64bit directory.
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework64')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir64') or guess_fw

    @property
    def FrameworkVersion32(self):
        """
        Microsoft .NET Framework 32bit versions.
        """
        return self._find_dot_net_versions(32)

    @property
    def FrameworkVersion64(self):
        """
        Microsoft .NET Framework 64bit versions.
        """
        return self._find_dot_net_versions(64)

    def _find_dot_net_versions(self, bits=32):
        """
        Find Microsoft .NET Framework versions.

        Parameters
        ----------
        bits: int
            Platform number of bits: 32 or 64.
        """
        # Find actual .NET version
        ver = self.ri.lookup(self.ri.vc, 'frameworkver%d' % bits) or ''

        # Set .NET versions for specified MSVC++ version
        if self.vc_ver >= 12.0:
            frameworkver = (ver, 'v4.0')
        elif self.vc_ver >= 10.0:
            frameworkver = ('v4.0.30319' if ver.lower()[:2] != 'v4' else ver,
                            'v3.5')
        elif self.vc_ver == 9.0:
            frameworkver = ('v3.5', 'v2.0.50727')
        if self.vc_ver == 8.0:
            frameworkver = ('v3.0', 'v2.0.50727')
        return frameworkver


class EnvironmentInfo:
    """
    Return environment variables for specified Microsoft Visual C++ version
    and platform : Lib, Include, Path and libpath.

    This function is compatible with Microsoft Visual C++ 9.0 to 14.0.

    Script created by analysing Microsoft environment configuration files like
    "vcvars[...].bat", "SetEnv.Cmd", "vcbuildtools.bat", ...

    Parameters
    ----------
    arch: str
        Target architecture.
    vc_ver: float
        Required Microsoft Visual C++ version. If not set, autodetect the last
        version.
    vc_min_ver: float
        Minimum Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparaison.

    def __init__(self, arch, vc_ver=None, vc_min_ver=None):
        self.pi = PlatformInfo(arch)
        self.ri = RegistryInfo(self.pi)
        self.si = SystemInfo(self.ri, vc_ver)

        if vc_min_ver:
            if self.vc_ver < vc_min_ver:
                err = 'No suitable Microsoft Visual C++ version found'
                raise distutils.errors.DistutilsPlatformError(err)

    @property
    def vc_ver(self):
        """
        Microsoft Visual C++ version.
        """
        return self.si.vc_ver

    @property
    def VSTools(self):
        """
        Microsoft Visual Studio Tools
        """
        paths = [r'Common7\IDE', r'Common7\Tools']

        if self.vc_ver >= 14.0:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            paths += [r'Common7\IDE\CommonExtensions\Microsoft\TestWindow']
            paths += [r'Team Tools\Performance Tools']
            paths += [r'Team Tools\Performance Tools%s' % arch_subdir]

        return [os.path.join(self.si.VSInstallDir, path) for path in paths]

    @property
    def VCIncludes(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Includes
        """
        return [os.path.join(self.si.VCInstallDir, 'Include'),
                os.path.join(self.si.VCInstallDir, r'ATLMFC\Include')]

    @property
    def VCLibraries(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Libraries
        """
        arch_subdir = self.pi.target_dir(hidex86=True)
        paths = ['Lib%s' % arch_subdir, r'ATLMFC\Lib%s' % arch_subdir]

        if self.vc_ver >= 14.0:
            paths += [r'Lib\store%s' % arch_subdir]

        return [os.path.join(self.si.VCInstallDir, path) for path in paths]

    @property
    def VCStoreRefs(self):
        """
        Microsoft Visual C++ store references Libraries
        """
        if self.vc_ver < 14.0:
            return []
        return [os.path.join(self.si.VCInstallDir, r'Lib\store\references')]

    @property
    def VCTools(self):
        """
        Microsoft Visual C++ Tools
        """
        si = self.si
        tools = [os.path.join(si.VCInstallDir, 'VCPackages')]

        forcex86 = True if self.vc_ver <= 10.0 else False
        arch_subdir = self.pi.cross_dir(forcex86)
        if arch_subdir:
            tools += [os.path.join(si.VCInstallDir, 'Bin%s' % arch_subdir)]

        if self.vc_ver >= 14.0:
            path = 'Bin%s' % self.pi.current_dir(hidex86=True)
            tools += [os.path.join(si.VCInstallDir, path)]

        else:
            tools += [os.path.join(si.VCInstallDir, 'Bin')]

        return tools

    @property
    def OSLibraries(self):
        """
        Microsoft Windows SDK Libraries
        """
        if self.vc_ver <= 10.0:
            arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
            return [os.path.join(self.si.WindowsSdkDir, 'Lib%s' % arch_subdir)]

        else:
            arch_subdir = self.pi.target_dir(x64=True)
            lib = os.path.join(self.si.WindowsSdkDir, 'lib')
            libver = self._get_content_dirname(lib)
            return [os.path.join(lib, '%sum%s' % (libver, arch_subdir))]

    @property
    def OSIncludes(self):
        """
        Microsoft Windows SDK Include
        """
        include = os.path.join(self.si.WindowsSdkDir, 'include')

        if self.vc_ver <= 10.0:
            return [include, os.path.join(include, 'gl')]

        else:
            if self.vc_ver >= 14.0:
                sdkver = self._get_content_dirname(include)
            else:
                sdkver = ''
            return [os.path.join(include, '%sshared' % sdkver),
                    os.path.join(include, '%sum' % sdkver),
                    os.path.join(include, '%swinrt' % sdkver)]

    @property
    def OSLibpath(self):
        """
        Microsoft Windows SDK Libraries Paths
        """
        ref = os.path.join(self.si.WindowsSdkDir, 'References')
        libpath = []

        if self.vc_ver <= 9.0:
            libpath += self.OSLibraries

        if self.vc_ver >= 11.0:
            libpath += [os.path.join(ref, r'CommonConfiguration\Neutral')]

        if self.vc_ver >= 14.0:
            libpath += [
                ref,
                os.path.join(self.si.WindowsSdkDir, 'UnionMetadata'),
                os.path.join(
                    ref,
                    'Windows.Foundation.UniversalApiContract',
                    '1.0.0.0',
                ),
                os.path.join(
                    ref,
                    'Windows.Foundation.FoundationContract',
                    '1.0.0.0',
                ),
                os.path.join(
                    ref,
                    'Windows.Networking.Connectivity.WwanContract',
                    '1.0.0.0',
                ),
                os.path.join(
                    self.si.WindowsSdkDir,
                    'ExtensionSDKs',
                    'Microsoft.VCLibs',
                    '%0.1f' % self.vc_ver,
                    'References',
                    'CommonConfiguration',
                    'neutral',
                ),
            ]
        return libpath

    @property
    def SdkTools(self):
        """
        Microsoft Windows SDK Tools
        """
        bin_dir = 'Bin' if self.vc_ver <= 11.0 else r'Bin\x86'
        tools = [os.path.join(self.si.WindowsSdkDir, bin_dir)]

        if not self.pi.current_is_x86():
            arch_subdir = self.pi.current_dir(x64=True)
            path = 'Bin%s' % arch_subdir
            tools += [os.path.join(self.si.WindowsSdkDir, path)]

        if self.vc_ver == 10.0 or self.vc_ver == 11.0:
            if self.pi.target_is_x86():
                arch_subdir = ''
            else:
                arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            path = r'Bin\NETFX 4.0 Tools%s' % arch_subdir
            tools += [os.path.join(self.si.WindowsSdkDir, path)]

        if self.si.WindowsSDKExecutablePath:
            tools += [self.si.WindowsSDKExecutablePath]

        return tools

    @property
    def SdkSetup(self):
        """
        Microsoft Windows SDK Setup
        """
        if self.vc_ver > 9.0:
            return []

        return [os.path.join(self.si.WindowsSdkDir, 'Setup')]

    @property
    def FxTools(self):
        """
        Microsoft .NET Framework Tools
        """
        pi = self.pi
        si = self.si

        if self.vc_ver <= 10.0:
            include32 = True
            include64 = not pi.target_is_x86() and not pi.current_is_x86()
        else:
            include32 = pi.target_is_x86() or pi.current_is_x86()
            include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'

        tools = []
        if include32:
            tools += [os.path.join(si.FrameworkDir32, ver)
                      for ver in si.FrameworkVersion32]
        if include64:
            tools += [os.path.join(si.FrameworkDir64, ver)
                      for ver in si.FrameworkVersion64]
        return tools

    @property
    def NetFxSDKLibraries(self):
        """
        Microsoft .Net Framework SDK Libraries
        """
        if self.vc_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        return [os.path.join(self.si.NetFxSdkDir, r'lib\um%s' % arch_subdir)]

    @property
    def NetFxSDKIncludes(self):
        """
        Microsoft .Net Framework SDK Includes
        """
        if self.vc_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        return [os.path.join(self.si.NetFxSdkDir, r'include\um')]

    @property
    def VsTDb(self):
        """
        Microsoft Visual Studio Team System Database
        """
        return [os.path.join(self.si.VSInstallDir, r'VSTSDB\Deploy')]

    @property
    def MSBuild(self):
        """
        Microsoft Build Engine
        """
        if self.vc_ver < 12.0:
            return []

        arch_subdir = self.pi.current_dir(hidex86=True)
        path = r'MSBuild\%0.1f\bin%s' % (self.vc_ver, arch_subdir)
        return [os.path.join(self.si.ProgramFilesx86, path)]

    @property
    def HTMLHelpWorkshop(self):
        """
        Microsoft HTML Help Workshop
        """
        if self.vc_ver < 11.0:
            return []

        return [os.path.join(self.si.ProgramFilesx86, 'HTML Help Workshop')]

    @property
    def UCRTLibraries(self):
        """
        Microsoft Universal CRT Libraries
        """
        if self.vc_ver < 14.0:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        lib = os.path.join(self.si.UniversalCRTSdkDir, 'lib')
        ucrtver = self._get_content_dirname(lib)
        return [os.path.join(lib, '%sucrt%s' % (ucrtver, arch_subdir))]

    @property
    def UCRTIncludes(self):
        """
        Microsoft Universal CRT Include
        """
        if self.vc_ver < 14.0:
            return []

        include = os.path.join(self.si.UniversalCRTSdkDir, 'include')
        ucrtver = self._get_content_dirname(include)
        return [os.path.join(include, '%sucrt' % ucrtver)]

    @property
    def FSharp(self):
        """
        Microsoft Visual F#
        """
        if self.vc_ver < 11.0 and self.vc_ver > 12.0:
            return []

        return self.si.FSharpInstallDir

    @property
    def VCRuntimeRedist(self):
        """
        Microsoft Visual C++ runtime redistribuable dll
        """
        arch_subdir = self.pi.target_dir(x64=True)
        vcruntime = 'redist%s\\Microsoft.VC%d0.CRT\\vcruntime%d0.dll'
        vcruntime = vcruntime % (arch_subdir, self.vc_ver, self.vc_ver)
        return os.path.join(self.si.VCInstallDir, vcruntime)

    def return_env(self, exists=True):
        """
        Return environment dict.

        Parameters
        ----------
        exists: bool
            It True, only return existing paths.
        """
        env = dict(
            include=self._build_paths('include',
                                      [self.VCIncludes,
                                       self.OSIncludes,
                                       self.UCRTIncludes,
                                       self.NetFxSDKIncludes],
                                      exists),
            lib=self._build_paths('lib',
                                  [self.VCLibraries,
                                   self.OSLibraries,
                                   self.FxTools,
                                   self.UCRTLibraries,
                                   self.NetFxSDKLibraries],
                                  exists),
            libpath=self._build_paths('libpath',
                                      [self.VCLibraries,
                                       self.FxTools,
                                       self.VCStoreRefs,
                                       self.OSLibpath],
                                      exists),
            path=self._build_paths('path',
                                   [self.VCTools,
                                    self.VSTools,
                                    self.VsTDb,
                                    self.SdkTools,
                                    self.SdkSetup,
                                    self.FxTools,
                                    self.MSBuild,
                                    self.HTMLHelpWorkshop,
                                    self.FSharp],
                                   exists),
        )
        if self.vc_ver >= 14 and os.path.isfile(self.VCRuntimeRedist):
            env['py_vcruntime_redist'] = self.VCRuntimeRedist
        return env

    def _build_paths(self, name, spec_path_lists, exists):
        """
        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.
        """
        # flatten spec_path_lists
        spec_paths = itertools.chain.from_iterable(spec_path_lists)
        env_paths = safe_env.get(name, '').split(os.pathsep)
        paths = itertools.chain(spec_paths, env_paths)
        extant_paths = list(filter(os.path.isdir, paths)) if exists else paths
        if not extant_paths:
            msg = "%s environment variable is empty" % name.upper()
            raise distutils.errors.DistutilsPlatformError(msg)
        unique_paths = self._unique_everseen(extant_paths)
        return os.pathsep.join(unique_paths)

    # from Python docs
    def _unique_everseen(self, iterable, key=None):
        """
        List unique elements, preserving order.
        Remember all elements ever seen.

        _unique_everseen('AAAABBBCCDAABBB') --> A B C D

        _unique_everseen('ABBCcAD', str.lower) --> A B C D
        """
        seen = set()
        seen_add = seen.add
        if key is None:
            for element in filterfalse(seen.__contains__, iterable):
                seen_add(element)
                yield element
        else:
            for element in iterable:
                k = key(element)
                if k not in seen:
                    seen_add(k)
                    yield element

    def _get_content_dirname(self, path):
        """
        Return name of the first dir in path or '' if no dir found.

        Parameters
        ----------
        path: str
            Path where search dir.

        Return
        ------
        foldername: str
            "name\" or ""
        """
        try:
            name = os.listdir(path)
            if name:
                return '%s\\' % name[0]
            return ''
        except (OSError, IOError):
            return ''
