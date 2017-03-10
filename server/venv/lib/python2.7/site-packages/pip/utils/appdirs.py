"""
This code was taken from https://github.com/ActiveState/appdirs and modified
to suit our purposes.
"""
from __future__ import absolute_import

import os
import sys

from pip.compat import WINDOWS, expanduser
from pip._vendor.six import PY2, text_type


def user_cache_dir(appname):
    r"""
    Return full path to the user-specific cache dir for this application.

        "appname" is the name of application.

    Typical user cache directories are:
        macOS:      ~/Library/Caches/<AppName>
        Unix:       ~/.cache/<AppName> (XDG default)
        Windows:    C:\Users\<username>\AppData\Local\<AppName>\Cache

    On Windows the only suggestion in the MSDN docs is that local settings go
    in the `CSIDL_LOCAL_APPDATA` directory. This is identical to the
    non-roaming app data dir (the default returned by `user_data_dir`). Apps
    typically put cache data somewhere *under* the given dir here. Some
    examples:
        ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
        ...\Acme\SuperApp\Cache\1.0

    OPINION: This function appends "Cache" to the `CSIDL_LOCAL_APPDATA` value.
    """
    if WINDOWS:
        # Get the base path
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))

        # When using Python 2, return paths as bytes on Windows like we do on
        # other operating systems. See helper function docs for more details.
        if PY2 and isinstance(path, text_type):
            path = _win_path_to_bytes(path)

        # Add our app name and Cache directory to it
        path = os.path.join(path, appname, "Cache")
    elif sys.platform == "darwin":
        # Get the base path
        path = expanduser("~/Library/Caches")

        # Add our app name to it
        path = os.path.join(path, appname)
    else:
        # Get the base path
        path = os.getenv("XDG_CACHE_HOME", expanduser("~/.cache"))

        # Add our app name to it
        path = os.path.join(path, appname)

    return path


def user_data_dir(appname, roaming=False):
    """
    Return full path to the user-specific data dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user data directories are:
        macOS:                  ~/Library/Application Support/<AppName>
        Unix:                   ~/.local/share/<AppName>    # or in
                                $XDG_DATA_HOME, if defined
        Win XP (not roaming):   C:\Documents and Settings\<username>\ ...
                                ...Application Data\<AppName>
        Win XP (roaming):       C:\Documents and Settings\<username>\Local ...
                                ...Settings\Application Data\<AppName>
        Win 7  (not roaming):   C:\\Users\<username>\AppData\Local\<AppName>
        Win 7  (roaming):       C:\\Users\<username>\AppData\Roaming\<AppName>

    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.
    That means, by default "~/.local/share/<AppName>".
    """
    if WINDOWS:
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        path = os.path.join(os.path.normpath(_get_win_folder(const)), appname)
    elif sys.platform == "darwin":
        path = os.path.join(
            expanduser('~/Library/Application Support/'),
            appname,
        )
    else:
        path = os.path.join(
            os.getenv('XDG_DATA_HOME', expanduser("~/.local/share")),
            appname,
        )

    return path


def user_config_dir(appname, roaming=True):
    """Return full path to the user-specific config dir for this application.

        "appname" is the name of application.
            If None, just the system directory is returned.
        "roaming" (boolean, default True) can be set False to not use the
            Windows roaming appdata directory. That means that for users on a
            Windows network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Typical user data directories are:
        macOS:                  same as user_data_dir
        Unix:                   ~/.config/<AppName>
        Win *:                  same as user_data_dir

    For Unix, we follow the XDG spec and support $XDG_CONFIG_HOME.
    That means, by default "~/.config/<AppName>".
    """
    if WINDOWS:
        path = user_data_dir(appname, roaming=roaming)
    elif sys.platform == "darwin":
        path = user_data_dir(appname)
    else:
        path = os.getenv('XDG_CONFIG_HOME', expanduser("~/.config"))
        path = os.path.join(path, appname)

    return path


# for the discussion regarding site_config_dirs locations
# see <https://github.com/pypa/pip/issues/1733>
def site_config_dirs(appname):
    """Return a list of potential user-shared config dirs for this application.

        "appname" is the name of application.

    Typical user config directories are:
        macOS:      /Library/Application Support/<AppName>/
        Unix:       /etc or $XDG_CONFIG_DIRS[i]/<AppName>/ for each value in
                    $XDG_CONFIG_DIRS
        Win XP:     C:\Documents and Settings\All Users\Application ...
                    ...Data\<AppName>\
        Vista:      (Fail! "C:\ProgramData" is a hidden *system* directory
                    on Vista.)
        Win 7:      Hidden, but writeable on Win 7:
                    C:\ProgramData\<AppName>\
    """
    if WINDOWS:
        path = os.path.normpath(_get_win_folder("CSIDL_COMMON_APPDATA"))
        pathlist = [os.path.join(path, appname)]
    elif sys.platform == 'darwin':
        pathlist = [os.path.join('/Library/Application Support', appname)]
    else:
        # try looking in $XDG_CONFIG_DIRS
        xdg_config_dirs = os.getenv('XDG_CONFIG_DIRS', '/etc/xdg')
        if xdg_config_dirs:
            pathlist = [
                os.path.join(expanduser(x), appname)
                for x in xdg_config_dirs.split(os.pathsep)
            ]
        else:
            pathlist = []

        # always look in /etc directly as well
        pathlist.append('/etc')

    return pathlist


# -- Windows support functions --

def _get_win_folder_from_registry(csidl_name):
    """
    This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
    import _winreg

    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }[csidl_name]

    key = _winreg.OpenKey(
        _winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
    )
    directory, _type = _winreg.QueryValueEx(key, shell_folder_name)
    return directory


def _get_win_folder_with_ctypes(csidl_name):
    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    return buf.value

if WINDOWS:
    try:
        import ctypes
        _get_win_folder = _get_win_folder_with_ctypes
    except ImportError:
        _get_win_folder = _get_win_folder_from_registry


def _win_path_to_bytes(path):
    """Encode Windows paths to bytes. Only used on Python 2.

    Motivation is to be consistent with other operating systems where paths
    are also returned as bytes. This avoids problems mixing bytes and Unicode
    elsewhere in the codebase. For more details and discussion see
    <https://github.com/pypa/pip/issues/3463>.

    If encoding using ASCII and MBCS fails, return the original Unicode path.
    """
    for encoding in ('ASCII', 'MBCS'):
        try:
            return path.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            pass
    return path
