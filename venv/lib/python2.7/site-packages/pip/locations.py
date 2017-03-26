"""Locations where we look for configs, install stuff, etc"""
from __future__ import absolute_import

import os
import os.path
import site
import sys

from distutils import sysconfig
from distutils.command.install import install, SCHEME_KEYS  # noqa

from pip.compat import WINDOWS, expanduser
from pip.utils import appdirs


# Application Directories
USER_CACHE_DIR = appdirs.user_cache_dir("pip")


DELETE_MARKER_MESSAGE = '''\
This file is placed here by pip to indicate the source was put
here by pip.

Once this package is successfully installed this source code will be
deleted (unless you remove this file).
'''
PIP_DELETE_MARKER_FILENAME = 'pip-delete-this-directory.txt'


def write_delete_marker_file(directory):
    """
    Write the pip delete marker file into this directory.
    """
    filepath = os.path.join(directory, PIP_DELETE_MARKER_FILENAME)
    with open(filepath, 'w') as marker_fp:
        marker_fp.write(DELETE_MARKER_MESSAGE)


def running_under_virtualenv():
    """
    Return True if we're running inside a virtualenv, False otherwise.

    """
    if hasattr(sys, 'real_prefix'):
        return True
    elif sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True

    return False


def virtualenv_no_global():
    """
    Return True if in a venv and no system site packages.
    """
    # this mirrors the logic in virtualenv.py for locating the
    # no-global-site-packages.txt file
    site_mod_dir = os.path.dirname(os.path.abspath(site.__file__))
    no_global_file = os.path.join(site_mod_dir, 'no-global-site-packages.txt')
    if running_under_virtualenv() and os.path.isfile(no_global_file):
        return True


if running_under_virtualenv():
    src_prefix = os.path.join(sys.prefix, 'src')
else:
    # FIXME: keep src in cwd for now (it is not a temporary folder)
    try:
        src_prefix = os.path.join(os.getcwd(), 'src')
    except OSError:
        # In case the current working directory has been renamed or deleted
        sys.exit(
            "The folder you are executing pip from can no longer be found."
        )

# under macOS + virtualenv sys.prefix is not properly resolved
# it is something like /path/to/python/bin/..
# Note: using realpath due to tmp dirs on OSX being symlinks
src_prefix = os.path.abspath(src_prefix)

# FIXME doesn't account for venv linked to global site-packages

site_packages = sysconfig.get_python_lib()
user_site = site.USER_SITE
user_dir = expanduser('~')
if WINDOWS:
    bin_py = os.path.join(sys.prefix, 'Scripts')
    bin_user = os.path.join(user_site, 'Scripts')
    # buildout uses 'bin' on Windows too?
    if not os.path.exists(bin_py):
        bin_py = os.path.join(sys.prefix, 'bin')
        bin_user = os.path.join(user_site, 'bin')

    config_basename = 'pip.ini'

    legacy_storage_dir = os.path.join(user_dir, 'pip')
    legacy_config_file = os.path.join(
        legacy_storage_dir,
        config_basename,
    )
else:
    bin_py = os.path.join(sys.prefix, 'bin')
    bin_user = os.path.join(user_site, 'bin')

    config_basename = 'pip.conf'

    legacy_storage_dir = os.path.join(user_dir, '.pip')
    legacy_config_file = os.path.join(
        legacy_storage_dir,
        config_basename,
    )

    # Forcing to use /usr/local/bin for standard macOS framework installs
    # Also log to ~/Library/Logs/ for use with the Console.app log viewer
    if sys.platform[:6] == 'darwin' and sys.prefix[:16] == '/System/Library/':
        bin_py = '/usr/local/bin'

site_config_files = [
    os.path.join(path, config_basename)
    for path in appdirs.site_config_dirs('pip')
]


def distutils_scheme(dist_name, user=False, home=None, root=None,
                     isolated=False, prefix=None):
    """
    Return a distutils install scheme
    """
    from distutils.dist import Distribution

    scheme = {}

    if isolated:
        extra_dist_args = {"script_args": ["--no-user-cfg"]}
    else:
        extra_dist_args = {}
    dist_args = {'name': dist_name}
    dist_args.update(extra_dist_args)

    d = Distribution(dist_args)
    d.parse_config_files()
    i = d.get_command_obj('install', create=True)
    # NOTE: setting user or home has the side-effect of creating the home dir
    # or user base for installations during finalize_options()
    # ideally, we'd prefer a scheme class that has no side-effects.
    assert not (user and prefix), "user={0} prefix={1}".format(user, prefix)
    i.user = user or i.user
    if user:
        i.prefix = ""
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, 'install_' + key)

    # install_lib specified in setup.cfg should install *everything*
    # into there (i.e. it takes precedence over both purelib and
    # platlib).  Note, i.install_lib is *always* set after
    # finalize_options(); we only want to override here if the user
    # has explicitly requested it hence going back to the config
    if 'install_lib' in d.get_option_dict('install'):
        scheme.update(dict(purelib=i.install_lib, platlib=i.install_lib))

    if running_under_virtualenv():
        scheme['headers'] = os.path.join(
            sys.prefix,
            'include',
            'site',
            'python' + sys.version[:3],
            dist_name,
        )

        if root is not None:
            path_no_drive = os.path.splitdrive(
                os.path.abspath(scheme["headers"]))[1]
            scheme["headers"] = os.path.join(
                root,
                path_no_drive[1:],
            )

    return scheme
