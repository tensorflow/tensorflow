# Copyright 2015,2016 Nir Cohen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The ``distro`` package (``distro`` stands for Linux Distribution) provides
information about the Linux distribution it runs on, such as a reliable
machine-readable distro ID, or version information.

It is a renewed alternative implementation for Python's original
:py:func:`platform.linux_distribution` function, but it provides much more
functionality. An alternative implementation became necessary because Python
3.5 deprecated this function, and Python 3.7 is expected to remove it
altogether. Its predecessor function :py:func:`platform.dist` was already
deprecated since Python 2.6 and is also expected to be removed in Python 3.7.
Still, there are many cases in which access to Linux distribution information
is needed. See `Python issue 1322 <https://bugs.python.org/issue1322>`_ for
more information.
"""

import os
import re
import sys
import json
import shlex
import logging
import subprocess


if not sys.platform.startswith('linux'):
    raise ImportError('Unsupported platform: {0}'.format(sys.platform))

_UNIXCONFDIR = '/etc'
_OS_RELEASE_BASENAME = 'os-release'

#: Translation table for normalizing the "ID" attribute defined in os-release
#: files, for use by the :func:`distro.id` method.
#:
#: * Key: Value as defined in the os-release file, translated to lower case,
#:   with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_OS_ID = {}

#: Translation table for normalizing the "Distributor ID" attribute returned by
#: the lsb_release command, for use by the :func:`distro.id` method.
#:
#: * Key: Value as returned by the lsb_release command, translated to lower
#:   case, with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_LSB_ID = {
    'enterpriseenterprise': 'oracle',  # Oracle Enterprise Linux
    'redhatenterpriseworkstation': 'rhel',  # RHEL 6.7
}

#: Translation table for normalizing the distro ID derived from the file name
#: of distro release files, for use by the :func:`distro.id` method.
#:
#: * Key: Value as derived from the file name of a distro release file,
#:   translated to lower case, with blanks translated to underscores.
#:
#: * Value: Normalized value.
NORMALIZED_DISTRO_ID = {
    'redhat': 'rhel',  # RHEL 6.x, 7.x
}

# Pattern for content of distro release file (reversed)
_DISTRO_RELEASE_CONTENT_REVERSED_PATTERN = re.compile(
    r'(?:[^)]*\)(.*)\()? *(?:STL )?([\d.+\-a-z]*\d) *(?:esaeler *)?(.+)')

# Pattern for base file name of distro release file
_DISTRO_RELEASE_BASENAME_PATTERN = re.compile(
    r'(\w+)[-_](release|version)$')

# Base file names to be ignored when searching for distro release file
_DISTRO_RELEASE_IGNORE_BASENAMES = (
    'debian_version',
    'lsb-release',
    'oem-release',
    _OS_RELEASE_BASENAME,
    'system-release'
)


def linux_distribution(full_distribution_name=True):
    """
    Return information about the current Linux distribution as a tuple
    ``(id_name, version, codename)`` with items as follows:

    * ``id_name``:  If *full_distribution_name* is false, the result of
      :func:`distro.id`. Otherwise, the result of :func:`distro.name`.

    * ``version``:  The result of :func:`distro.version`.

    * ``codename``:  The result of :func:`distro.codename`.

    The interface of this function is compatible with the original
    :py:func:`platform.linux_distribution` function, supporting a subset of
    its parameters.

    The data it returns may not exactly be the same, because it uses more data
    sources than the original function, and that may lead to different data if
    the Linux distribution is not consistent across multiple data sources it
    provides (there are indeed such distributions ...).

    Another reason for differences is the fact that the :func:`distro.id`
    method normalizes the distro ID string to a reliable machine-readable value
    for a number of popular Linux distributions.
    """
    return _distro.linux_distribution(full_distribution_name)


def id():
    """
    Return the distro ID of the current Linux distribution, as a
    machine-readable string.

    For a number of Linux distributions, the returned distro ID value is
    *reliable*, in the sense that it is documented and that it does not change
    across releases of the distribution.

    This package maintains the following reliable distro ID values:

    ==============  =========================================
    Distro ID       Distribution
    ==============  =========================================
    "ubuntu"        Ubuntu
    "debian"        Debian
    "rhel"          RedHat Enterprise Linux
    "centos"        CentOS
    "fedora"        Fedora
    "sles"          SUSE Linux Enterprise Server
    "opensuse"      openSUSE
    "amazon"        Amazon Linux
    "arch"          Arch Linux
    "cloudlinux"    CloudLinux OS
    "exherbo"       Exherbo Linux
    "gentoo"        GenToo Linux
    "ibm_powerkvm"  IBM PowerKVM
    "kvmibm"        KVM for IBM z Systems
    "linuxmint"     Linux Mint
    "mageia"        Mageia
    "mandriva"      Mandriva Linux
    "parallels"     Parallels
    "pidora"        Pidora
    "raspbian"      Raspbian
    "oracle"        Oracle Linux (and Oracle Enterprise Linux)
    "scientific"    Scientific Linux
    "slackware"     Slackware
    "xenserver"     XenServer
    ==============  =========================================

    If you have a need to get distros for reliable IDs added into this set,
    or if you find that the :func:`distro.id` function returns a different
    distro ID for one of the listed distros, please create an issue in the
    `distro issue tracker`_.

    **Lookup hierarchy and transformations:**

    First, the ID is obtained from the following sources, in the specified
    order. The first available and non-empty value is used:

    * the value of the "ID" attribute of the os-release file,

    * the value of the "Distributor ID" attribute returned by the lsb_release
      command,

    * the first part of the file name of the distro release file,

    The so determined ID value then passes the following transformations,
    before it is returned by this method:

    * it is translated to lower case,

    * blanks (which should not be there anyway) are translated to underscores,

    * a normalization of the ID is performed, based upon
      `normalization tables`_. The purpose of this normalization is to ensure
      that the ID is as reliable as possible, even across incompatible changes
      in the Linux distributions. A common reason for an incompatible change is
      the addition of an os-release file, or the addition of the lsb_release
      command, with ID values that differ from what was previously determined
      from the distro release file name.
    """
    return _distro.id()


def name(pretty=False):
    """
    Return the name of the current Linux distribution, as a human-readable
    string.

    If *pretty* is false, the name is returned without version or codename.
    (e.g. "CentOS Linux")

    If *pretty* is true, the version and codename are appended.
    (e.g. "CentOS Linux 7.1.1503 (Core)")

    **Lookup hierarchy:**

    The name is obtained from the following sources, in the specified order.
    The first available and non-empty value is used:

    * If *pretty* is false:

      - the value of the "NAME" attribute of the os-release file,

      - the value of the "Distributor ID" attribute returned by the lsb_release
        command,

      - the value of the "<name>" field of the distro release file.

    * If *pretty* is true:

      - the value of the "PRETTY_NAME" attribute of the os-release file,

      - the value of the "Description" attribute returned by the lsb_release
        command,

      - the value of the "<name>" field of the distro release file, appended
        with the value of the pretty version ("<version_id>" and "<codename>"
        fields) of the distro release file, if available.
    """
    return _distro.name(pretty)


def version(pretty=False, best=False):
    """
    Return the version of the current Linux distribution, as a human-readable
    string.

    If *pretty* is false, the version is returned without codename (e.g.
    "7.0").

    If *pretty* is true, the codename in parenthesis is appended, if the
    codename is non-empty (e.g. "7.0 (Maipo)").

    Some distributions provide version numbers with different precisions in
    the different sources of distribution information. Examining the different
    sources in a fixed priority order does not always yield the most precise
    version (e.g. for Debian 8.2, or CentOS 7.1).

    The *best* parameter can be used to control the approach for the returned
    version:

    If *best* is false, the first non-empty version number in priority order of
    the examined sources is returned.

    If *best* is true, the most precise version number out of all examined
    sources is returned.

    **Lookup hierarchy:**

    In all cases, the version number is obtained from the following sources.
    If *best* is false, this order represents the priority order:

    * the value of the "VERSION_ID" attribute of the os-release file,
    * the value of the "Release" attribute returned by the lsb_release
      command,
    * the version number parsed from the "<version_id>" field of the first line
      of the distro release file,
    * the version number parsed from the "PRETTY_NAME" attribute of the
      os-release file, if it follows the format of the distro release files.
    * the version number parsed from the "Description" attribute returned by
      the lsb_release command, if it follows the format of the distro release
      files.
    """
    return _distro.version(pretty, best)


def version_parts(best=False):
    """
    Return the version of the current Linux distribution as a tuple
    ``(major, minor, build_number)`` with items as follows:

    * ``major``:  The result of :func:`distro.major_version`.

    * ``minor``:  The result of :func:`distro.minor_version`.

    * ``build_number``:  The result of :func:`distro.build_number`.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.version_parts(best)


def major_version(best=False):
    """
    Return the major version of the current Linux distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The major version is the first
    part of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.major_version(best)


def minor_version(best=False):
    """
    Return the minor version of the current Linux distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The minor version is the second
    part of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.minor_version(best)


def build_number(best=False):
    """
    Return the build number of the current Linux distribution, as a string,
    if provided.
    Otherwise, the empty string is returned. The build number is the third part
    of the dot-separated version string.

    For a description of the *best* parameter, see the :func:`distro.version`
    method.
    """
    return _distro.build_number(best)


def like():
    """
    Return a space-separated list of distro IDs of distributions that are
    closely related to the current Linux distribution in regards to packaging
    and programming interfaces, for example distributions the current
    distribution is a derivative from.

    **Lookup hierarchy:**

    This information item is only provided by the os-release file.
    For details, see the description of the "ID_LIKE" attribute in the
    `os-release man page
    <http://www.freedesktop.org/software/systemd/man/os-release.html>`_.
    """
    return _distro.like()


def codename():
    """
    Return the codename for the release of the current Linux distribution,
    as a string.

    If the distribution does not have a codename, an empty string is returned.

    Note that the returned codename is not always really a codename. For
    example, openSUSE returns "x86_64". This function does not handle such
    cases in any special way and just returns the string it finds, if any.

    **Lookup hierarchy:**

    * the codename within the "VERSION" attribute of the os-release file, if
      provided,

    * the value of the "Codename" attribute returned by the lsb_release
      command,

    * the value of the "<codename>" field of the distro release file.
    """
    return _distro.codename()


def info(pretty=False, best=False):
    """
    Return certain machine-readable information items about the current Linux
    distribution in a dictionary, as shown in the following example:

    .. sourcecode:: python

        {
            'id': 'rhel',
            'version': '7.0',
            'version_parts': {
                'major': '7',
                'minor': '0',
                'build_number': ''
            },
            'like': 'fedora',
            'codename': 'Maipo'
        }

    The dictionary structure and keys are always the same, regardless of which
    information items are available in the underlying data sources. The values
    for the various keys are as follows:

    * ``id``:  The result of :func:`distro.id`.

    * ``version``:  The result of :func:`distro.version`.

    * ``version_parts -> major``:  The result of :func:`distro.major_version`.

    * ``version_parts -> minor``:  The result of :func:`distro.minor_version`.

    * ``version_parts -> build_number``:  The result of
      :func:`distro.build_number`.

    * ``like``:  The result of :func:`distro.like`.

    * ``codename``:  The result of :func:`distro.codename`.

    For a description of the *pretty* and *best* parameters, see the
    :func:`distro.version` method.
    """
    return _distro.info(pretty, best)


def os_release_info():
    """
    Return a dictionary containing key-value pairs for the information items
    from the os-release file data source of the current Linux distribution.

    See `os-release file`_ for details about these information items.
    """
    return _distro.os_release_info()


def lsb_release_info():
    """
    Return a dictionary containing key-value pairs for the information items
    from the lsb_release command data source of the current Linux distribution.

    See `lsb_release command output`_ for details about these information
    items.
    """
    return _distro.lsb_release_info()


def distro_release_info():
    """
    Return a dictionary containing key-value pairs for the information items
    from the distro release file data source of the current Linux distribution.

    See `distro release file`_ for details about these information items.
    """
    return _distro.distro_release_info()


def os_release_attr(attribute):
    """
    Return a single named information item from the os-release file data source
    of the current Linux distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `os-release file`_ for details about these information items.
    """
    return _distro.os_release_attr(attribute)


def lsb_release_attr(attribute):
    """
    Return a single named information item from the lsb_release command output
    data source of the current Linux distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `lsb_release command output`_ for details about these information
    items.
    """
    return _distro.lsb_release_attr(attribute)


def distro_release_attr(attribute):
    """
    Return a single named information item from the distro release file
    data source of the current Linux distribution.

    Parameters:

    * ``attribute`` (string): Key of the information item.

    Returns:

    * (string): Value of the information item, if the item exists.
      The empty string, if the item does not exist.

    See `distro release file`_ for details about these information items.
    """
    return _distro.distro_release_attr(attribute)


class LinuxDistribution(object):
    """
    Provides information about a Linux distribution.

    This package creates a private module-global instance of this class with
    default initialization arguments, that is used by the
    `consolidated accessor functions`_ and `single source accessor functions`_.
    By using default initialization arguments, that module-global instance
    returns data about the current Linux distribution (i.e. the distro this
    package runs on).

    Normally, it is not necessary to create additional instances of this class.
    However, in situations where control is needed over the exact data sources
    that are used, instances of this class can be created with a specific
    distro release file, or a specific os-release file, or without invoking the
    lsb_release command.
    """

    def __init__(self,
                 include_lsb=True,
                 os_release_file='',
                 distro_release_file=''):
        """
        The initialization method of this class gathers information from the
        available data sources, and stores that in private instance attributes.
        Subsequent access to the information items uses these private instance
        attributes, so that the data sources are read only once.

        Parameters:

        * ``include_lsb`` (bool): Controls whether the
          `lsb_release command output`_ is included as a data source.

          If the lsb_release command is not available in the program execution
          path, the data source for the lsb_release command will be empty.

        * ``os_release_file`` (string): The path name of the
          `os-release file`_ that is to be used as a data source.

          An empty string (the default) will cause the default path name to
          be used (see `os-release file`_ for details).

          If the specified or defaulted os-release file does not exist, the
          data source for the os-release file will be empty.

        * ``distro_release_file`` (string): The path name of the
          `distro release file`_ that is to be used as a data source.

          An empty string (the default) will cause a default search algorithm
          to be used (see `distro release file`_ for details).

          If the specified distro release file does not exist, or if no default
          distro release file can be found, the data source for the distro
          release file will be empty.

        Public instance attributes:

        * ``os_release_file`` (string): The path name of the
          `os-release file`_ that is actually used as a data source. The
          empty string if no distro release file is used as a data source.

        * ``distro_release_file`` (string): The path name of the
          `distro release file`_ that is actually used as a data source. The
          empty string if no distro release file is used as a data source.

        Raises:

        * :py:exc:`IOError`: Some I/O issue with an os-release file or distro
          release file.

        * :py:exc:`subprocess.CalledProcessError`: The lsb_release command had
          some issue (other than not being available in the program execution
          path).

        * :py:exc:`UnicodeError`: A data source has unexpected characters or
          uses an unexpected encoding.
        """
        self.os_release_file = os_release_file or \
            os.path.join(_UNIXCONFDIR, _OS_RELEASE_BASENAME)
        self.distro_release_file = distro_release_file or ''  # updated later
        self._os_release_info = self._get_os_release_info()
        self._lsb_release_info = self._get_lsb_release_info() \
            if include_lsb else {}
        self._distro_release_info = self._get_distro_release_info()

    def __repr__(self):
        """Return repr of all info
        """
        return \
            "LinuxDistribution(" \
            "os_release_file={0!r}, " \
            "distro_release_file={1!r}, " \
            "_os_release_info={2!r}, " \
            "_lsb_release_info={3!r}, " \
            "_distro_release_info={4!r})".format(
                self.os_release_file,
                self.distro_release_file,
                self._os_release_info,
                self._lsb_release_info,
                self._distro_release_info)

    def linux_distribution(self, full_distribution_name=True):
        """
        Return information about the Linux distribution that is compatible
        with Python's :func:`platform.linux_distribution`, supporting a subset
        of its parameters.

        For details, see :func:`distro.linux_distribution`.
        """
        return (
            self.name() if full_distribution_name else self.id(),
            self.version(),
            self.codename()
        )

    def id(self):
        """Return the distro ID of the Linux distribution, as a string.

        For details, see :func:`distro.id`.
        """
        def normalize(distro_id, table):
            distro_id = distro_id.lower().replace(' ', '_')
            return table.get(distro_id, distro_id)

        distro_id = self.os_release_attr('id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_OS_ID)

        distro_id = self.lsb_release_attr('distributor_id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_LSB_ID)

        distro_id = self.distro_release_attr('id')
        if distro_id:
            return normalize(distro_id, NORMALIZED_DISTRO_ID)

        return ''

    def name(self, pretty=False):
        """
        Return the name of the Linux distribution, as a string.

        For details, see :func:`distro.name`.
        """
        name = self.os_release_attr('name') \
            or self.lsb_release_attr('distributor_id') \
            or self.distro_release_attr('name')
        if pretty:
            name = self.os_release_attr('pretty_name') \
                or self.lsb_release_attr('description')
            if not name:
                name = self.distro_release_attr('name')
                version = self.version(pretty=True)
                if version:
                    name = name + ' ' + version
        return name or ''

    def version(self, pretty=False, best=False):
        """
        Return the version of the Linux distribution, as a string.

        For details, see :func:`distro.version`.
        """
        versions = [
            self.os_release_attr('version_id'),
            self.lsb_release_attr('release'),
            self.distro_release_attr('version_id'),
            self._parse_distro_release_content(
                self.os_release_attr('pretty_name')).get('version_id', ''),
            self._parse_distro_release_content(
                self.lsb_release_attr('description')).get('version_id', '')
        ]
        version = ''
        if best:
            # This algorithm uses the last version in priority order that has
            # the best precision. If the versions are not in conflict, that
            # does not matter; otherwise, using the last one instead of the
            # first one might be considered a surprise.
            for v in versions:
                if v.count(".") > version.count(".") or version == '':
                    version = v
        else:
            for v in versions:
                if v != '':
                    version = v
                    break
        if pretty and version and self.codename():
            version = u'{0} ({1})'.format(version, self.codename())
        return version

    def version_parts(self, best=False):
        """
        Return the version of the Linux distribution, as a tuple of version
        numbers.

        For details, see :func:`distro.version_parts`.
        """
        version_str = self.version(best=best)
        if version_str:
            version_regex = re.compile(r'(\d+)\.?(\d+)?\.?(\d+)?')
            matches = version_regex.match(version_str)
            if matches:
                major, minor, build_number = matches.groups()
                return major, minor or '', build_number or ''
        return '', '', ''

    def major_version(self, best=False):
        """
        Return the major version number of the current distribution.

        For details, see :func:`distro.major_version`.
        """
        return self.version_parts(best)[0]

    def minor_version(self, best=False):
        """
        Return the minor version number of the Linux distribution.

        For details, see :func:`distro.minor_version`.
        """
        return self.version_parts(best)[1]

    def build_number(self, best=False):
        """
        Return the build number of the Linux distribution.

        For details, see :func:`distro.build_number`.
        """
        return self.version_parts(best)[2]

    def like(self):
        """
        Return the IDs of distributions that are like the Linux distribution.

        For details, see :func:`distro.like`.
        """
        return self.os_release_attr('id_like') or ''

    def codename(self):
        """
        Return the codename of the Linux distribution.

        For details, see :func:`distro.codename`.
        """
        return self.os_release_attr('codename') \
            or self.lsb_release_attr('codename') \
            or self.distro_release_attr('codename') \
            or ''

    def info(self, pretty=False, best=False):
        """
        Return certain machine-readable information about the Linux
        distribution.

        For details, see :func:`distro.info`.
        """
        return dict(
            id=self.id(),
            version=self.version(pretty, best),
            version_parts=dict(
                major=self.major_version(best),
                minor=self.minor_version(best),
                build_number=self.build_number(best)
            ),
            like=self.like(),
            codename=self.codename(),
        )

    def os_release_info(self):
        """
        Return a dictionary containing key-value pairs for the information
        items from the os-release file data source of the Linux distribution.

        For details, see :func:`distro.os_release_info`.
        """
        return self._os_release_info

    def lsb_release_info(self):
        """
        Return a dictionary containing key-value pairs for the information
        items from the lsb_release command data source of the Linux
        distribution.

        For details, see :func:`distro.lsb_release_info`.
        """
        return self._lsb_release_info

    def distro_release_info(self):
        """
        Return a dictionary containing key-value pairs for the information
        items from the distro release file data source of the Linux
        distribution.

        For details, see :func:`distro.distro_release_info`.
        """
        return self._distro_release_info

    def os_release_attr(self, attribute):
        """
        Return a single named information item from the os-release file data
        source of the Linux distribution.

        For details, see :func:`distro.os_release_attr`.
        """
        return self._os_release_info.get(attribute, '')

    def lsb_release_attr(self, attribute):
        """
        Return a single named information item from the lsb_release command
        output data source of the Linux distribution.

        For details, see :func:`distro.lsb_release_attr`.
        """
        return self._lsb_release_info.get(attribute, '')

    def distro_release_attr(self, attribute):
        """
        Return a single named information item from the distro release file
        data source of the Linux distribution.

        For details, see :func:`distro.distro_release_attr`.
        """
        return self._distro_release_info.get(attribute, '')

    def _get_os_release_info(self):
        """
        Get the information items from the specified os-release file.

        Returns:
            A dictionary containing all information items.
        """
        if os.path.isfile(self.os_release_file):
            with open(self.os_release_file) as release_file:
                return self._parse_os_release_content(release_file)
        return {}

    @staticmethod
    def _parse_os_release_content(lines):
        """
        Parse the lines of an os-release file.

        Parameters:

        * lines: Iterable through the lines in the os-release file.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
        props = {}
        lexer = shlex.shlex(lines, posix=True)
        lexer.whitespace_split = True

        # The shlex module defines its `wordchars` variable using literals,
        # making it dependent on the encoding of the Python source file.
        # In Python 2.6 and 2.7, the shlex source file is encoded in
        # 'iso-8859-1', and the `wordchars` variable is defined as a byte
        # string. This causes a UnicodeDecodeError to be raised when the
        # parsed content is a unicode object. The following fix resolves that
        # (... but it should be fixed in shlex...):
        if sys.version_info[0] == 2 and isinstance(lexer.wordchars, bytes):
            lexer.wordchars = lexer.wordchars.decode('iso-8859-1')

        tokens = list(lexer)
        for token in tokens:
            # At this point, all shell-like parsing has been done (i.e.
            # comments processed, quotes and backslash escape sequences
            # processed, multi-line values assembled, trailing newlines
            # stripped, etc.), so the tokens are now either:
            # * variable assignments: var=value
            # * commands or their arguments (not allowed in os-release)
            if '=' in token:
                k, v = token.split('=', 1)
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                props[k.lower()] = v
                if k == 'VERSION':
                    # this handles cases in which the codename is in
                    # the `(CODENAME)` (rhel, centos, fedora) format
                    # or in the `, CODENAME` format (Ubuntu).
                    codename = re.search(r'(\(\D+\))|,(\s+)?\D+', v)
                    if codename:
                        codename = codename.group()
                        codename = codename.strip('()')
                        codename = codename.strip(',')
                        codename = codename.strip()
                        # codename appears within paranthese.
                        props['codename'] = codename
                    else:
                        props['codename'] = ''
            else:
                # Ignore any tokens that are not variable assignments
                pass
        return props

    def _get_lsb_release_info(self):
        """
        Get the information items from the lsb_release command output.

        Returns:
            A dictionary containing all information items.
        """
        cmd = 'lsb_release -a'
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
        code = process.returncode
        if code == 0:
            content = stdout.splitlines()
            return self._parse_lsb_release_content(content)
        elif code == 127:  # Command not found
            return {}
        else:
            if sys.version_info[:2] >= (3, 5):
                raise subprocess.CalledProcessError(code, cmd, stdout, stderr)
            elif sys.version_info[:2] >= (2, 7):
                raise subprocess.CalledProcessError(code, cmd, stdout)
            elif sys.version_info[:2] == (2, 6):
                raise subprocess.CalledProcessError(code, cmd)

    @staticmethod
    def _parse_lsb_release_content(lines):
        """
        Parse the output of the lsb_release command.

        Parameters:

        * lines: Iterable through the lines of the lsb_release output.
                 Each line must be a unicode string or a UTF-8 encoded byte
                 string.

        Returns:
            A dictionary containing all information items.
        """
        props = {}
        for line in lines:
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            kv = line.strip('\n').split(':', 1)
            if len(kv) != 2:
                # Ignore lines without colon.
                continue
            k, v = kv
            props.update({k.replace(' ', '_').lower(): v.strip()})
        return props

    def _get_distro_release_info(self):
        """
        Get the information items from the specified distro release file.

        Returns:
            A dictionary containing all information items.
        """
        if self.distro_release_file:
            # If it was specified, we use it and parse what we can, even if
            # its file name or content does not match the expected pattern.
            distro_info = self._parse_distro_release_file(
                self.distro_release_file)
            basename = os.path.basename(self.distro_release_file)
            # The file name pattern for user-specified distro release files
            # is somewhat more tolerant (compared to when searching for the
            # file), because we want to use what was specified as best as
            # possible.
            match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
            if match:
                distro_info['id'] = match.group(1)
            return distro_info
        else:
            basenames = os.listdir(_UNIXCONFDIR)
            # We sort for repeatability in cases where there are multiple
            # distro specific files; e.g. CentOS, Oracle, Enterprise all
            # containing `redhat-release` on top of their own.
            basenames.sort()
            for basename in basenames:
                if basename in _DISTRO_RELEASE_IGNORE_BASENAMES:
                    continue
                match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
                if match:
                    filepath = os.path.join(_UNIXCONFDIR, basename)
                    distro_info = self._parse_distro_release_file(filepath)
                    if 'name' in distro_info:
                        # The name is always present if the pattern matches
                        self.distro_release_file = filepath
                        distro_info['id'] = match.group(1)
                        return distro_info
            return {}

    def _parse_distro_release_file(self, filepath):
        """
        Parse a distro release file.

        Parameters:

        * filepath: Path name of the distro release file.

        Returns:
            A dictionary containing all information items.
        """
        if os.path.isfile(filepath):
            with open(filepath) as fp:
                # Only parse the first line. For instance, on SLES there
                # are multiple lines. We don't want them...
                return self._parse_distro_release_content(fp.readline())
        return {}

    @staticmethod
    def _parse_distro_release_content(line):
        """
        Parse a line from a distro release file.

        Parameters:
        * line: Line from the distro release file. Must be a unicode string
                or a UTF-8 encoded byte string.

        Returns:
            A dictionary containing all information items.
        """
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        matches = _DISTRO_RELEASE_CONTENT_REVERSED_PATTERN.match(
            line.strip()[::-1])
        distro_info = {}
        if matches:
            # regexp ensures non-None
            distro_info['name'] = matches.group(3)[::-1]
            if matches.group(2):
                distro_info['version_id'] = matches.group(2)[::-1]
            if matches.group(1):
                distro_info['codename'] = matches.group(1)[::-1]
        elif line:
            distro_info['name'] = line.strip()
        return distro_info


_distro = LinuxDistribution()


def main():
    import argparse

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser(description="Linux distro info tool")
    parser.add_argument(
        '--json',
        '-j',
        help="Output in machine readable format",
        action="store_true")
    args = parser.parse_args()

    if args.json:
        logger.info(json.dumps(info(), indent=4, sort_keys=True))
    else:
        logger.info('Name: %s', name(pretty=True))
        distribution_version = version(pretty=True)
        if distribution_version:
            logger.info('Version: %s', distribution_version)
        distribution_codename = codename()
        if distribution_codename:
            logger.info('Codename: %s', distribution_codename)


if __name__ == '__main__':
    main()
