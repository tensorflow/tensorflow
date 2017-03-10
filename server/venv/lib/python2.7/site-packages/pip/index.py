"""Routines related to PyPI, indexes"""
from __future__ import absolute_import

import logging
import cgi
from collections import namedtuple
import itertools
import sys
import os
import re
import mimetypes
import posixpath
import warnings

from pip._vendor.six.moves.urllib import parse as urllib_parse
from pip._vendor.six.moves.urllib import request as urllib_request

from pip.compat import ipaddress
from pip.utils import (
    cached_property, splitext, normalize_path,
    ARCHIVE_EXTENSIONS, SUPPORTED_EXTENSIONS,
)
from pip.utils.deprecation import RemovedInPip10Warning
from pip.utils.logging import indent_log
from pip.utils.packaging import check_requires_python
from pip.exceptions import (
    DistributionNotFound, BestVersionAlreadyInstalled, InvalidWheelFilename,
    UnsupportedWheel,
)
from pip.download import HAS_TLS, is_url, path_to_url, url_to_path
from pip.wheel import Wheel, wheel_ext
from pip.pep425tags import get_supported
from pip._vendor import html5lib, requests, six
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging import specifiers
from pip._vendor.requests.exceptions import SSLError
from pip._vendor.distlib.compat import unescape


__all__ = ['FormatControl', 'fmt_ctl_handle_mutual_exclude', 'PackageFinder']


SECURE_ORIGINS = [
    # protocol, hostname, port
    # Taken from Chrome's list of secure origins (See: http://bit.ly/1qrySKC)
    ("https", "*", "*"),
    ("*", "localhost", "*"),
    ("*", "127.0.0.0/8", "*"),
    ("*", "::1/128", "*"),
    ("file", "*", None),
    # ssh is always secure.
    ("ssh", "*", "*"),
]


logger = logging.getLogger(__name__)


class InstallationCandidate(object):

    def __init__(self, project, version, location):
        self.project = project
        self.version = parse_version(version)
        self.location = location
        self._key = (self.project, self.version, self.location)

    def __repr__(self):
        return "<InstallationCandidate({0!r}, {1!r}, {2!r})>".format(
            self.project, self.version, self.location,
        )

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other, method):
        if not isinstance(other, InstallationCandidate):
            return NotImplemented

        return method(self._key, other._key)


class PackageFinder(object):
    """This finds packages.

    This is meant to match easy_install's technique for looking for
    packages, by reading pages and looking for appropriate links.
    """

    def __init__(self, find_links, index_urls, allow_all_prereleases=False,
                 trusted_hosts=None, process_dependency_links=False,
                 session=None, format_control=None, platform=None,
                 versions=None, abi=None, implementation=None):
        """Create a PackageFinder.

        :param format_control: A FormatControl object or None. Used to control
            the selection of source packages / binary packages when consulting
            the index and links.
        :param platform: A string or None. If None, searches for packages
            that are supported by the current system. Otherwise, will find
            packages that can be built on the platform passed in. These
            packages will only be downloaded for distribution: they will
            not be built locally.
        :param versions: A list of strings or None. This is passed directly
            to pep425tags.py in the get_supported() method.
        :param abi: A string or None. This is passed directly
            to pep425tags.py in the get_supported() method.
        :param implementation: A string or None. This is passed directly
            to pep425tags.py in the get_supported() method.
        """
        if session is None:
            raise TypeError(
                "PackageFinder() missing 1 required keyword argument: "
                "'session'"
            )

        # Build find_links. If an argument starts with ~, it may be
        # a local file relative to a home directory. So try normalizing
        # it and if it exists, use the normalized version.
        # This is deliberately conservative - it might be fine just to
        # blindly normalize anything starting with a ~...
        self.find_links = []
        for link in find_links:
            if link.startswith('~'):
                new_link = normalize_path(link)
                if os.path.exists(new_link):
                    link = new_link
            self.find_links.append(link)

        self.index_urls = index_urls
        self.dependency_links = []

        # These are boring links that have already been logged somehow:
        self.logged_links = set()

        self.format_control = format_control or FormatControl(set(), set())

        # Domains that we won't emit warnings for when not using HTTPS
        self.secure_origins = [
            ("*", host, "*")
            for host in (trusted_hosts if trusted_hosts else [])
        ]

        # Do we want to allow _all_ pre-releases?
        self.allow_all_prereleases = allow_all_prereleases

        # Do we process dependency links?
        self.process_dependency_links = process_dependency_links

        # The Session we'll use to make requests
        self.session = session

        # The valid tags to check potential found wheel candidates against
        self.valid_tags = get_supported(
            versions=versions,
            platform=platform,
            abi=abi,
            impl=implementation,
        )

        # If we don't have TLS enabled, then WARN if anyplace we're looking
        # relies on TLS.
        if not HAS_TLS:
            for link in itertools.chain(self.index_urls, self.find_links):
                parsed = urllib_parse.urlparse(link)
                if parsed.scheme == "https":
                    logger.warning(
                        "pip is configured with locations that require "
                        "TLS/SSL, however the ssl module in Python is not "
                        "available."
                    )
                    break

    def add_dependency_links(self, links):
        # # FIXME: this shouldn't be global list this, it should only
        # # apply to requirements of the package that specifies the
        # # dependency_links value
        # # FIXME: also, we should track comes_from (i.e., use Link)
        if self.process_dependency_links:
            warnings.warn(
                "Dependency Links processing has been deprecated and will be "
                "removed in a future release.",
                RemovedInPip10Warning,
            )
            self.dependency_links.extend(links)

    @staticmethod
    def _sort_locations(locations, expand_dir=False):
        """
        Sort locations into "files" (archives) and "urls", and return
        a pair of lists (files,urls)
        """
        files = []
        urls = []

        # puts the url for the given file path into the appropriate list
        def sort_path(path):
            url = path_to_url(path)
            if mimetypes.guess_type(url, strict=False)[0] == 'text/html':
                urls.append(url)
            else:
                files.append(url)

        for url in locations:

            is_local_path = os.path.exists(url)
            is_file_url = url.startswith('file:')

            if is_local_path or is_file_url:
                if is_local_path:
                    path = url
                else:
                    path = url_to_path(url)
                if os.path.isdir(path):
                    if expand_dir:
                        path = os.path.realpath(path)
                        for item in os.listdir(path):
                            sort_path(os.path.join(path, item))
                    elif is_file_url:
                        urls.append(url)
                elif os.path.isfile(path):
                    sort_path(path)
                else:
                    logger.warning(
                        "Url '%s' is ignored: it is neither a file "
                        "nor a directory.", url)
            elif is_url(url):
                # Only add url with clear scheme
                urls.append(url)
            else:
                logger.warning(
                    "Url '%s' is ignored. It is either a non-existing "
                    "path or lacks a specific scheme.", url)

        return files, urls

    def _candidate_sort_key(self, candidate):
        """
        Function used to generate link sort key for link tuples.
        The greater the return value, the more preferred it is.
        If not finding wheels, then sorted by version only.
        If finding wheels, then the sort order is by version, then:
          1. existing installs
          2. wheels ordered via Wheel.support_index_min(self.valid_tags)
          3. source archives
        Note: it was considered to embed this logic into the Link
              comparison operators, but then different sdist links
              with the same version, would have to be considered equal
        """
        support_num = len(self.valid_tags)
        if candidate.location.is_wheel:
            # can raise InvalidWheelFilename
            wheel = Wheel(candidate.location.filename)
            if not wheel.supported(self.valid_tags):
                raise UnsupportedWheel(
                    "%s is not a supported wheel for this platform. It "
                    "can't be sorted." % wheel.filename
                )
            pri = -(wheel.support_index_min(self.valid_tags))
        else:  # sdist
            pri = -(support_num)
        return (candidate.version, pri)

    def _validate_secure_origin(self, logger, location):
        # Determine if this url used a secure transport mechanism
        parsed = urllib_parse.urlparse(str(location))
        origin = (parsed.scheme, parsed.hostname, parsed.port)

        # The protocol to use to see if the protocol matches.
        # Don't count the repository type as part of the protocol: in
        # cases such as "git+ssh", only use "ssh". (I.e., Only verify against
        # the last scheme.)
        protocol = origin[0].rsplit('+', 1)[-1]

        # Determine if our origin is a secure origin by looking through our
        # hardcoded list of secure origins, as well as any additional ones
        # configured on this PackageFinder instance.
        for secure_origin in (SECURE_ORIGINS + self.secure_origins):
            if protocol != secure_origin[0] and secure_origin[0] != "*":
                continue

            try:
                # We need to do this decode dance to ensure that we have a
                # unicode object, even on Python 2.x.
                addr = ipaddress.ip_address(
                    origin[1]
                    if (
                        isinstance(origin[1], six.text_type) or
                        origin[1] is None
                    )
                    else origin[1].decode("utf8")
                )
                network = ipaddress.ip_network(
                    secure_origin[1]
                    if isinstance(secure_origin[1], six.text_type)
                    else secure_origin[1].decode("utf8")
                )
            except ValueError:
                # We don't have both a valid address or a valid network, so
                # we'll check this origin against hostnames.
                if (origin[1] and
                        origin[1].lower() != secure_origin[1].lower() and
                        secure_origin[1] != "*"):
                    continue
            else:
                # We have a valid address and network, so see if the address
                # is contained within the network.
                if addr not in network:
                    continue

            # Check to see if the port patches
            if (origin[2] != secure_origin[2] and
                    secure_origin[2] != "*" and
                    secure_origin[2] is not None):
                continue

            # If we've gotten here, then this origin matches the current
            # secure origin and we should return True
            return True

        # If we've gotten to this point, then the origin isn't secure and we
        # will not accept it as a valid location to search. We will however
        # log a warning that we are ignoring it.
        logger.warning(
            "The repository located at %s is not a trusted or secure host and "
            "is being ignored. If this repository is available via HTTPS it "
            "is recommended to use HTTPS instead, otherwise you may silence "
            "this warning and allow it anyways with '--trusted-host %s'.",
            parsed.hostname,
            parsed.hostname,
        )

        return False

    def _get_index_urls_locations(self, project_name):
        """Returns the locations found via self.index_urls

        Checks the url_name on the main (first in the list) index and
        use this url_name to produce all locations
        """

        def mkurl_pypi_url(url):
            loc = posixpath.join(
                url,
                urllib_parse.quote(canonicalize_name(project_name)))
            # For maximum compatibility with easy_install, ensure the path
            # ends in a trailing slash.  Although this isn't in the spec
            # (and PyPI can handle it without the slash) some other index
            # implementations might break if they relied on easy_install's
            # behavior.
            if not loc.endswith('/'):
                loc = loc + '/'
            return loc

        return [mkurl_pypi_url(url) for url in self.index_urls]

    def find_all_candidates(self, project_name):
        """Find all available InstallationCandidate for project_name

        This checks index_urls, find_links and dependency_links.
        All versions found are returned as an InstallationCandidate list.

        See _link_package_versions for details on which files are accepted
        """
        index_locations = self._get_index_urls_locations(project_name)
        index_file_loc, index_url_loc = self._sort_locations(index_locations)
        fl_file_loc, fl_url_loc = self._sort_locations(
            self.find_links, expand_dir=True)
        dep_file_loc, dep_url_loc = self._sort_locations(self.dependency_links)

        file_locations = (
            Link(url) for url in itertools.chain(
                index_file_loc, fl_file_loc, dep_file_loc)
        )

        # We trust every url that the user has given us whether it was given
        #   via --index-url or --find-links
        # We explicitly do not trust links that came from dependency_links
        # We want to filter out any thing which does not have a secure origin.
        url_locations = [
            link for link in itertools.chain(
                (Link(url) for url in index_url_loc),
                (Link(url) for url in fl_url_loc),
                (Link(url) for url in dep_url_loc),
            )
            if self._validate_secure_origin(logger, link)
        ]

        logger.debug('%d location(s) to search for versions of %s:',
                     len(url_locations), project_name)

        for location in url_locations:
            logger.debug('* %s', location)

        canonical_name = canonicalize_name(project_name)
        formats = fmt_ctl_formats(self.format_control, canonical_name)
        search = Search(project_name, canonical_name, formats)
        find_links_versions = self._package_versions(
            # We trust every directly linked archive in find_links
            (Link(url, '-f') for url in self.find_links),
            search
        )

        page_versions = []
        for page in self._get_pages(url_locations, project_name):
            logger.debug('Analyzing links from page %s', page.url)
            with indent_log():
                page_versions.extend(
                    self._package_versions(page.links, search)
                )

        dependency_versions = self._package_versions(
            (Link(url) for url in self.dependency_links), search
        )
        if dependency_versions:
            logger.debug(
                'dependency_links found: %s',
                ', '.join([
                    version.location.url for version in dependency_versions
                ])
            )

        file_versions = self._package_versions(file_locations, search)
        if file_versions:
            file_versions.sort(reverse=True)
            logger.debug(
                'Local files found: %s',
                ', '.join([
                    url_to_path(candidate.location.url)
                    for candidate in file_versions
                ])
            )

        # This is an intentional priority ordering
        return (
            file_versions + find_links_versions + page_versions +
            dependency_versions
        )

    def find_requirement(self, req, upgrade):
        """Try to find a Link matching req

        Expects req, an InstallRequirement and upgrade, a boolean
        Returns a Link if found,
        Raises DistributionNotFound or BestVersionAlreadyInstalled otherwise
        """
        all_candidates = self.find_all_candidates(req.name)

        # Filter out anything which doesn't match our specifier
        compatible_versions = set(
            req.specifier.filter(
                # We turn the version object into a str here because otherwise
                # when we're debundled but setuptools isn't, Python will see
                # packaging.version.Version and
                # pkg_resources._vendor.packaging.version.Version as different
                # types. This way we'll use a str as a common data interchange
                # format. If we stop using the pkg_resources provided specifier
                # and start using our own, we can drop the cast to str().
                [str(c.version) for c in all_candidates],
                prereleases=(
                    self.allow_all_prereleases
                    if self.allow_all_prereleases else None
                ),
            )
        )
        applicable_candidates = [
            # Again, converting to str to deal with debundling.
            c for c in all_candidates if str(c.version) in compatible_versions
        ]

        if applicable_candidates:
            best_candidate = max(applicable_candidates,
                                 key=self._candidate_sort_key)
        else:
            best_candidate = None

        if req.satisfied_by is not None:
            installed_version = parse_version(req.satisfied_by.version)
        else:
            installed_version = None

        if installed_version is None and best_candidate is None:
            logger.critical(
                'Could not find a version that satisfies the requirement %s '
                '(from versions: %s)',
                req,
                ', '.join(
                    sorted(
                        set(str(c.version) for c in all_candidates),
                        key=parse_version,
                    )
                )
            )

            raise DistributionNotFound(
                'No matching distribution found for %s' % req
            )

        best_installed = False
        if installed_version and (
                best_candidate is None or
                best_candidate.version <= installed_version):
            best_installed = True

        if not upgrade and installed_version is not None:
            if best_installed:
                logger.debug(
                    'Existing installed version (%s) is most up-to-date and '
                    'satisfies requirement',
                    installed_version,
                )
            else:
                logger.debug(
                    'Existing installed version (%s) satisfies requirement '
                    '(most up-to-date version is %s)',
                    installed_version,
                    best_candidate.version,
                )
            return None

        if best_installed:
            # We have an existing version, and its the best version
            logger.debug(
                'Installed version (%s) is most up-to-date (past versions: '
                '%s)',
                installed_version,
                ', '.join(sorted(compatible_versions, key=parse_version)) or
                "none",
            )
            raise BestVersionAlreadyInstalled

        logger.debug(
            'Using version %s (newest of versions: %s)',
            best_candidate.version,
            ', '.join(sorted(compatible_versions, key=parse_version))
        )
        return best_candidate.location

    def _get_pages(self, locations, project_name):
        """
        Yields (page, page_url) from the given locations, skipping
        locations that have errors.
        """
        seen = set()
        for location in locations:
            if location in seen:
                continue
            seen.add(location)

            page = self._get_page(location)
            if page is None:
                continue

            yield page

    _py_version_re = re.compile(r'-py([123]\.?[0-9]?)$')

    def _sort_links(self, links):
        """
        Returns elements of links in order, non-egg links first, egg links
        second, while eliminating duplicates
        """
        eggs, no_eggs = [], []
        seen = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                if link.egg_fragment:
                    eggs.append(link)
                else:
                    no_eggs.append(link)
        return no_eggs + eggs

    def _package_versions(self, links, search):
        result = []
        for link in self._sort_links(links):
            v = self._link_package_versions(link, search)
            if v is not None:
                result.append(v)
        return result

    def _log_skipped_link(self, link, reason):
        if link not in self.logged_links:
            logger.debug('Skipping link %s; %s', link, reason)
            self.logged_links.add(link)

    def _link_package_versions(self, link, search):
        """Return an InstallationCandidate or None"""
        version = None
        if link.egg_fragment:
            egg_info = link.egg_fragment
            ext = link.ext
        else:
            egg_info, ext = link.splitext()
            if not ext:
                self._log_skipped_link(link, 'not a file')
                return
            if ext not in SUPPORTED_EXTENSIONS:
                self._log_skipped_link(
                    link, 'unsupported archive format: %s' % ext)
                return
            if "binary" not in search.formats and ext == wheel_ext:
                self._log_skipped_link(
                    link, 'No binaries permitted for %s' % search.supplied)
                return
            if "macosx10" in link.path and ext == '.zip':
                self._log_skipped_link(link, 'macosx10 one')
                return
            if ext == wheel_ext:
                try:
                    wheel = Wheel(link.filename)
                except InvalidWheelFilename:
                    self._log_skipped_link(link, 'invalid wheel filename')
                    return
                if canonicalize_name(wheel.name) != search.canonical:
                    self._log_skipped_link(
                        link, 'wrong project name (not %s)' % search.supplied)
                    return

                if not wheel.supported(self.valid_tags):
                    self._log_skipped_link(
                        link, 'it is not compatible with this Python')
                    return

                version = wheel.version

        # This should be up by the search.ok_binary check, but see issue 2700.
        if "source" not in search.formats and ext != wheel_ext:
            self._log_skipped_link(
                link, 'No sources permitted for %s' % search.supplied)
            return

        if not version:
            version = egg_info_matches(egg_info, search.supplied, link)
        if version is None:
            self._log_skipped_link(
                link, 'wrong project name (not %s)' % search.supplied)
            return

        match = self._py_version_re.search(version)
        if match:
            version = version[:match.start()]
            py_version = match.group(1)
            if py_version != sys.version[:3]:
                self._log_skipped_link(
                    link, 'Python version is incorrect')
                return
        try:
            support_this_python = check_requires_python(link.requires_python)
        except specifiers.InvalidSpecifier:
            logger.debug("Package %s has an invalid Requires-Python entry: %s",
                         link.filename, link.requires_python)
            support_this_python = True

        if not support_this_python:
            logger.debug("The package %s is incompatible with the python"
                         "version in use. Acceptable python versions are:%s",
                         link, link.requires_python)
            return
        logger.debug('Found link %s, version: %s', link, version)

        return InstallationCandidate(search.supplied, version, link)

    def _get_page(self, link):
        return HTMLPage.get_page(link, session=self.session)


def egg_info_matches(
        egg_info, search_name, link,
        _egg_info_re=re.compile(r'([a-z0-9_.]+)-([a-z0-9_.!+-]+)', re.I)):
    """Pull the version part out of a string.

    :param egg_info: The string to parse. E.g. foo-2.1
    :param search_name: The name of the package this belongs to. None to
        infer the name. Note that this cannot unambiguously parse strings
        like foo-2-2 which might be foo, 2-2 or foo-2, 2.
    :param link: The link the string came from, for logging on failure.
    """
    match = _egg_info_re.search(egg_info)
    if not match:
        logger.debug('Could not parse version from link: %s', link)
        return None
    if search_name is None:
        full_match = match.group(0)
        return full_match[full_match.index('-'):]
    name = match.group(0).lower()
    # To match the "safe" name that pkg_resources creates:
    name = name.replace('_', '-')
    # project name and version must be separated by a dash
    look_for = search_name.lower() + "-"
    if name.startswith(look_for):
        return match.group(0)[len(look_for):]
    else:
        return None


class HTMLPage(object):
    """Represents one page, along with its URL"""

    def __init__(self, content, url, headers=None):
        # Determine if we have any encoding information in our headers
        encoding = None
        if headers and "Content-Type" in headers:
            content_type, params = cgi.parse_header(headers["Content-Type"])

            if "charset" in params:
                encoding = params['charset']

        self.content = content
        self.parsed = html5lib.parse(
            self.content,
            transport_encoding=encoding,
            namespaceHTMLElements=False,
        )
        self.url = url
        self.headers = headers

    def __str__(self):
        return self.url

    @classmethod
    def get_page(cls, link, skip_archives=True, session=None):
        if session is None:
            raise TypeError(
                "get_page() missing 1 required keyword argument: 'session'"
            )

        url = link.url
        url = url.split('#', 1)[0]

        # Check for VCS schemes that do not support lookup as web pages.
        from pip.vcs import VcsSupport
        for scheme in VcsSupport.schemes:
            if url.lower().startswith(scheme) and url[len(scheme)] in '+:':
                logger.debug('Cannot look at %s URL %s', scheme, link)
                return None

        try:
            if skip_archives:
                filename = link.filename
                for bad_ext in ARCHIVE_EXTENSIONS:
                    if filename.endswith(bad_ext):
                        content_type = cls._get_content_type(
                            url, session=session,
                        )
                        if content_type.lower().startswith('text/html'):
                            break
                        else:
                            logger.debug(
                                'Skipping page %s because of Content-Type: %s',
                                link,
                                content_type,
                            )
                            return

            logger.debug('Getting page %s', url)

            # Tack index.html onto file:// URLs that point to directories
            (scheme, netloc, path, params, query, fragment) = \
                urllib_parse.urlparse(url)
            if (scheme == 'file' and
                    os.path.isdir(urllib_request.url2pathname(path))):
                # add trailing slash if not present so urljoin doesn't trim
                # final segment
                if not url.endswith('/'):
                    url += '/'
                url = urllib_parse.urljoin(url, 'index.html')
                logger.debug(' file: URL is directory, getting %s', url)

            resp = session.get(
                url,
                headers={
                    "Accept": "text/html",
                    "Cache-Control": "max-age=600",
                },
            )
            resp.raise_for_status()

            # The check for archives above only works if the url ends with
            # something that looks like an archive. However that is not a
            # requirement of an url. Unless we issue a HEAD request on every
            # url we cannot know ahead of time for sure if something is HTML
            # or not. However we can check after we've downloaded it.
            content_type = resp.headers.get('Content-Type', 'unknown')
            if not content_type.lower().startswith("text/html"):
                logger.debug(
                    'Skipping page %s because of Content-Type: %s',
                    link,
                    content_type,
                )
                return

            inst = cls(resp.content, resp.url, resp.headers)
        except requests.HTTPError as exc:
            cls._handle_fail(link, exc, url)
        except SSLError as exc:
            reason = ("There was a problem confirming the ssl certificate: "
                      "%s" % exc)
            cls._handle_fail(link, reason, url, meth=logger.info)
        except requests.ConnectionError as exc:
            cls._handle_fail(link, "connection error: %s" % exc, url)
        except requests.Timeout:
            cls._handle_fail(link, "timed out", url)
        else:
            return inst

    @staticmethod
    def _handle_fail(link, reason, url, meth=None):
        if meth is None:
            meth = logger.debug

        meth("Could not fetch URL %s: %s - skipping", link, reason)

    @staticmethod
    def _get_content_type(url, session):
        """Get the Content-Type of the given url, using a HEAD request"""
        scheme, netloc, path, query, fragment = urllib_parse.urlsplit(url)
        if scheme not in ('http', 'https'):
            # FIXME: some warning or something?
            # assertion error?
            return ''

        resp = session.head(url, allow_redirects=True)
        resp.raise_for_status()

        return resp.headers.get("Content-Type", "")

    @cached_property
    def base_url(self):
        bases = [
            x for x in self.parsed.findall(".//base")
            if x.get("href") is not None
        ]
        if bases and bases[0].get("href"):
            return bases[0].get("href")
        else:
            return self.url

    @property
    def links(self):
        """Yields all links in the page"""
        for anchor in self.parsed.findall(".//a"):
            if anchor.get("href"):
                href = anchor.get("href")
                url = self.clean_link(
                    urllib_parse.urljoin(self.base_url, href)
                )
                pyrequire = anchor.get('data-requires-python')
                pyrequire = unescape(pyrequire) if pyrequire else None
                yield Link(url, self, requires_python=pyrequire)

    _clean_re = re.compile(r'[^a-z0-9$&+,/:;=?@.#%_\\|-]', re.I)

    def clean_link(self, url):
        """Makes sure a link is fully encoded.  That is, if a ' ' shows up in
        the link, it will be rewritten to %20 (while not over-quoting
        % or other characters)."""
        return self._clean_re.sub(
            lambda match: '%%%2x' % ord(match.group(0)), url)


class Link(object):

    def __init__(self, url, comes_from=None, requires_python=None):
        """
        Object representing a parsed link from https://pypi.python.org/simple/*

        url:
            url of the resource pointed to (href of the link)
        comes_from:
            instance of HTMLPage where the link was found, or string.
        requires_python:
            String containing the `Requires-Python` metadata field, specified
            in PEP 345. This may be specified by a data-requires-python
            attribute in the HTML link tag, as described in PEP 503.
        """

        # url can be a UNC windows share
        if url.startswith('\\\\'):
            url = path_to_url(url)

        self.url = url
        self.comes_from = comes_from
        self.requires_python = requires_python if requires_python else None

    def __str__(self):
        if self.requires_python:
            rp = ' (requires-python:%s)' % self.requires_python
        else:
            rp = ''
        if self.comes_from:
            return '%s (from %s)%s' % (self.url, self.comes_from, rp)
        else:
            return str(self.url)

    def __repr__(self):
        return '<Link %s>' % self

    def __eq__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url == other.url

    def __ne__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url != other.url

    def __lt__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url < other.url

    def __le__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url <= other.url

    def __gt__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url > other.url

    def __ge__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.url >= other.url

    def __hash__(self):
        return hash(self.url)

    @property
    def filename(self):
        _, netloc, path, _, _ = urllib_parse.urlsplit(self.url)
        name = posixpath.basename(path.rstrip('/')) or netloc
        name = urllib_parse.unquote(name)
        assert name, ('URL %r produced no filename' % self.url)
        return name

    @property
    def scheme(self):
        return urllib_parse.urlsplit(self.url)[0]

    @property
    def netloc(self):
        return urllib_parse.urlsplit(self.url)[1]

    @property
    def path(self):
        return urllib_parse.unquote(urllib_parse.urlsplit(self.url)[2])

    def splitext(self):
        return splitext(posixpath.basename(self.path.rstrip('/')))

    @property
    def ext(self):
        return self.splitext()[1]

    @property
    def url_without_fragment(self):
        scheme, netloc, path, query, fragment = urllib_parse.urlsplit(self.url)
        return urllib_parse.urlunsplit((scheme, netloc, path, query, None))

    _egg_fragment_re = re.compile(r'[#&]egg=([^&]*)')

    @property
    def egg_fragment(self):
        match = self._egg_fragment_re.search(self.url)
        if not match:
            return None
        return match.group(1)

    _subdirectory_fragment_re = re.compile(r'[#&]subdirectory=([^&]*)')

    @property
    def subdirectory_fragment(self):
        match = self._subdirectory_fragment_re.search(self.url)
        if not match:
            return None
        return match.group(1)

    _hash_re = re.compile(
        r'(sha1|sha224|sha384|sha256|sha512|md5)=([a-f0-9]+)'
    )

    @property
    def hash(self):
        match = self._hash_re.search(self.url)
        if match:
            return match.group(2)
        return None

    @property
    def hash_name(self):
        match = self._hash_re.search(self.url)
        if match:
            return match.group(1)
        return None

    @property
    def show_url(self):
        return posixpath.basename(self.url.split('#', 1)[0].split('?', 1)[0])

    @property
    def is_wheel(self):
        return self.ext == wheel_ext

    @property
    def is_artifact(self):
        """
        Determines if this points to an actual artifact (e.g. a tarball) or if
        it points to an "abstract" thing like a path or a VCS location.
        """
        from pip.vcs import vcs

        if self.scheme in vcs.all_schemes:
            return False

        return True


FormatControl = namedtuple('FormatControl', 'no_binary only_binary')
"""This object has two fields, no_binary and only_binary.

If a field is falsy, it isn't set. If it is {':all:'}, it should match all
packages except those listed in the other field. Only one field can be set
to {':all:'} at a time. The rest of the time exact package name matches
are listed, with any given package only showing up in one field at a time.
"""


def fmt_ctl_handle_mutual_exclude(value, target, other):
    new = value.split(',')
    while ':all:' in new:
        other.clear()
        target.clear()
        target.add(':all:')
        del new[:new.index(':all:') + 1]
        if ':none:' not in new:
            # Without a none, we want to discard everything as :all: covers it
            return
    for name in new:
        if name == ':none:':
            target.clear()
            continue
        name = canonicalize_name(name)
        other.discard(name)
        target.add(name)


def fmt_ctl_formats(fmt_ctl, canonical_name):
    result = set(["binary", "source"])
    if canonical_name in fmt_ctl.only_binary:
        result.discard('source')
    elif canonical_name in fmt_ctl.no_binary:
        result.discard('binary')
    elif ':all:' in fmt_ctl.only_binary:
        result.discard('source')
    elif ':all:' in fmt_ctl.no_binary:
        result.discard('binary')
    return frozenset(result)


def fmt_ctl_no_binary(fmt_ctl):
    fmt_ctl_handle_mutual_exclude(
        ':all:', fmt_ctl.no_binary, fmt_ctl.only_binary)


def fmt_ctl_no_use_wheel(fmt_ctl):
    fmt_ctl_no_binary(fmt_ctl)
    warnings.warn(
        '--no-use-wheel is deprecated and will be removed in the future. '
        ' Please use --no-binary :all: instead.', RemovedInPip10Warning,
        stacklevel=2)


Search = namedtuple('Search', 'supplied canonical formats')
"""Capture key aspects of a search.

:attribute supplied: The user supplied package.
:attribute canonical: The canonical package name.
:attribute formats: The formats allowed for this package. Should be a set
    with 'binary' or 'source' or both in it.
"""
