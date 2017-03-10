"""
Operations on existing wheel files, including basic installation. 
"""
# XXX see patched pip to install

import sys
import warnings
import os.path
import re
import zipfile
import hashlib
import csv

import shutil

try:
    _big_number = sys.maxsize
except NameError:
    _big_number = sys.maxint

from wheel.decorator import reify
from wheel.util import (urlsafe_b64encode, from_json, urlsafe_b64decode,
                        native, binary, HashingFile)
from wheel import signatures
from wheel.pkginfo import read_pkg_info_bytes
from wheel.util import open_for_csv

from .pep425tags import get_supported
from .paths import get_install_paths

# The next major version after this version of the 'wheel' tool:
VERSION_TOO_HIGH = (1, 0)

# Non-greedy matching of an optional build number may be too clever (more
# invalid wheel filenames will match). Separate regex for .dist-info?
WHEEL_INFO_RE = re.compile(
    r"""^(?P<namever>(?P<name>.+?)(-(?P<ver>\d.+?))?)
    ((-(?P<build>\d.*?))?-(?P<pyver>.+?)-(?P<abi>.+?)-(?P<plat>.+?)
    \.whl|\.dist-info)$""",
    re.VERBOSE).match

def parse_version(version):
    """Use parse_version from pkg_resources or distutils as available."""
    global parse_version
    try:
        from pkg_resources import parse_version
    except ImportError:
        from distutils.version import LooseVersion as parse_version
    return parse_version(version)

class BadWheelFile(ValueError):
    pass


class WheelFile(object):
    """Parse wheel-specific attributes from a wheel (.whl) file and offer
    basic installation and verification support.
    
    WheelFile can be used to simply parse a wheel filename by avoiding the
    methods that require the actual file contents."""

    WHEEL_INFO = "WHEEL"
    RECORD = "RECORD"

    def __init__(self,
                 filename,
                 fp=None,
                 append=False,
                 context=get_supported):
        """
        :param fp: A seekable file-like object or None to open(filename).
        :param append: Open archive in append mode.
        :param context: Function returning list of supported tags. Wheels
        must have the same context to be sortable.
        """
        self.filename = filename
        self.fp = fp
        self.append = append
        self.context = context
        basename = os.path.basename(filename)
        self.parsed_filename = WHEEL_INFO_RE(basename)
        if not basename.endswith('.whl') or self.parsed_filename is None:
            raise BadWheelFile("Bad filename '%s'" % filename)

    def __repr__(self):
        return self.filename

    @property
    def distinfo_name(self):
        return "%s.dist-info" % self.parsed_filename.group('namever')

    @property
    def datadir_name(self):
        return "%s.data" % self.parsed_filename.group('namever')

    @property
    def record_name(self):
        return "%s/%s" % (self.distinfo_name, self.RECORD)

    @property
    def wheelinfo_name(self):
        return "%s/%s" % (self.distinfo_name, self.WHEEL_INFO)

    @property
    def tags(self):
        """A wheel file is compatible with the Cartesian product of the
        period-delimited tags in its filename.
        To choose a wheel file among several candidates having the same
        distribution version 'ver', an installer ranks each triple of
        (pyver, abi, plat) that its Python installation can run, sorting
        the wheels by the best-ranked tag it supports and then by their
        arity which is just len(list(compatibility_tags)).
        """
        tags = self.parsed_filename.groupdict()
        for pyver in tags['pyver'].split('.'):
            for abi in tags['abi'].split('.'):
                for plat in tags['plat'].split('.'):
                    yield (pyver, abi, plat)

    compatibility_tags = tags

    @property
    def arity(self):
        """The number of compatibility tags the wheel declares."""
        return len(list(self.compatibility_tags))

    @property
    def rank(self):
        """
        Lowest index of any of this wheel's tags in self.context(), and the
        arity e.g. (0, 1)
        """
        return self.compatibility_rank(self.context())

    @property
    def compatible(self):
        return self.rank[0] != _big_number  # bad API!

    # deprecated:
    def compatibility_rank(self, supported):
        """Rank the wheel against the supported tags. Smaller ranks are more
        compatible!

        :param supported: A list of compatibility tags that the current
            Python implemenation can run.
        """
        preferences = []
        for tag in self.compatibility_tags:
            try:
                preferences.append(supported.index(tag))
            # Tag not present
            except ValueError:
                pass
        if len(preferences):
            return (min(preferences), self.arity)
        return (_big_number, 0)

    # deprecated
    def supports_current_python(self, x):
        assert self.context == x, 'context mismatch'
        return self.compatible

    # Comparability.
    # Wheels are equal if they refer to the same file.
    # If two wheels are not equal, compare based on (in this order):
    #   1. Name
    #   2. Version
    #   3. Compatibility rank
    #   4. Filename (as a tiebreaker)
    @property
    def _sort_key(self):
        return (self.parsed_filename.group('name'),
                parse_version(self.parsed_filename.group('ver')),
                tuple(-x for x in self.rank),
                self.filename)

    def __eq__(self, other):
        return self.filename == other.filename

    def __ne__(self, other):
        return self.filename != other.filename

    def __lt__(self, other):
        if self.context != other.context:
            raise TypeError("{0}.context != {1}.context".format(self, other))

        return self._sort_key < other._sort_key

        # XXX prune

        sn = self.parsed_filename.group('name')
        on = other.parsed_filename.group('name')
        if sn != on:
            return sn < on
        sv = parse_version(self.parsed_filename.group('ver'))
        ov = parse_version(other.parsed_filename.group('ver'))
        if sv != ov:
            return sv < ov
        # Compatibility
        if self.context != other.context:
            raise TypeError("{0}.context != {1}.context".format(self, other))
        sc = self.rank
        oc = other.rank
        if sc != None and oc != None and sc != oc:
            # Smaller compatibility ranks are "better" than larger ones,
            # so we have to reverse the sense of the comparison here!
            return sc > oc
        elif sc == None and oc != None:
            return False
        return self.filename < other.filename

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or other < self

    #
    # Methods using the file's contents:
    #

    @reify
    def zipfile(self):
        mode = "r"
        if self.append:
            mode = "a"
        vzf = VerifyingZipFile(self.fp if self.fp else self.filename, mode)
        if not self.append:
            self.verify(vzf)
        return vzf

    @reify
    def parsed_wheel_info(self):
        """Parse wheel metadata (the .data/WHEEL file)"""
        return read_pkg_info_bytes(self.zipfile.read(self.wheelinfo_name))

    def check_version(self):
        version = self.parsed_wheel_info['Wheel-Version']
        if tuple(map(int, version.split('.'))) >= VERSION_TOO_HIGH:
            raise ValueError("Wheel version is too high")

    @reify
    def install_paths(self):
        """
        Consult distutils to get the install paths for our dist.  A dict with
        ('purelib', 'platlib', 'headers', 'scripts', 'data').
        
        We use the name from our filename as the dist name, which means headers
        could be installed in the wrong place if the filesystem-escaped name
        is different than the Name.  Who cares? 
        """
        name = self.parsed_filename.group('name')
        return get_install_paths(name)

    def install(self, force=False, overrides={}):
        """
        Install the wheel into site-packages.
        """

        # Utility to get the target directory for a particular key
        def get_path(key):
            return overrides.get(key) or self.install_paths[key]

        # The base target location is either purelib or platlib
        if self.parsed_wheel_info['Root-Is-Purelib'] == 'true':
            root = get_path('purelib')
        else:
            root = get_path('platlib')

        # Parse all the names in the archive
        name_trans = {}
        for info in self.zipfile.infolist():
            name = info.filename
            # Zip files can contain entries representing directories.
            # These end in a '/'.
            # We ignore these, as we create directories on demand.
            if name.endswith('/'):
                continue

            # Pathnames in a zipfile namelist are always /-separated.
            # In theory, paths could start with ./ or have other oddities
            # but this won't happen in practical cases of well-formed wheels.
            # We'll cover the simple case of an initial './' as it's both easy
            # to do and more common than most other oddities.
            if name.startswith('./'):
                name = name[2:]

            # Split off the base directory to identify files that are to be
            # installed in non-root locations
            basedir, sep, filename = name.partition('/')
            if sep and basedir == self.datadir_name:
                # Data file. Target destination is elsewhere
                key, sep, filename = filename.partition('/')
                if not sep:
                    raise ValueError("Invalid filename in wheel: {0}".format(name))
                target = get_path(key)
            else:
                # Normal file. Target destination is root
                key = ''
                target = root
                filename = name

            # Map the actual filename from the zipfile to its intended target
            # directory and the pathname relative to that directory.
            dest = os.path.normpath(os.path.join(target, filename))
            name_trans[info] = (key, target, filename, dest)

        # We're now ready to start processing the actual install. The process
        # is as follows:
        #   1. Prechecks - is the wheel valid, is its declared architecture
        #      OK, etc. [[Responsibility of the caller]]
        #   2. Overwrite check - do any of the files to be installed already
        #      exist?
        #   3. Actual install - put the files in their target locations.
        #   4. Update RECORD - write a suitably modified RECORD file to
        #      reflect the actual installed paths.

        if not force:
            for info, v in name_trans.items():
                k = info.filename
                key, target, filename, dest = v
                if os.path.exists(dest):
                    raise ValueError("Wheel file {0} would overwrite {1}. Use force if this is intended".format(k, dest))

        # Get the name of our executable, for use when replacing script
        # wrapper hashbang lines.
        # We encode it using getfilesystemencoding, as that is "the name of
        # the encoding used to convert Unicode filenames into system file
        # names".
        exename = sys.executable.encode(sys.getfilesystemencoding())
        record_data = []
        record_name = self.distinfo_name + '/RECORD'
        for info, (key, target, filename, dest) in name_trans.items():
            name = info.filename
            source = self.zipfile.open(info)
            # Skip the RECORD file
            if name == record_name:
                continue
            ddir = os.path.dirname(dest)
            if not os.path.isdir(ddir):
                os.makedirs(ddir)
            destination = HashingFile(open(dest, 'wb'))
            if key == 'scripts':
                hashbang = source.readline()
                if hashbang.startswith(b'#!python'):
                    hashbang = b'#!' + exename + binary(os.linesep)
                destination.write(hashbang)
            shutil.copyfileobj(source, destination)
            reldest = os.path.relpath(dest, root)
            reldest.replace(os.sep, '/')
            record_data.append((reldest, destination.digest(), destination.length))
            destination.close()
            source.close()
            # preserve attributes (especially +x bit for scripts)
            attrs = info.external_attr >> 16
            if attrs:  # tends to be 0 if Windows.
                os.chmod(dest, info.external_attr >> 16)

        record_name = os.path.join(root, self.record_name)
        writer = csv.writer(open_for_csv(record_name, 'w+'))
        for reldest, digest, length in sorted(record_data):
            writer.writerow((reldest, digest, length))
        writer.writerow((self.record_name, '', ''))

    def verify(self, zipfile=None):
        """Configure the VerifyingZipFile `zipfile` by verifying its signature 
        and setting expected hashes for every hash in RECORD.
        Caller must complete the verification process by completely reading 
        every file in the archive (e.g. with extractall)."""
        sig = None
        if zipfile is None:
            zipfile = self.zipfile
        zipfile.strict = True

        record_name = '/'.join((self.distinfo_name, 'RECORD'))
        sig_name = '/'.join((self.distinfo_name, 'RECORD.jws'))
        # tolerate s/mime signatures:
        smime_sig_name = '/'.join((self.distinfo_name, 'RECORD.p7s'))
        zipfile.set_expected_hash(record_name, None)
        zipfile.set_expected_hash(sig_name, None)
        zipfile.set_expected_hash(smime_sig_name, None)
        record = zipfile.read(record_name)

        record_digest = urlsafe_b64encode(hashlib.sha256(record).digest())
        try:
            sig = from_json(native(zipfile.read(sig_name)))
        except KeyError:  # no signature
            pass
        if sig:
            headers, payload = signatures.verify(sig)
            if payload['hash'] != "sha256=" + native(record_digest):
                msg = "RECORD.sig claimed RECORD hash {0} != computed hash {1}."
                raise BadWheelFile(msg.format(payload['hash'],
                                              native(record_digest)))

        reader = csv.reader((native(r) for r in record.splitlines()))

        for row in reader:
            filename = row[0]
            hash = row[1]
            if not hash:
                if filename not in (record_name, sig_name):
                    sys.stderr.write("%s has no hash!\n" % filename)
                continue
            algo, data = row[1].split('=', 1)
            assert algo == "sha256", "Unsupported hash algorithm"
            zipfile.set_expected_hash(filename, urlsafe_b64decode(binary(data)))


class VerifyingZipFile(zipfile.ZipFile):
    """ZipFile that can assert that each of its extracted contents matches
    an expected sha256 hash. Note that each file must be completly read in 
    order for its hash to be checked."""

    def __init__(self, file, mode="r",
                 compression=zipfile.ZIP_STORED,
                 allowZip64=False):
        zipfile.ZipFile.__init__(self, file, mode, compression, allowZip64)

        self.strict = False
        self._expected_hashes = {}
        self._hash_algorithm = hashlib.sha256

    def set_expected_hash(self, name, hash):
        """
        :param name: name of zip entry
        :param hash: bytes of hash (or None for "don't care")
        """
        self._expected_hashes[name] = hash

    def open(self, name_or_info, mode="r", pwd=None):
        """Return file-like object for 'name'."""
        # A non-monkey-patched version would contain most of zipfile.py
        ef = zipfile.ZipFile.open(self, name_or_info, mode, pwd)
        if isinstance(name_or_info, zipfile.ZipInfo):
            name = name_or_info.filename
        else:
            name = name_or_info
        if (name in self._expected_hashes
            and self._expected_hashes[name] != None):
            expected_hash = self._expected_hashes[name]
            try:
                _update_crc_orig = ef._update_crc
            except AttributeError:
                warnings.warn('Need ZipExtFile._update_crc to implement '
                              'file hash verification (in Python >= 2.7)')
                return ef
            running_hash = self._hash_algorithm()
            if hasattr(ef, '_eof'):  # py33
                def _update_crc(data):
                    _update_crc_orig(data)
                    running_hash.update(data)
                    if ef._eof and running_hash.digest() != expected_hash:
                        raise BadWheelFile("Bad hash for file %r" % ef.name)
            else:
                def _update_crc(data, eof=None):
                    _update_crc_orig(data, eof=eof)
                    running_hash.update(data)
                    if eof and running_hash.digest() != expected_hash:
                        raise BadWheelFile("Bad hash for file %r" % ef.name)
            ef._update_crc = _update_crc
        elif self.strict and name not in self._expected_hashes:
            raise BadWheelFile("No expected hash for file %r" % ef.name)
        return ef

    def pop(self):
        """Truncate the last file off this zipfile.
        Assumes infolist() is in the same order as the files (true for
        ordinary zip files created by Python)"""
        if not self.fp:
            raise RuntimeError(
                  "Attempt to pop from ZIP archive that was already closed")
        last = self.infolist().pop()
        del self.NameToInfo[last.filename]
        self.fp.seek(last.header_offset, os.SEEK_SET)
        self.fp.truncate()
        self._didModify = True
