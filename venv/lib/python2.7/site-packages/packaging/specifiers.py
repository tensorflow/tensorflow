# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import absolute_import, division, print_function

import abc
import functools
import itertools
import re

from ._compat import string_types, with_metaclass
from .version import Version, LegacyVersion, parse


class InvalidSpecifier(ValueError):
    """
    An invalid specifier was found, users should refer to PEP 440.
    """


class BaseSpecifier(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def __str__(self):
        """
        Returns the str representation of this Specifier like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self):
        """
        Returns a hash value for this Specifier like object.
        """

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are equal.
        """

    @abc.abstractmethod
    def __ne__(self, other):
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are not equal.
        """

    @abc.abstractproperty
    def prereleases(self):
        """
        Returns whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @prereleases.setter
    def prereleases(self, value):
        """
        Sets whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @abc.abstractmethod
    def contains(self, item, prereleases=None):
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(self, iterable, prereleases=None):
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class _IndividualSpecifier(BaseSpecifier):

    _operators = {}

    def __init__(self, spec="", prereleases=None):
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier("Invalid specifier: '{0}'".format(spec))

        self._spec = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    def __repr__(self):
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<{0}({1!r}{2})>".format(
            self.__class__.__name__,
            str(self),
            pre,
        )

    def __str__(self):
        return "{0}{1}".format(*self._spec)

    def __hash__(self):
        return hash(self._spec)

    def __eq__(self, other):
        if isinstance(other, string_types):
            try:
                other = self.__class__(other)
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._spec == other._spec

    def __ne__(self, other):
        if isinstance(other, string_types):
            try:
                other = self.__class__(other)
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._spec != other._spec

    def _get_operator(self, op):
        return getattr(self, "_compare_{0}".format(self._operators[op]))

    def _coerce_version(self, version):
        if not isinstance(version, (LegacyVersion, Version)):
            version = parse(version)
        return version

    @property
    def operator(self):
        return self._spec[0]

    @property
    def version(self):
        return self._spec[1]

    @property
    def prereleases(self):
        return self._prereleases

    @prereleases.setter
    def prereleases(self, value):
        self._prereleases = value

    def __contains__(self, item):
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version or LegacyVersion, this allows us to have
        # a shortcut for ``"2.0" in Specifier(">=2")
        item = self._coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        return self._get_operator(self.operator)(item, self.version)

    def filter(self, iterable, prereleases=None):
        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = self._coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later incase nothing
                # else matches this specifier.
                if (parsed_version.is_prerelease and not
                        (prereleases or self.prereleases)):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the begining.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


class LegacySpecifier(_IndividualSpecifier):

    _regex_str = (
        r"""
        (?P<operator>(==|!=|<=|>=|<|>))
        \s*
        (?P<version>
            [^,;\s)]* # Since this is a "legacy" specifier, and the version
                      # string can be just about anything, we match everything
                      # except for whitespace, a semi-colon for marker support,
                      # a closing paren since versions can be enclosed in
                      # them, and a comma since it's a version separator.
        )
        """
    )

    _regex = re.compile(
        r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
    }

    def _coerce_version(self, version):
        if not isinstance(version, LegacyVersion):
            version = LegacyVersion(str(version))
        return version

    def _compare_equal(self, prospective, spec):
        return prospective == self._coerce_version(spec)

    def _compare_not_equal(self, prospective, spec):
        return prospective != self._coerce_version(spec)

    def _compare_less_than_equal(self, prospective, spec):
        return prospective <= self._coerce_version(spec)

    def _compare_greater_than_equal(self, prospective, spec):
        return prospective >= self._coerce_version(spec)

    def _compare_less_than(self, prospective, spec):
        return prospective < self._coerce_version(spec)

    def _compare_greater_than(self, prospective, spec):
        return prospective > self._coerce_version(spec)


def _require_version_compare(fn):
    @functools.wraps(fn)
    def wrapped(self, prospective, spec):
        if not isinstance(prospective, Version):
            return False
        return fn(self, prospective, spec)
    return wrapped


class Specifier(_IndividualSpecifier):

    _regex_str = (
        r"""
        (?P<operator>(~=|==|!=|<=|>=|<|>|===))
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s]*    # We just match everything, except for whitespace
                          # since we are only testing for strict identity.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?

                # You cannot use a wild card and a dev or local version
                # together so group them with a | and make them optional.
                (?:
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                    |
                    \.\*  # Wild card syntax of .*
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """
    )

    _regex = re.compile(
        r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    @_require_version_compare
    def _compare_compatible(self, prospective, spec):
        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.

        # We want everything but the last item in the version, but we want to
        # ignore post and dev releases and we want to treat the pre-release as
        # it's own separate segment.
        prefix = ".".join(
            list(
                itertools.takewhile(
                    lambda x: (not x.startswith("post") and not
                               x.startswith("dev")),
                    _version_split(spec),
                )
            )[:-1]
        )

        # Add the prefix notation to the end of our string
        prefix += ".*"

        return (self._get_operator(">=")(prospective, spec) and
                self._get_operator("==")(prospective, prefix))

    @_require_version_compare
    def _compare_equal(self, prospective, spec):
        # We need special logic to handle prefix matching
        if spec.endswith(".*"):
            # In the case of prefix matching we want to ignore local segment.
            prospective = Version(prospective.public)
            # Split the spec out by dots, and pretend that there is an implicit
            # dot in between a release segment and a pre-release segment.
            spec = _version_split(spec[:-2])  # Remove the trailing .*

            # Split the prospective version out by dots, and pretend that there
            # is an implicit dot in between a release segment and a pre-release
            # segment.
            prospective = _version_split(str(prospective))

            # Shorten the prospective version to be the same length as the spec
            # so that we can determine if the specifier is a prefix of the
            # prospective version or not.
            prospective = prospective[:len(spec)]

            # Pad out our two sides with zeros so that they both equal the same
            # length.
            spec, prospective = _pad_version(spec, prospective)
        else:
            # Convert our spec string into a Version
            spec = Version(spec)

            # If the specifier does not have a local segment, then we want to
            # act as if the prospective version also does not have a local
            # segment.
            if not spec.local:
                prospective = Version(prospective.public)

        return prospective == spec

    @_require_version_compare
    def _compare_not_equal(self, prospective, spec):
        return not self._compare_equal(prospective, spec)

    @_require_version_compare
    def _compare_less_than_equal(self, prospective, spec):
        return prospective <= Version(spec)

    @_require_version_compare
    def _compare_greater_than_equal(self, prospective, spec):
        return prospective >= Version(spec)

    @_require_version_compare
    def _compare_less_than(self, prospective, spec):
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec)

        # Check to see if the prospective version is less than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    @_require_version_compare
    def _compare_greater_than(self, prospective, spec):
        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec)

        # Check to see if the prospective version is greater than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is techincally greater than, to match.
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    def _compare_arbitrary(self, prospective, spec):
        return str(prospective).lower() == str(spec).lower()

    @property
    def prereleases(self):
        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Look at all of our specifiers and determine if they are inclusive
        # operators, and if they are if they are including an explicit
        # prerelease.
        operator, version = self._spec
        if operator in ["==", ">=", "<=", "~=", "==="]:
            # The == specifier can include a trailing .*, if it does we
            # want to remove before parsing.
            if operator == "==" and version.endswith(".*"):
                version = version[:-2]

            # Parse the version, and if it is a pre-release than this
            # specifier allows pre-releases.
            if parse(version).is_prerelease:
                return True

        return False

    @prereleases.setter
    def prereleases(self, value):
        self._prereleases = value


_prefix_regex = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


def _version_split(version):
    result = []
    for item in version.split("."):
        match = _prefix_regex.search(item)
        if match:
            result.extend(match.groups())
        else:
            result.append(item)
    return result


def _pad_version(left, right):
    left_split, right_split = [], []

    # Get the release segment of our versions
    left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
    right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

    # Get the rest of our versions
    left_split.append(left[len(left_split[0]):])
    right_split.append(right[len(right_split[0]):])

    # Insert our padding
    left_split.insert(
        1,
        ["0"] * max(0, len(right_split[0]) - len(left_split[0])),
    )
    right_split.insert(
        1,
        ["0"] * max(0, len(left_split[0]) - len(right_split[0])),
    )

    return (
        list(itertools.chain(*left_split)),
        list(itertools.chain(*right_split)),
    )


class SpecifierSet(BaseSpecifier):

    def __init__(self, specifiers="", prereleases=None):
        # Split on , to break each indidivual specifier into it's own item, and
        # strip each item to remove leading/trailing whitespace.
        specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

        # Parsed each individual specifier, attempting first to make it a
        # Specifier and falling back to a LegacySpecifier.
        parsed = set()
        for specifier in specifiers:
            try:
                parsed.add(Specifier(specifier))
            except InvalidSpecifier:
                parsed.add(LegacySpecifier(specifier))

        # Turn our parsed specifiers into a frozen set and save them for later.
        self._specs = frozenset(parsed)

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    def __repr__(self):
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<SpecifierSet({0!r}{1})>".format(str(self), pre)

    def __str__(self):
        return ",".join(sorted(str(s) for s in self._specs))

    def __hash__(self):
        return hash(self._specs)

    def __and__(self, other):
        if isinstance(other, string_types):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)

        if self._prereleases is None and other._prereleases is not None:
            specifier._prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease "
                "overrides."
            )

        return specifier

    def __eq__(self, other):
        if isinstance(other, string_types):
            other = SpecifierSet(other)
        elif isinstance(other, _IndividualSpecifier):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs == other._specs

    def __ne__(self, other):
        if isinstance(other, string_types):
            other = SpecifierSet(other)
        elif isinstance(other, _IndividualSpecifier):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs != other._specs

    def __len__(self):
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    @property
    def prereleases(self):
        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @prereleases.setter
    def prereleases(self, value):
        self._prereleases = value

    def __contains__(self, item):
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # Ensure that our item is a Version or LegacyVersion instance.
        if not isinstance(item, (LegacyVersion, Version)):
            item = parse(item)

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # We can determine if we're going to allow pre-releases by looking to
        # see if any of the underlying items supports them. If none of them do
        # and this item is a pre-release then we do not allow it and we can
        # short circuit that here.
        # Note: This means that 1.0.dev1 would not be contained in something
        #       like >=1.0.devabc however it would be in >=1.0.debabc,>0.0.dev0
        if not prereleases and item.is_prerelease:
            return False

        # We simply dispatch to the underlying specs here to make sure that the
        # given version is contained within all of them.
        # Note: This use of all() here means that an empty set of specifiers
        #       will always return True, this is an explicit design decision.
        return all(
            s.contains(item, prereleases=prereleases)
            for s in self._specs
        )

    def filter(self, iterable, prereleases=None):
        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iterable
        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases, and which will filter out LegacyVersion in general.
        else:
            filtered = []
            found_prereleases = []

            for item in iterable:
                # Ensure that we some kind of Version class for this item.
                if not isinstance(item, (LegacyVersion, Version)):
                    parsed_version = parse(item)
                else:
                    parsed_version = item

                # Filter out any item which is parsed as a LegacyVersion
                if isinstance(parsed_version, LegacyVersion):
                    continue

                # Store any item which is a pre-release for later unless we've
                # already found a final version or we are accepting prereleases
                if parsed_version.is_prerelease and not prereleases:
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)

            # If we've found no items except for pre-releases, then we'll go
            # ahead and use the pre-releases
            if not filtered and found_prereleases and prereleases is None:
                return found_prereleases

            return filtered
