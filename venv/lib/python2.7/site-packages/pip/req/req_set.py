from __future__ import absolute_import

from collections import defaultdict
from itertools import chain
import logging
import os

from pip._vendor import pkg_resources
from pip._vendor import requests

from pip.compat import expanduser
from pip.download import (is_file_url, is_dir_url, is_vcs_url, url_to_path,
                          unpack_url)
from pip.exceptions import (InstallationError, BestVersionAlreadyInstalled,
                            DistributionNotFound, PreviousBuildDirError,
                            HashError, HashErrors, HashUnpinned,
                            DirectoryUrlHashUnsupported, VcsHashUnsupported,
                            UnsupportedPythonVersion)
from pip.req.req_install import InstallRequirement
from pip.utils import (
    display_path, dist_in_usersite, ensure_dir, normalize_path)
from pip.utils.hashes import MissingHashes
from pip.utils.logging import indent_log
from pip.utils.packaging import check_dist_requires_python
from pip.vcs import vcs
from pip.wheel import Wheel

logger = logging.getLogger(__name__)


class Requirements(object):

    def __init__(self):
        self._keys = []
        self._dict = {}

    def keys(self):
        return self._keys

    def values(self):
        return [self._dict[key] for key in self._keys]

    def __contains__(self, item):
        return item in self._keys

    def __setitem__(self, key, value):
        if key not in self._keys:
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        values = ['%s: %s' % (repr(k), repr(self[k])) for k in self.keys()]
        return 'Requirements({%s})' % ', '.join(values)


class DistAbstraction(object):
    """Abstracts out the wheel vs non-wheel prepare_files logic.

    The requirements for anything installable are as follows:
     - we must be able to determine the requirement name
       (or we can't correctly handle the non-upgrade case).
     - we must be able to generate a list of run-time dependencies
       without installing any additional packages (or we would
       have to either burn time by doing temporary isolated installs
       or alternatively violate pips 'don't start installing unless
       all requirements are available' rule - neither of which are
       desirable).
     - for packages with setup requirements, we must also be able
       to determine their requirements without installing additional
       packages (for the same reason as run-time dependencies)
     - we must be able to create a Distribution object exposing the
       above metadata.
    """

    def __init__(self, req_to_install):
        self.req_to_install = req_to_install

    def dist(self, finder):
        """Return a setuptools Dist object."""
        raise NotImplementedError(self.dist)

    def prep_for_dist(self):
        """Ensure that we can get a Dist for this requirement."""
        raise NotImplementedError(self.dist)


def make_abstract_dist(req_to_install):
    """Factory to make an abstract dist object.

    Preconditions: Either an editable req with a source_dir, or satisfied_by or
    a wheel link, or a non-editable req with a source_dir.

    :return: A concrete DistAbstraction.
    """
    if req_to_install.editable:
        return IsSDist(req_to_install)
    elif req_to_install.link and req_to_install.link.is_wheel:
        return IsWheel(req_to_install)
    else:
        return IsSDist(req_to_install)


class IsWheel(DistAbstraction):

    def dist(self, finder):
        return list(pkg_resources.find_distributions(
            self.req_to_install.source_dir))[0]

    def prep_for_dist(self):
        # FIXME:https://github.com/pypa/pip/issues/1112
        pass


class IsSDist(DistAbstraction):

    def dist(self, finder):
        dist = self.req_to_install.get_dist()
        # FIXME: shouldn't be globally added:
        if dist.has_metadata('dependency_links.txt'):
            finder.add_dependency_links(
                dist.get_metadata_lines('dependency_links.txt')
            )
        return dist

    def prep_for_dist(self):
        self.req_to_install.run_egg_info()
        self.req_to_install.assert_source_matches_version()


class Installed(DistAbstraction):

    def dist(self, finder):
        return self.req_to_install.satisfied_by

    def prep_for_dist(self):
        pass


class RequirementSet(object):

    def __init__(self, build_dir, src_dir, download_dir, upgrade=False,
                 upgrade_strategy=None, ignore_installed=False, as_egg=False,
                 target_dir=None, ignore_dependencies=False,
                 force_reinstall=False, use_user_site=False, session=None,
                 pycompile=True, isolated=False, wheel_download_dir=None,
                 wheel_cache=None, require_hashes=False,
                 ignore_requires_python=False):
        """Create a RequirementSet.

        :param wheel_download_dir: Where still-packed .whl files should be
            written to. If None they are written to the download_dir parameter.
            Separate to download_dir to permit only keeping wheel archives for
            pip wheel.
        :param download_dir: Where still packed archives should be written to.
            If None they are not saved, and are deleted immediately after
            unpacking.
        :param wheel_cache: The pip wheel cache, for passing to
            InstallRequirement.
        """
        if session is None:
            raise TypeError(
                "RequirementSet() missing 1 required keyword argument: "
                "'session'"
            )

        self.build_dir = build_dir
        self.src_dir = src_dir
        # XXX: download_dir and wheel_download_dir overlap semantically and may
        # be combined if we're willing to have non-wheel archives present in
        # the wheelhouse output by 'pip wheel'.
        self.download_dir = download_dir
        self.upgrade = upgrade
        self.upgrade_strategy = upgrade_strategy
        self.ignore_installed = ignore_installed
        self.force_reinstall = force_reinstall
        self.requirements = Requirements()
        # Mapping of alias: real_name
        self.requirement_aliases = {}
        self.unnamed_requirements = []
        self.ignore_dependencies = ignore_dependencies
        self.ignore_requires_python = ignore_requires_python
        self.successfully_downloaded = []
        self.successfully_installed = []
        self.reqs_to_cleanup = []
        self.as_egg = as_egg
        self.use_user_site = use_user_site
        self.target_dir = target_dir  # set from --target option
        self.session = session
        self.pycompile = pycompile
        self.isolated = isolated
        if wheel_download_dir:
            wheel_download_dir = normalize_path(wheel_download_dir)
        self.wheel_download_dir = wheel_download_dir
        self._wheel_cache = wheel_cache
        self.require_hashes = require_hashes
        # Maps from install_req -> dependencies_of_install_req
        self._dependencies = defaultdict(list)

    def __str__(self):
        reqs = [req for req in self.requirements.values()
                if not req.comes_from]
        reqs.sort(key=lambda req: req.name.lower())
        return ' '.join([str(req.req) for req in reqs])

    def __repr__(self):
        reqs = [req for req in self.requirements.values()]
        reqs.sort(key=lambda req: req.name.lower())
        reqs_str = ', '.join([str(req.req) for req in reqs])
        return ('<%s object; %d requirement(s): %s>'
                % (self.__class__.__name__, len(reqs), reqs_str))

    def add_requirement(self, install_req, parent_req_name=None,
                        extras_requested=None):
        """Add install_req as a requirement to install.

        :param parent_req_name: The name of the requirement that needed this
            added. The name is used because when multiple unnamed requirements
            resolve to the same name, we could otherwise end up with dependency
            links that point outside the Requirements set. parent_req must
            already be added. Note that None implies that this is a user
            supplied requirement, vs an inferred one.
        :param extras_requested: an iterable of extras used to evaluate the
            environement markers.
        :return: Additional requirements to scan. That is either [] if
            the requirement is not applicable, or [install_req] if the
            requirement is applicable and has just been added.
        """
        name = install_req.name
        if not install_req.match_markers(extras_requested):
            logger.warning("Ignoring %s: markers '%s' don't match your "
                           "environment", install_req.name,
                           install_req.markers)
            return []

        # This check has to come after we filter requirements with the
        # environment markers.
        if install_req.link and install_req.link.is_wheel:
            wheel = Wheel(install_req.link.filename)
            if not wheel.supported():
                raise InstallationError(
                    "%s is not a supported wheel on this platform." %
                    wheel.filename
                )

        install_req.as_egg = self.as_egg
        install_req.use_user_site = self.use_user_site
        install_req.target_dir = self.target_dir
        install_req.pycompile = self.pycompile
        install_req.is_direct = (parent_req_name is None)

        if not name:
            # url or path requirement w/o an egg fragment
            self.unnamed_requirements.append(install_req)
            return [install_req]
        else:
            try:
                existing_req = self.get_requirement(name)
            except KeyError:
                existing_req = None
            if (parent_req_name is None and existing_req and not
                    existing_req.constraint and
                    existing_req.extras == install_req.extras and not
                    existing_req.req.specifier == install_req.req.specifier):
                raise InstallationError(
                    'Double requirement given: %s (already in %s, name=%r)'
                    % (install_req, existing_req, name))
            if not existing_req:
                # Add requirement
                self.requirements[name] = install_req
                # FIXME: what about other normalizations?  E.g., _ vs. -?
                if name.lower() != name:
                    self.requirement_aliases[name.lower()] = name
                result = [install_req]
            else:
                # Assume there's no need to scan, and that we've already
                # encountered this for scanning.
                result = []
                if not install_req.constraint and existing_req.constraint:
                    if (install_req.link and not (existing_req.link and
                       install_req.link.path == existing_req.link.path)):
                        self.reqs_to_cleanup.append(install_req)
                        raise InstallationError(
                            "Could not satisfy constraints for '%s': "
                            "installation from path or url cannot be "
                            "constrained to a version" % name)
                    # If we're now installing a constraint, mark the existing
                    # object for real installation.
                    existing_req.constraint = False
                    existing_req.extras = tuple(
                        sorted(set(existing_req.extras).union(
                               set(install_req.extras))))
                    logger.debug("Setting %s extras to: %s",
                                 existing_req, existing_req.extras)
                    # And now we need to scan this.
                    result = [existing_req]
                # Canonicalise to the already-added object for the backref
                # check below.
                install_req = existing_req
            if parent_req_name:
                parent_req = self.get_requirement(parent_req_name)
                self._dependencies[parent_req].append(install_req)
            return result

    def has_requirement(self, project_name):
        name = project_name.lower()
        if (name in self.requirements and
           not self.requirements[name].constraint or
           name in self.requirement_aliases and
           not self.requirements[self.requirement_aliases[name]].constraint):
            return True
        return False

    @property
    def has_requirements(self):
        return list(req for req in self.requirements.values() if not
                    req.constraint) or self.unnamed_requirements

    @property
    def is_download(self):
        if self.download_dir:
            self.download_dir = expanduser(self.download_dir)
            if os.path.exists(self.download_dir):
                return True
            else:
                logger.critical('Could not find download directory')
                raise InstallationError(
                    "Could not find or access download directory '%s'"
                    % display_path(self.download_dir))
        return False

    def get_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements:
                return self.requirements[name]
            if name in self.requirement_aliases:
                return self.requirements[self.requirement_aliases[name]]
        raise KeyError("No project with the name %r" % project_name)

    def uninstall(self, auto_confirm=False):
        for req in self.requirements.values():
            if req.constraint:
                continue
            req.uninstall(auto_confirm=auto_confirm)
            req.commit_uninstall()

    def prepare_files(self, finder):
        """
        Prepare process. Create temp directories, download and/or unpack files.
        """
        # make the wheelhouse
        if self.wheel_download_dir:
            ensure_dir(self.wheel_download_dir)

        # If any top-level requirement has a hash specified, enter
        # hash-checking mode, which requires hashes from all.
        root_reqs = self.unnamed_requirements + self.requirements.values()
        require_hashes = (self.require_hashes or
                          any(req.has_hash_options for req in root_reqs))
        if require_hashes and self.as_egg:
            raise InstallationError(
                '--egg is not allowed with --require-hashes mode, since it '
                'delegates dependency resolution to setuptools and could thus '
                'result in installation of unhashed packages.')

        # Actually prepare the files, and collect any exceptions. Most hash
        # exceptions cannot be checked ahead of time, because
        # req.populate_link() needs to be called before we can make decisions
        # based on link type.
        discovered_reqs = []
        hash_errors = HashErrors()
        for req in chain(root_reqs, discovered_reqs):
            try:
                discovered_reqs.extend(self._prepare_file(
                    finder,
                    req,
                    require_hashes=require_hashes,
                    ignore_dependencies=self.ignore_dependencies))
            except HashError as exc:
                exc.req = req
                hash_errors.append(exc)

        if hash_errors:
            raise hash_errors

    def _is_upgrade_allowed(self, req):
        return self.upgrade and (
            self.upgrade_strategy == "eager" or (
                self.upgrade_strategy == "only-if-needed" and req.is_direct
            )
        )

    def _check_skip_installed(self, req_to_install, finder):
        """Check if req_to_install should be skipped.

        This will check if the req is installed, and whether we should upgrade
        or reinstall it, taking into account all the relevant user options.

        After calling this req_to_install will only have satisfied_by set to
        None if the req_to_install is to be upgraded/reinstalled etc. Any
        other value will be a dist recording the current thing installed that
        satisfies the requirement.

        Note that for vcs urls and the like we can't assess skipping in this
        routine - we simply identify that we need to pull the thing down,
        then later on it is pulled down and introspected to assess upgrade/
        reinstalls etc.

        :return: A text reason for why it was skipped, or None.
        """
        # Check whether to upgrade/reinstall this req or not.
        req_to_install.check_if_exists()
        if req_to_install.satisfied_by:
            upgrade_allowed = self._is_upgrade_allowed(req_to_install)

            # Is the best version is installed.
            best_installed = False

            if upgrade_allowed:
                # For link based requirements we have to pull the
                # tree down and inspect to assess the version #, so
                # its handled way down.
                if not (self.force_reinstall or req_to_install.link):
                    try:
                        finder.find_requirement(
                            req_to_install, upgrade_allowed)
                    except BestVersionAlreadyInstalled:
                        best_installed = True
                    except DistributionNotFound:
                        # No distribution found, so we squash the
                        # error - it will be raised later when we
                        # re-try later to do the install.
                        # Why don't we just raise here?
                        pass

                if not best_installed:
                    # don't uninstall conflict if user install and
                    # conflict is not user install
                    if not (self.use_user_site and not
                            dist_in_usersite(req_to_install.satisfied_by)):
                        req_to_install.conflicts_with = \
                            req_to_install.satisfied_by
                    req_to_install.satisfied_by = None

            # Figure out a nice message to say why we're skipping this.
            if best_installed:
                skip_reason = 'already up-to-date'
            elif self.upgrade_strategy == "only-if-needed":
                skip_reason = 'not upgraded as not directly required'
            else:
                skip_reason = 'already satisfied'

            return skip_reason
        else:
            return None

    def _prepare_file(self,
                      finder,
                      req_to_install,
                      require_hashes=False,
                      ignore_dependencies=False):
        """Prepare a single requirements file.

        :return: A list of additional InstallRequirements to also install.
        """
        # Tell user what we are doing for this requirement:
        # obtain (editable), skipping, processing (local url), collecting
        # (remote url or package name)
        if req_to_install.constraint or req_to_install.prepared:
            return []

        req_to_install.prepared = True

        # ###################### #
        # # print log messages # #
        # ###################### #
        if req_to_install.editable:
            logger.info('Obtaining %s', req_to_install)
        else:
            # satisfied_by is only evaluated by calling _check_skip_installed,
            # so it must be None here.
            assert req_to_install.satisfied_by is None
            if not self.ignore_installed:
                skip_reason = self._check_skip_installed(
                    req_to_install, finder)

            if req_to_install.satisfied_by:
                assert skip_reason is not None, (
                    '_check_skip_installed returned None but '
                    'req_to_install.satisfied_by is set to %r'
                    % (req_to_install.satisfied_by,))
                logger.info(
                    'Requirement %s: %s', skip_reason,
                    req_to_install)
            else:
                if (req_to_install.link and
                        req_to_install.link.scheme == 'file'):
                    path = url_to_path(req_to_install.link.url)
                    logger.info('Processing %s', display_path(path))
                else:
                    logger.info('Collecting %s', req_to_install)

        with indent_log():
            # ################################ #
            # # vcs update or unpack archive # #
            # ################################ #
            if req_to_install.editable:
                if require_hashes:
                    raise InstallationError(
                        'The editable requirement %s cannot be installed when '
                        'requiring hashes, because there is no single file to '
                        'hash.' % req_to_install)
                req_to_install.ensure_has_source_dir(self.src_dir)
                req_to_install.update_editable(not self.is_download)
                abstract_dist = make_abstract_dist(req_to_install)
                abstract_dist.prep_for_dist()
                if self.is_download:
                    req_to_install.archive(self.download_dir)
                req_to_install.check_if_exists()
            elif req_to_install.satisfied_by:
                if require_hashes:
                    logger.debug(
                        'Since it is already installed, we are trusting this '
                        'package without checking its hash. To ensure a '
                        'completely repeatable environment, install into an '
                        'empty virtualenv.')
                abstract_dist = Installed(req_to_install)
            else:
                # @@ if filesystem packages are not marked
                # editable in a req, a non deterministic error
                # occurs when the script attempts to unpack the
                # build directory
                req_to_install.ensure_has_source_dir(self.build_dir)
                # If a checkout exists, it's unwise to keep going.  version
                # inconsistencies are logged later, but do not fail the
                # installation.
                # FIXME: this won't upgrade when there's an existing
                # package unpacked in `req_to_install.source_dir`
                if os.path.exists(
                        os.path.join(req_to_install.source_dir, 'setup.py')):
                    raise PreviousBuildDirError(
                        "pip can't proceed with requirements '%s' due to a"
                        " pre-existing build directory (%s). This is "
                        "likely due to a previous installation that failed"
                        ". pip is being responsible and not assuming it "
                        "can delete this. Please delete it and try again."
                        % (req_to_install, req_to_install.source_dir)
                    )
                req_to_install.populate_link(
                    finder,
                    self._is_upgrade_allowed(req_to_install),
                    require_hashes
                )
                # We can't hit this spot and have populate_link return None.
                # req_to_install.satisfied_by is None here (because we're
                # guarded) and upgrade has no impact except when satisfied_by
                # is not None.
                # Then inside find_requirement existing_applicable -> False
                # If no new versions are found, DistributionNotFound is raised,
                # otherwise a result is guaranteed.
                assert req_to_install.link
                link = req_to_install.link

                # Now that we have the real link, we can tell what kind of
                # requirements we have and raise some more informative errors
                # than otherwise. (For example, we can raise VcsHashUnsupported
                # for a VCS URL rather than HashMissing.)
                if require_hashes:
                    # We could check these first 2 conditions inside
                    # unpack_url and save repetition of conditions, but then
                    # we would report less-useful error messages for
                    # unhashable requirements, complaining that there's no
                    # hash provided.
                    if is_vcs_url(link):
                        raise VcsHashUnsupported()
                    elif is_file_url(link) and is_dir_url(link):
                        raise DirectoryUrlHashUnsupported()
                    if (not req_to_install.original_link and
                            not req_to_install.is_pinned):
                        # Unpinned packages are asking for trouble when a new
                        # version is uploaded. This isn't a security check, but
                        # it saves users a surprising hash mismatch in the
                        # future.
                        #
                        # file:/// URLs aren't pinnable, so don't complain
                        # about them not being pinned.
                        raise HashUnpinned()
                hashes = req_to_install.hashes(
                    trust_internet=not require_hashes)
                if require_hashes and not hashes:
                    # Known-good hashes are missing for this requirement, so
                    # shim it with a facade object that will provoke hash
                    # computation and then raise a HashMissing exception
                    # showing the user what the hash should be.
                    hashes = MissingHashes()

                try:
                    download_dir = self.download_dir
                    # We always delete unpacked sdists after pip ran.
                    autodelete_unpacked = True
                    if req_to_install.link.is_wheel \
                            and self.wheel_download_dir:
                        # when doing 'pip wheel` we download wheels to a
                        # dedicated dir.
                        download_dir = self.wheel_download_dir
                    if req_to_install.link.is_wheel:
                        if download_dir:
                            # When downloading, we only unpack wheels to get
                            # metadata.
                            autodelete_unpacked = True
                        else:
                            # When installing a wheel, we use the unpacked
                            # wheel.
                            autodelete_unpacked = False
                    unpack_url(
                        req_to_install.link, req_to_install.source_dir,
                        download_dir, autodelete_unpacked,
                        session=self.session, hashes=hashes)
                except requests.HTTPError as exc:
                    logger.critical(
                        'Could not install requirement %s because '
                        'of error %s',
                        req_to_install,
                        exc,
                    )
                    raise InstallationError(
                        'Could not install requirement %s because '
                        'of HTTP error %s for URL %s' %
                        (req_to_install, exc, req_to_install.link)
                    )
                abstract_dist = make_abstract_dist(req_to_install)
                abstract_dist.prep_for_dist()
                if self.is_download:
                    # Make a .zip of the source_dir we already created.
                    if req_to_install.link.scheme in vcs.all_schemes:
                        req_to_install.archive(self.download_dir)
                # req_to_install.req is only avail after unpack for URL
                # pkgs repeat check_if_exists to uninstall-on-upgrade
                # (#14)
                if not self.ignore_installed:
                    req_to_install.check_if_exists()
                if req_to_install.satisfied_by:
                    if self.upgrade or self.ignore_installed:
                        # don't uninstall conflict if user install and
                        # conflict is not user install
                        if not (self.use_user_site and not
                                dist_in_usersite(
                                    req_to_install.satisfied_by)):
                            req_to_install.conflicts_with = \
                                req_to_install.satisfied_by
                        req_to_install.satisfied_by = None
                    else:
                        logger.info(
                            'Requirement already satisfied (use '
                            '--upgrade to upgrade): %s',
                            req_to_install,
                        )

            # ###################### #
            # # parse dependencies # #
            # ###################### #
            dist = abstract_dist.dist(finder)
            try:
                check_dist_requires_python(dist)
            except UnsupportedPythonVersion as e:
                if self.ignore_requires_python:
                    logger.warning(e.args[0])
                else:
                    req_to_install.remove_temporary_source()
                    raise
            more_reqs = []

            def add_req(subreq, extras_requested):
                sub_install_req = InstallRequirement(
                    str(subreq),
                    req_to_install,
                    isolated=self.isolated,
                    wheel_cache=self._wheel_cache,
                )
                more_reqs.extend(self.add_requirement(
                    sub_install_req, req_to_install.name,
                    extras_requested=extras_requested))

            # We add req_to_install before its dependencies, so that we
            # can refer to it when adding dependencies.
            if not self.has_requirement(req_to_install.name):
                # 'unnamed' requirements will get added here
                self.add_requirement(req_to_install, None)

            if not ignore_dependencies:
                if (req_to_install.extras):
                    logger.debug(
                        "Installing extra requirements: %r",
                        ','.join(req_to_install.extras),
                    )
                missing_requested = sorted(
                    set(req_to_install.extras) - set(dist.extras)
                )
                for missing in missing_requested:
                    logger.warning(
                        '%s does not provide the extra \'%s\'',
                        dist, missing
                    )

                available_requested = sorted(
                    set(dist.extras) & set(req_to_install.extras)
                )
                for subreq in dist.requires(available_requested):
                    add_req(subreq, extras_requested=available_requested)

            # cleanup tmp src
            self.reqs_to_cleanup.append(req_to_install)

            if not req_to_install.editable and not req_to_install.satisfied_by:
                # XXX: --no-install leads this to report 'Successfully
                # downloaded' for only non-editable reqs, even though we took
                # action on them.
                self.successfully_downloaded.append(req_to_install)

        return more_reqs

    def cleanup_files(self):
        """Clean up files, remove builds."""
        logger.debug('Cleaning up...')
        with indent_log():
            for req in self.reqs_to_cleanup:
                req.remove_temporary_source()

    def _to_install(self):
        """Create the installation order.

        The installation order is topological - requirements are installed
        before the requiring thing. We break cycles at an arbitrary point,
        and make no other guarantees.
        """
        # The current implementation, which we may change at any point
        # installs the user specified things in the order given, except when
        # dependencies must come earlier to achieve topological order.
        order = []
        ordered_reqs = set()

        def schedule(req):
            if req.satisfied_by or req in ordered_reqs:
                return
            if req.constraint:
                return
            ordered_reqs.add(req)
            for dep in self._dependencies[req]:
                schedule(dep)
            order.append(req)
        for install_req in self.requirements.values():
            schedule(install_req)
        return order

    def install(self, install_options, global_options=(), *args, **kwargs):
        """
        Install everything in this set (after having downloaded and unpacked
        the packages)
        """
        to_install = self._to_install()

        if to_install:
            logger.info(
                'Installing collected packages: %s',
                ', '.join([req.name for req in to_install]),
            )

        with indent_log():
            for requirement in to_install:
                if requirement.conflicts_with:
                    logger.info(
                        'Found existing installation: %s',
                        requirement.conflicts_with,
                    )
                    with indent_log():
                        requirement.uninstall(auto_confirm=True)
                try:
                    requirement.install(
                        install_options,
                        global_options,
                        *args,
                        **kwargs
                    )
                except:
                    # if install did not succeed, rollback previous uninstall
                    if (requirement.conflicts_with and not
                            requirement.install_succeeded):
                        requirement.rollback_uninstall()
                    raise
                else:
                    if (requirement.conflicts_with and
                            requirement.install_succeeded):
                        requirement.commit_uninstall()
                requirement.remove_temporary_source()

        self.successfully_installed = to_install
