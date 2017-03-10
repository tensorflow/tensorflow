import logging

from pip.basecommand import Command
from pip.operations.check import check_requirements
from pip.utils import get_installed_distributions


logger = logging.getLogger(__name__)


class CheckCommand(Command):
    """Verify installed packages have compatible dependencies."""
    name = 'check'
    usage = """
      %prog [options]"""
    summary = 'Verify installed packages have compatible dependencies.'

    def run(self, options, args):
        dists = get_installed_distributions(local_only=False, skip=())
        missing_reqs_dict, incompatible_reqs_dict = check_requirements(dists)

        for dist in dists:
            key = '%s==%s' % (dist.project_name, dist.version)

            for requirement in missing_reqs_dict.get(key, []):
                logger.info(
                    "%s %s requires %s, which is not installed.",
                    dist.project_name, dist.version, requirement.project_name)

            for requirement, actual in incompatible_reqs_dict.get(key, []):
                logger.info(
                    "%s %s has requirement %s, but you have %s %s.",
                    dist.project_name, dist.version, requirement,
                    actual.project_name, actual.version)

        if missing_reqs_dict or incompatible_reqs_dict:
            return 1
        else:
            logger.info("No broken requirements found.")
