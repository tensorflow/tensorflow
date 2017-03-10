

def check_requirements(installed_dists):
    missing_reqs_dict = {}
    incompatible_reqs_dict = {}

    for dist in installed_dists:
        key = '%s==%s' % (dist.project_name, dist.version)

        missing_reqs = list(get_missing_reqs(dist, installed_dists))
        if missing_reqs:
            missing_reqs_dict[key] = missing_reqs

        incompatible_reqs = list(get_incompatible_reqs(
            dist, installed_dists))
        if incompatible_reqs:
            incompatible_reqs_dict[key] = incompatible_reqs

    return (missing_reqs_dict, incompatible_reqs_dict)


def get_missing_reqs(dist, installed_dists):
    """Return all of the requirements of `dist` that aren't present in
    `installed_dists`.

    """
    installed_names = set(d.project_name.lower() for d in installed_dists)
    missing_requirements = set()

    for requirement in dist.requires():
        if requirement.project_name.lower() not in installed_names:
            missing_requirements.add(requirement)
            yield requirement


def get_incompatible_reqs(dist, installed_dists):
    """Return all of the requirements of `dist` that are present in
    `installed_dists`, but have incompatible versions.

    """
    installed_dists_by_name = {}
    for installed_dist in installed_dists:
        installed_dists_by_name[installed_dist.project_name] = installed_dist

    for requirement in dist.requires():
        present_dist = installed_dists_by_name.get(requirement.project_name)

        if present_dist and present_dist not in requirement:
            yield (requirement, present_dist)
