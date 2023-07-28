"""Helpers for conditional OpenXLA compilation."""

def if_openxla(then, otherwise = []):
    return select({
        ":with_openxla_runtime": then,
        "//conditions:default": otherwise,
    })

def if_not_openxla(then, otherwise = []):
    return select({
        ":with_openxla_runtime": otherwise,
        "//conditions:default": then,
    })
