from __future__ import annotations

from pip._vendor.platformdirs import PlatformDirs, __version__

PROPS = (
    "user_data_dir",
    "user_config_dir",
    "user_cache_dir",
    "user_state_dir",
    "user_log_dir",
    "user_documents_dir",
    "user_runtime_dir",
    "site_data_dir",
    "site_config_dir",
)


def main() -> None:
    app_name = "MyApp"
    app_author = "MyCompany"

    print(f"-- platformdirs {__version__} --")

    print("-- app dirs (with optional 'version')")
    dirs = PlatformDirs(app_name, app_author, version="1.0")
    for prop in PROPS:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'version')")
    dirs = PlatformDirs(app_name, app_author)
    for prop in PROPS:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (without optional 'appauthor')")
    dirs = PlatformDirs(app_name)
    for prop in PROPS:
        print(f"{prop}: {getattr(dirs, prop)}")

    print("\n-- app dirs (with disabled 'appauthor')")
    dirs = PlatformDirs(app_name, appauthor=False)
    for prop in PROPS:
        print(f"{prop}: {getattr(dirs, prop)}")


if __name__ == "__main__":
    main()
