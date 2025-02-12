"""Functions common across configure rules."""

BAZEL_SH = "BAZEL_SH"
PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
PYTHON_LIB_PATH = "PYTHON_LIB_PATH"
TF_PYTHON_CONFIG_REPO = "TF_PYTHON_CONFIG_REPO"

def auto_config_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sConfiguration Error:%s %s\n" % (red, no_color, msg))

def which(repository_ctx, program_name, allow_failure = False):
    """Returns the full path to a program on the execution platform.

    Args:
      repository_ctx: the repository_ctx
      program_name: name of the program on the PATH

    Returns:
      The full path to a program on the execution platform.
    """
    if is_windows(repository_ctx):
        if not program_name.endswith(".exe"):
            program_name = program_name + ".exe"
        out = execute(
            repository_ctx,
            ["C:\\Windows\\System32\\where.exe", program_name],
            allow_failure = allow_failure,
        ).stdout
        if out != None:
            out = out.replace("\\", "\\\\").rstrip()
        return out

    out = execute(
        repository_ctx,
        ["which", program_name],
        allow_failure = allow_failure,
    ).stdout
    if out != None:
        out = out.replace("\\", "\\\\").rstrip()
    return out

def get_python_bin(repository_ctx):
    """Gets the python bin path.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      The python bin path.
    """
    python_bin = get_host_environ(repository_ctx, PYTHON_BIN_PATH)
    if python_bin:
        return python_bin

    # First check for an explicit "python3"
    python_bin = which(repository_ctx, "python3", True)
    if python_bin:
        return python_bin

    # Some systems just call pythone3 "python"
    python_bin = which(repository_ctx, "python", True)
    if python_bin:
        return python_bin

    auto_config_fail("Cannot find python in PATH, please make sure " +
                     "python is installed and add its directory in PATH, or --define " +
                     "%s='/something/else'.\nPATH=%s" % (
                         PYTHON_BIN_PATH,
                         get_environ(repository_ctx, "PATH"),
                     ))
    return python_bin  # unreachable

def get_bash_bin(repository_ctx):
    """Gets the bash bin path.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      The bash bin path.
    """
    bash_bin = get_host_environ(repository_ctx, BAZEL_SH)
    if bash_bin != None:
        return bash_bin
    bash_bin_path = which(repository_ctx, "bash")
    if bash_bin_path == None:
        auto_config_fail("Cannot find bash in PATH, please make sure " +
                         "bash is installed and add its directory in PATH, or --define " +
                         "%s='/path/to/bash'.\nPATH=%s" % (
                             BAZEL_SH,
                             get_environ(repository_ctx, "PATH"),
                         ))
    return bash_bin_path

def read_dir(repository_ctx, src_dir):
    """Returns a sorted list with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks.

    Args:
      repository_ctx: the repository_ctx
      src_dir: the directory to traverse

    Returns:
      A sorted list with all files in a directory.
    """
    if is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = execute(
            repository_ctx,
            ["C:\\Windows\\System32\\cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            allow_failure = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            allow_failure = True,
        )
        result = find_result.stdout
    return sorted(result.splitlines())

def get_environ(repository_ctx, name, default_value = None):
    """Returns the value of an environment variable on the execution platform.

    Args:
      repository_ctx: the repository_ctx
      name: the name of environment variable
      default_value: the value to return if not set

    Returns:
      The value of the environment variable 'name' on the execution platform
      or 'default_value' if it's not set.
    """
    if is_windows(repository_ctx):
        result = execute(
            repository_ctx,
            ["C:\\Windows\\System32\\cmd.exe", "/c", "echo", "%" + name + "%"],
            allow_failure = True,
        )
    else:
        cmd = "echo -n \"$%s\"" % name
        result = execute(
            repository_ctx,
            [get_bash_bin(repository_ctx), "-c", cmd],
            allow_failure = True,
        )
    if len(result.stdout) == 0:
        return default_value
    return result.stdout

def get_host_environ(repository_ctx, name, default_value = None):
    """Returns the value of an environment variable on the host platform.

    The host platform is the machine that Bazel runs on.

    Args:
      repository_ctx: the repository_ctx
      name: the name of environment variable

    Returns:
      The value of the environment variable 'name' on the host platform.
    """
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ.get(name).strip()

    if hasattr(repository_ctx.attr, "environ") and name in repository_ctx.attr.environ:
        return repository_ctx.attr.environ.get(name).strip()

    return default_value

def is_windows(repository_ctx):
    """Returns true if the execution platform is Windows.

    Args:
      repository_ctx: the repository_ctx

    Returns:
      If the execution platform is Windows.
    """
    os_name = ""
    if hasattr(repository_ctx.attr, "exec_properties") and "OSFamily" in repository_ctx.attr.exec_properties:
        os_name = repository_ctx.attr.exec_properties["OSFamily"]
    else:
        os_name = repository_ctx.os.name

    return os_name.lower().find("windows") != -1

def get_cpu_value(repository_ctx):
    """Returns the name of the host operating system.

    Args:
      repository_ctx: The repository context.
    Returns:
      A string containing the name of the host operating system.
    """
    if is_windows(repository_ctx):
        return "Windows"
    result = raw_exec(repository_ctx, ["uname", "-s"])
    return result.stdout.strip()

def execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        allow_failure = False,
        env_vars = {}):
    """Executes an arbitrary shell command.

    Args:
      repository_ctx: the repository_ctx object
      cmdline: list of strings, the command to execute
      error_msg: string, a summary of the error if the command fails
      error_details: string, details about the error or steps to fix it
      allow_failure: bool, if True, an empty stdout result or output to stderr
        is fine, otherwise either of these is an error
      env_vars: environment variables
    Returns:
      The result of repository_ctx.execute(cmdline)
    """
    result = raw_exec(repository_ctx, cmdline, env_vars)
    if (result.stderr or not result.stdout) and not allow_failure:
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def raw_exec(repository_ctx, cmdline, env_vars = {}):
    """Executes a command via repository_ctx.execute() and returns the result.

    This method is useful for debugging purposes. For example, to print all
    commands executed as well as their return code.

    Args:
      repository_ctx: the repository_ctx
      cmdline: the list of args
      env_vars: environment variables

    Returns:
      The 'exec_result' of repository_ctx.execute().
    """
    return repository_ctx.execute(cmdline, environment = env_vars)

def files_exist(repository_ctx, paths, bash_bin = None):
    """Checks which files in paths exists.

    Args:
      repository_ctx: the repository_ctx
      paths: a list of paths
      bash_bin: path to the bash interpreter

    Returns:
      Returns a list of Bool. True means that the path at the
      same position in the paths list exists.
    """
    if bash_bin == None:
        bash_bin = get_bash_bin(repository_ctx)

    cmd_tpl = "[ -e \"%s\" ] && echo True || echo False"
    cmds = [cmd_tpl % path for path in paths]
    cmd = " ; ".join(cmds)

    stdout = execute(repository_ctx, [bash_bin, "-c", cmd]).stdout.strip()
    return [val == "True" for val in stdout.splitlines()]

def realpath(repository_ctx, path, bash_bin = None):
    """Returns the result of "realpath path".

    Args:
      repository_ctx: the repository_ctx
      path: a path on the file system
      bash_bin: path to the bash interpreter

    Returns:
      Returns the result of "realpath path"
    """
    if bash_bin == None:
        bash_bin = get_bash_bin(repository_ctx)

    return execute(repository_ctx, [bash_bin, "-c", "realpath \"%s\"" % path]).stdout.strip()

def relative_to(repository_ctx, base, path, bash_bin = None):
    """Returns the result of "realpath --relative-to".

    Args:
      repository_ctx: the repository_ctx
      base: a path on the file system
      path: a path on the file system
      bash_bin: path to the bash interpreter

    Returns:
      Returns the result of "realpath --relative-to"
    """
    if bash_bin == None:
        bash_bin = get_bash_bin(repository_ctx)

    return execute(repository_ctx, [bash_bin, "-c", "realpath --relative-to \"%s\" \"%s\"" % (base, path)]).stdout.strip()

def err_out(result):
    """Returns stderr if set, else stdout.

    This function is a workaround for a bug in RBE where stderr is returned as stdout. Instead
    of using result.stderr use err_out(result) instead.

    Args:
      result: the exec_result.

    Returns:
      The stderr if set, else stdout
    """
    if len(result.stderr) == 0:
        return result.stdout
    return result.stderr

def config_repo_label(config_repo, target):
    """Construct a label from config_repo and target.

    This function exists to ease the migration from preconfig to remote config. In preconfig
    the TF_*_CONFIG_REPO environ variables are set to packages in the main repo while in
    remote config they will point to remote repositories.

    Args:
      config_repo: a remote repository or package.
      target: a target
    Returns:
      A label constructed from config_repo and target.
    """
    if config_repo.startswith("@") and not config_repo.find("//") > 0:
        # remote config is being used.
        return Label(config_repo + "//" + target)
    elif target.startswith(":"):
        return Label(config_repo + target)
    else:
        return Label(config_repo + "/" + target)
