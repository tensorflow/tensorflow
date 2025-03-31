from triton._C.libproton import proton as libproton
from .flags import get_profiling_on
from functools import wraps


class state:
    """
    A context manager and decorator for entering and exiting a state.

    Usage:
        context manager:
        ```python
        with proton.state("test0"):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.state("test0")
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the state.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if not get_profiling_on():
            return self
        libproton.enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not get_profiling_on():
            return
        libproton.exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_profiling_on():
                libproton.enter_state(self.name)
            ret = func(*args, **kwargs)
            if get_profiling_on():
                libproton.exit_state()
            return ret

        return wrapper


def enter_state(name: str) -> None:
    libproton.enter_state(name)


def exit_state() -> None:
    libproton.exit_state()
