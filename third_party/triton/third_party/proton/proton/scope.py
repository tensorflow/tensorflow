import threading
import time
from functools import wraps
from typing import Optional, Union

from .flags import get_profiling_on
from triton._C.libproton import proton as libproton

thread_local_scopes = threading.local()

MetricValueType = Union[float, int]


class scope:
    """
    A context manager and decorator for entering and exiting a scope.

    Usage:
        context manager:
        ```python
        with proton.scope("test0", {metric_name: metric_value}):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.scope("test0", {metric_name: metric_value})
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the scope.
        metrics (dict[str, float], optional): The metrics of the scope. Default is None.
    """

    def __init__(self, name: str, metrics: Optional[dict[str, MetricValueType]] = None) -> None:
        self.name = name
        self.metrics = metrics
        self.id = None

    def _enter_scope(self):
        if not get_profiling_on():
            return
        self.id = libproton.record_scope()
        libproton.enter_scope(self.id, self.name)
        if self.metrics:
            libproton.add_metrics(self.id, self.metrics)

    def _exit_scope(self):
        if not get_profiling_on() or self.id is None:
            return
        libproton.exit_scope(self.id, self.name)

    def __enter__(self):
        self._enter_scope()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_scope()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            self._enter_scope()
            try:
                return func(*args, **kwargs)
            finally:
                self._exit_scope()

        return wrapper


class cpu_timed_scope(scope):
    """
    A scope that measures elapsed time (cpu_time).

    Args:
        name (str): The name of the scope.
        metrics (dict[str, float], optional): Additional metrics to add. Default is None.
    """

    def __init__(self, name: str, metrics: Optional[dict[str, float]] = None) -> None:
        super().__init__(name, metrics)
        self.start_time = None
        if metrics and "cpu_time" in metrics:
            raise ValueError("The metric name 'cpu_time' is reserved.")

    def _enter_scope(self):
        if not get_profiling_on():
            return
        self.start_time = time.time_ns()
        super()._enter_scope()

    def _exit_scope(self):
        if not get_profiling_on():
            return
        super()._exit_scope()
        if self.start_time is not None:
            cpu_time = time.time_ns() - self.start_time
            libproton.add_metrics(self.id, {"cpu_time (ns)(exc)": cpu_time})


def enter_scope(name: str, *, triton_op: bool = False, metrics: Optional[dict[str, MetricValueType]] = None) -> int:
    if not get_profiling_on():
        return -1
    id = libproton.record_scope()
    thread_local_scopes.scopes = getattr(thread_local_scopes, "scopes", [])
    thread_local_scopes.scopes.append((id, name))
    if triton_op:
        libproton.enter_op(id, name)
    else:
        libproton.enter_scope(id, name)
    if metrics:
        libproton.add_metrics(id, metrics)
    return id


def exit_scope(triton_op: bool = False, metrics: Optional[dict[str, MetricValueType]] = None) -> int:
    if not get_profiling_on():
        return -1
    id, name = thread_local_scopes.scopes.pop()
    if triton_op:
        libproton.exit_op(id, name)
    else:
        libproton.exit_scope(id, name)
    if metrics:
        libproton.add_metrics(id, metrics)
    return id
