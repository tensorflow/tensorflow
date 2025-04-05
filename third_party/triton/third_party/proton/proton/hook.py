from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope
from triton.compiler import CompiledKernel, LazyDict

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        metadata = lazy_dict.get()
        exit_state()
        fn_metrics = {k: metadata[k] for k in TritonHook.metrics if k in metadata}
        enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        exit_scope(triton_op=True)


def register_triton_hook() -> None:
    if CompiledKernel.launch_enter_hook is None:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit


def unregister_triton_hook() -> None:
    if CompiledKernel.launch_enter_hook == TritonHook.enter:
        CompiledKernel.launch_enter_hook = None
        CompiledKernel.launch_exit_hook = None
