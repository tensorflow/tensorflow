from triton.language import core as tl
from triton.language.core import builtin
import warnings


@builtin
def record(isStart: bool, regionId: int, _builder=None):
    warnings.warn(
        "\nWarning the proton language module within Proton contains under development features that are not intended to be used outside of the core development team"
    )
    return tl.tensor(_builder.create_proton_record(isStart, regionId), tl.void)
