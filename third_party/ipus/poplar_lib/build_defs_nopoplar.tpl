# Macros for supporting Poplar library

def poplar_available():
    """Returns false because Poplar library was not configured
    """
    return False

def poplar_lib_directory():
    """Returns the full path to the Poplar libraries directory
    """
    return ""

def poplibs_lib_directory():
    """Returns the full path to the Poplibs libraries directory
    """
    return ""

def tf_poplar_build_tag():
    """Returns a build tag/hash for displaying along with the Poplar version
    """
    return ""
