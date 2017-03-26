try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin


try:
    import cPickle as pickle
except ImportError:
    import pickle


from pip._vendor.requests.packages.urllib3.response import HTTPResponse
from pip._vendor.requests.packages.urllib3.util import is_fp_closed

# Replicate some six behaviour
try:
    text_type = (unicode,)
except NameError:
    text_type = (str,)
