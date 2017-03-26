from .adapter import CacheControlAdapter
from .cache import DictCache


def CacheControl(sess,
                 cache=None,
                 cache_etags=True,
                 serializer=None,
                 heuristic=None):

    cache = cache or DictCache()
    adapter = CacheControlAdapter(
        cache,
        cache_etags=cache_etags,
        serializer=serializer,
        heuristic=heuristic,
    )
    sess.mount('http://', adapter)
    sess.mount('https://', adapter)

    return sess
