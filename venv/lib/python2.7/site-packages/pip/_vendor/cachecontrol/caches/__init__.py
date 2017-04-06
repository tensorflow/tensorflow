from textwrap import dedent

try:
    from .file_cache import FileCache
except ImportError:
    notice = dedent('''
    NOTE: In order to use the FileCache you must have
    lockfile installed. You can install it via pip:
      pip install lockfile
    ''')
    print(notice)


try:
    import redis
    from .redis_cache import RedisCache
except ImportError:
    pass
