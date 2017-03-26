def __boot():
    import sys
    import os
    PYTHONPATH = os.environ.get('PYTHONPATH')
    if PYTHONPATH is None or (sys.platform == 'win32' and not PYTHONPATH):
        PYTHONPATH = []
    else:
        PYTHONPATH = PYTHONPATH.split(os.pathsep)

    pic = getattr(sys, 'path_importer_cache', {})
    stdpath = sys.path[len(PYTHONPATH):]
    mydir = os.path.dirname(__file__)

    for item in stdpath:
        if item == mydir or not item:
            continue  # skip if current dir. on Windows, or my own directory
        importer = pic.get(item)
        if importer is not None:
            loader = importer.find_module('site')
            if loader is not None:
                # This should actually reload the current module
                loader.load_module('site')
                break
        else:
            try:
                import imp  # Avoid import loop in Python >= 3.3
                stream, path, descr = imp.find_module('site', [item])
            except ImportError:
                continue
            if stream is None:
                continue
            try:
                # This should actually reload the current module
                imp.load_module('site', stream, path, descr)
            finally:
                stream.close()
            break
    else:
        raise ImportError("Couldn't find the real 'site' module")

    known_paths = dict([(makepath(item)[1], 1) for item in sys.path])  # 2.2 comp

    oldpos = getattr(sys, '__egginsert', 0)  # save old insertion position
    sys.__egginsert = 0  # and reset the current one

    for item in PYTHONPATH:
        addsitedir(item)

    sys.__egginsert += oldpos  # restore effective old position

    d, nd = makepath(stdpath[0])
    insert_at = None
    new_path = []

    for item in sys.path:
        p, np = makepath(item)

        if np == nd and insert_at is None:
            # We've hit the first 'system' path entry, so added entries go here
            insert_at = len(new_path)

        if np in known_paths or insert_at is None:
            new_path.append(item)
        else:
            # new path after the insert point, back-insert it
            new_path.insert(insert_at, item)
            insert_at += 1

    sys.path[:] = new_path


if __name__ == 'site':
    __boot()
    del __boot
