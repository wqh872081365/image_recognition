# -*- coding: utf-8 -*-

import re
import cProfile
import pstats
import os

# cProfile.run('re.compile("foo|bar")')


def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            # DO_PROF = os.getenv("PROFILING")
            DO_PROF = True
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper


@do_cprofile("./mkm_run.prof")
def run(**kwargs):
    re.compile("foo|bar")


def main():
    run()


if __name__ == '__main__':
    main()