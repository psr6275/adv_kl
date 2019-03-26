import os,sys,time,math
import errno

def mkdir_p(path):
    '''make dir if not exist'''

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno ==errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise