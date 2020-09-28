import os




def normabspath(basedir: str, filename: str):
    return os.path.normpath(os.path.join(basedir, filename))



