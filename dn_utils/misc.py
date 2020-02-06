import os

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)
