from rpy2.robjects import r


def loadfn(fn_name: str):
    with open(fn_name, 'r', 1, 'utf-8') as f:
        fn = r(f.read())
        return fn
