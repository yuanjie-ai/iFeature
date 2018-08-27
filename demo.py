

from ifeature.utils.decorator import execution_time



@execution_time
def f(x):
    return x

f(11111111)