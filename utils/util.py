from datetime import datetime, timedelta
from typing import Callable

def measure_time(func: Callable, args: tuple ) -> timedelta:
    """Calculate the runtime of the given function

    Args:
        func (Callable): The function to be tested
        args (tuple): the arguments for the function

    Returns:
        timedelta: The runtime of the function
    """
    start=datetime.now()
    func(*args)
    return (datetime.now()-start)

def timeit(func: Callable, args: tuple , tests_num: int=10) -> float:
    """Calculating the avrage runtime of the given function

    Args:
        func (Callable): The function to mesure
        args (tuple): Arguments for the function
        tests_num (int, optional): How many test to run. Defaults to 10.

    Returns:
        float: The avrage runtime of the function in secs
    """
    
    time = timedelta(0)
    for _ in range(tests_num):
        time += measure_time(func, args)
    return time.total_seconds()  / tests_num
    