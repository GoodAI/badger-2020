from typing import TypeVar, Callable, Optional

T = TypeVar('T')
R = TypeVar('R')


def apply(o: T, fn: Callable[[T], R]) -> Optional[R]:
    """
    Call function fn to object o if o is not None
    Args:
        o: Optional object
        fn: Function to be applied

    Returns:
        Result of fn(o)
    """
    return None if o is None else fn(o)
