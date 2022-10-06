from typing import Callable
import threading
import time


def MakeTimer(period: float, callback: Callable[[], None]) -> threading.Thread:
    """
    Prepares a detached thread that will call `callback` every `period` seconds.

    * Returns the thread without starting it. Call `thread.start()` to start the timer.
    """

    def work() -> None:
        while True:
            time.sleep(period)
            callback()

    thread = threading.Thread(target=work)
    return thread
