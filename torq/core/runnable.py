from ..utils import logging


class Runnable:
    def run(self, iters: int = -1, *args):
        try:
            if iters < 0:
                while True:
                    self(*args)
            else:
                for _ in range(iters):
                    self(*args)

        except KeyboardInterrupt:
            logging.info("Running system has been terminated")
