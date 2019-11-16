import logging

EPSILON = 1e-8


def config_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
