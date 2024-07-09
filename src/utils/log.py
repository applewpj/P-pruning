import logging


def create_logger(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fileHandler = logging.FileHandler(logpath, 'a')
    fileHandler.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s %(message)s")
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger