import logging
import logging.handlers
import os


# print(os.getcwd())
# logging.basicConfig(filename=os.path.join(os.getcwd(), 'test.log'))
# logging.info('hello')
# logging.warning('hello')

logger = logging.getLogger(__name__)
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename=os.path.join(os.getcwd(), 'test.log'), mode='w')

logger.setLevel(logging.INFO)
# handler1.setLevel(logging.WARNING)
# handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)

# 分别为 10、30、30
print(handler1.level)
print(handler2.level)
print(logger.level)

logger.debug('This is a customer debug message')
logger.info('This is an customer info message')
logger.warning('This is a customer warning message')
logger.error('This is an customer error message')
logger.critical('This is a customer critical message')
