from .models import BayesGLM

__version__ = '0.0+'

# set up logging (note if a handler has already been set then this won't do anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    datefmt='[ %H:%M:%S ]')


