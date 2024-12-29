import os

path_to_tests = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(path_to_tests, 'data')
path_to_stata = r'/Applications/Stata/'

# intialise stata API
import stata_setup
stata_setup.config(path_to_stata, edition='mp', splash=False)
from pystata import config
config.init('mp')
from pystata import stata
