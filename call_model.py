import sys
import configparser
import importlib


model_name = 'merging_info_net_custom_features'
action = 'inspection'

conf = configparser.ConfigParser()
conf.read('./conf.ini')

parent_path = conf['PATHS']['parent_dir']
sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

exe_module = importlib.import_module('vol2.models.' + model_name + '.' + action)

exe_module.run(conf)
