import sys
import configparser
import torch
import unittest
import importlib


# import config file
conf_path = './../../../conf.ini'
print(conf_path)
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['PARENT_DIR']
ins = sys.path.insert(1, parent_path)
ins if parent_path not in sys.path else 0

submodules = importlib.import_module('vol2.models.submodules')

# test residual_2d_module
res_mod = submodules.residual_2d_module(3)
for mod in res_mod.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.weight.data.fill_(1)
        mod.bias.data.fill_(0)
        mod.running_mean.fill_(0)
        mod.running_var.fill_(1)
        mod.eps = 0

    if isinstance(mod, torch.nn.Conv2d):
        mod.weight.data.fill_(1)
        if mod.bias is not None:
            mod.bias.data.fill_(1)
res_mod.eval()

inp = torch.ones((1, 3, 3, 3), dtype=torch.float32)
inp[:, :, 1, 1] = 0

out = res_mod(inp)

corr = torch.tensor([[190., 262., 190.],
                     [262., 360., 262.],
                     [190., 262., 190.]],
                    dtype=torch.float32)


class residual_2d_module_unit_test(unittest.TestCase):

    def test_residual(self):
        self.assertTrue(torch.equal(corr, out[0, 0]))


if __name__ == '__main__':
    unittest.main()
