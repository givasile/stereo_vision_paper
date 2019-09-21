import sys
import configparser
import importlib
import torch
import timeit
import time
import os


cnn_name: str = 'merging_info_net_custom_features'

# import config file
conf_path = './../../../conf.ini'
conf = configparser.ConfigParser()
conf.read(conf_path)

# add parent path, if not already added
parent_path = conf['PATHS']['parent_dir']
sys.path.insert(1, parent_path) if parent_path not in sys.path else 0


# import custom modules
# submodules = importlib.import_module('vol2.models.' + cnn_name + '.submodules')
net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
merged = importlib.import_module('raw_dataset.merged_dataset')
preprocess = importlib.import_module('preprocess')
evaluate = importlib.import_module('evaluate')
visualize = importlib.import_module('visualize')
utils = importlib.import_module('vol2.models.utils')

####################################################
################# Configuration ####################
####################################################

# directory to load_from
experiment_n = 1      # directory to load_from
checkpoint_n = 15                # which checkpoint to load weights/stats from

which = 'train'
form = 'full_im'
limit_maxD = True
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 100
device = 'cuda'

####################################################
# exec init operations and define global variables #
####################################################

# restore dataset
if get_common_dataset:
    merged_dataset = utils.load_common_dataset(common_dataset_name, conf['PATHS']['common_datasets'])
else:
    merged_dataset = utils.load_specific_dataset(os.path.join(conf['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n)))

# create model
if device == 'cpu':
    model_instance = net.model()
else:
    model_instance = net.model().cuda()

# restore weights
checkpoint = torch.load(os.path.join(conf['PATHS']['saved_models'], 'vol2', cnn_name, 'experiment_' + str(experiment_n), 'checkpoint_' + str(checkpoint_n) + '.tar'))
model_instance.load_state_dict(checkpoint['state_dict'])

# restore stats
stats = checkpoint['stats']

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model_instance.parameters()])))


# prepare input
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.type(torch.bool).unsqueeze(0).cuda()

# model settings
max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/4), round(h/4), round(w/4)],
          [round(max_disp/8), round(h/8), round(w/8)],
          [round(max_disp/16), round(h/16), round(w/16)],
          [round(max_disp/32), round(h/32), round(w/32)]]

prediction_from_scales = {0: ['after'],
                          1: ['after'],
                          2: ['after'],
                          3: ['after']}
# scales = [[round(max_disp/4), round(h/4), round(w/4)]]
# prediction_from_scales = {0: ['after']}

tmp = timeit.default_timer()
mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)
print("Inspection execution time: %s" % (timeit.default_timer()-tmp))

# visualize.imshow_3ch(imL.squeeze(0).cpu(), 'imL')
# visualize.imshow_3ch(imR.squeeze(0).cpu(), 'imR')
#
for i, im in enumerate(predictions_dict.values()):
    time.sleep(1)
    visualize.imshow_1ch(im['after'][0].cpu(), str(i), [0, dispL.max()])
visualize.imshow_1ch(dispL[0].cpu(), 'dispL')
image_pcg = evaluate.image_percentage_over_limit(predictions_dict[0]['after'].cpu(),
                                                 dispL.cpu(), maskL.cpu(), 3)
visualize.imshow_1ch(image_pcg[0], 'over_thres')
