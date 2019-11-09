import sys
import configparser
import importlib
import torch
import timeit
import PIL
import os

cnn_name = 'merging_info_net_custom_features_free_2d_3d_weights'

# import config file
conf_path = './../conf.ini'
conf_file = configparser.ConfigParser()
conf_file.read(conf_path)

# add parent path, if not already added
parent_path = conf_file['PATHS']['PARENT_DIR']
sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

# import custom modulesn
utils = importlib.import_module('vol2.models.utils')
net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)
preprocess = importlib.import_module('preprocess')

# directory to load_from
experiment_n = 1
checkpoint_n = 10
which = 'test'
form = 'full_im'
limit_maxD = True
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 150
device = 'cuda'

# create helpful paths
experiment_directory = os.path.join(conf_file['PATHS']['saved_models'], 'vol2', cnn_name,
                                    'experiment_' + str(experiment_n))
checkpoint_filepath = os.path.join(experiment_directory, 'checkpoint_' + str(checkpoint_n) + '.tar')

# load dataset
if get_common_dataset:
    merged_dataset = utils.load_common_dataset(common_dataset_name, conf_file['PATHS']['common_datasets'])
else:
    merged_dataset = utils.load_specific_dataset(experiment_directory)

# create model
if device == 'cpu':
    model_instance = net.model()
else:
    model_instance = net.model().cuda()

# restore weights
checkpoint = torch.load(checkpoint_filepath)
model_instance.load_state_dict(checkpoint['state_dict'])

# restore training statistics
stats = checkpoint['stats']

# input
data_feeder = preprocess.dataset(merged_dataset, which, form, limit_maxD)
imL, imR, dispL, maskL = data_feeder[example_num]
imL = imL.unsqueeze(0).cuda()
imR = imR.unsqueeze(0).cuda()
max_limit = dispL.max()
dispL = dispL.unsqueeze(0).cuda()
maskL = maskL.unsqueeze(0).cuda()

# inspection
max_disp = 192
h = imL.shape[2]
w = imL.shape[3]
initial_scale = [max_disp, h, w]
scales = [[round(max_disp/4), round(h/4), round(w/4)],
          [round(max_disp/8), round(h/8), round(w/8)],
          [round(max_disp/16), round(h/16), round(w/16)],
          [round(max_disp/32), round(h/32), round(w/32)]]

prediction_from_scales = {3: ['after'],
                          2: ['after'],
                          1: ['after'],
                          0: ['after']}


tmp = timeit.default_timer()
mae, err_pcg, imL_d, imR_d, volumes, volumes_dict, for_out_dict, predictions_dict = net.inspection(
    model_instance, initial_scale, scales, prediction_from_scales, device, imL, imR, dispL, maskL)
print("Inspection execution time: %s" % (timeit.default_timer()-tmp))


# visualize
imL = merged_dataset.load(example_num, which).imL_rgb
imR = merged_dataset.load(example_num, which).imR_rgb

imL_0 = imL.resize((scales[0][2], scales[0][1]), PIL.Image.BILINEAR)
imR_0 = imR.resize((scales[0][2], scales[0][1]), PIL.Image.BILINEAR)

imL_1 = imL.resize((scales[1][2], scales[1][1]), PIL.Image.BILINEAR)
imR_1 = imR.resize((scales[1][2], scales[1][1]), PIL.Image.BILINEAR)

imL_2 = imL.resize((scales[2][2], scales[2][1]), PIL.Image.BILINEAR)
imR_2 = imR.resize((scales[2][2], scales[2][1]), PIL.Image.BILINEAR)

imL_3 = imL.resize((scales[3][2], scales[3][1]), PIL.Image.BILINEAR)
imR_3 = imR.resize((scales[3][2], scales[3][1]), PIL.Image.BILINEAR)


pred_0 = predictions_dict[0]['after'].cpu().numpy()[0]
# save
base_path = "./latex/figures/"
with open(base_path + "block_diagram_imL.png", 'wb') as fm:
    imL.save(fm)
with open(base_path + "block_diagram_imR.png", 'wb') as fm:
    imR.save(fm)
with open(base_path + "block_diagram_imL_0.png", 'wb') as fm:
    imL_0.save(fm)
with open(base_path + "block_diagram_imR_0.png", 'wb') as fm:
    imR_0.save(fm)
with open(base_path + "block_diagram_imL_1.png", 'wb') as fm:
    imL_1.save(fm)
with open(base_path + "block_diagram_imR_1.png", 'wb') as fm:
    imR_1.save(fm)
with open(base_path + "block_diagram_imL_2.png", 'wb') as fm:
    imL_2.save(fm)
with open(base_path + "block_diagram_imR_2.png", 'wb') as fm:
    imR_2.save(fm)
with open(base_path + "block_diagram_imL_3.png", 'wb') as fm:
    imL_3.save(fm)
with open(base_path + "block_diagram_imR_3.png", 'wb') as fm:
    imR_3.save(fm)
