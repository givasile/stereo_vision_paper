import sys
import configparser
import importlib
import timeit

# configuration
cnn_name: str = 'merging_info_net_custom_features_free_2d_3d_weights'

experiment_n = 21
checkpoint_n = 10
which = 'test'
form = 'full_im'
limit_maxD = True
get_common_dataset = True
common_dataset_name = 'freiburg_tr_te'
example_num = 100
device = 'cuda'


def run(conf_file):
    utils = importlib.import_module('vol2.models.utils')
    net = importlib.import_module('vol2.models.' + cnn_name + '.' + cnn_name)

    # load everything needed
    imL, imR, dispL, maskL, model_instance = utils.load_things_for_inspection(
        cnn_name, conf_file, checkpoint_n, experiment_n, get_common_dataset,
        common_dataset_name, device, which, form, limit_maxD, example_num)

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

# visualize.imshow_3ch(imL.squeeze(0).cpu(), 'imL')
# visualize.imshow_3ch(imR.squeeze(0).cpu(), 'imR')

# for i, im in enumerate(predictions_dict.values()):
#     time.sleep(1)
#     visualize.imshow_1ch(im['after'][0].cpu(), str(i), [0, dispL.max()])
# visualize.imshow_1ch(dispL[0].cpu(), 'dispL')
# image_pcg = evaluate.image_percentage_over_limit(predictions_dict[0]['after'].cpu(),
#                                                  dispL.cpu(), maskL.cpu(), 3)
# visualize.imshow_1ch(image_pcg[0], 'over_thres')


if __name__ == '__main__':
    # import config file
    conf_path = './../../../conf.ini'
    conf = configparser.ConfigParser()
    conf.read(conf_path)

    # add parent path, if not already added
    parent_path = conf['PATHS']['parent_dir']
    sys.path.insert(1, parent_path) if parent_path not in sys.path else 0

    # run main
    run(conf)