import pickle
import os


def extract_state(file):
    with open(file, 'rb') as fm:
        dataset = pickle.load(fm)

    split_dat = ['freiburg_flying_split', 'freiburg_monkaa_split', 'freiburg_driving_split', 'kitti_2012_split', 'kitti_2015_split']
    dic = {}
    for dat_name in split_dat:
        dat_inst = getattr(dataset, dat_name)
        dic[dat_name] = {}
        dic[dat_name]['split'] = dat_inst.split
        dic[dat_name]['train'] = dat_inst.train
        dic[dat_name]['val'] = dat_inst.val
        dic[dat_name]['test'] = dat_inst.test

    with open(file.split('.pickle')[0] + '_state_dict.pickle', 'wb') as fm:
        pickle.dump(dic, fm)


base = '/home/givaisile/stereo_vision/saved_models/vol2/'
for dir1 in os.listdir(base):
    for dir2 in os.listdir(os.path.join(base, dir1)):
        print(os.path.join(base, dir1, dir2, 'merged_dataset.pickle'))
        extract_state(os.path.join(base, dir1, dir2, 'merged_dataset.pickle'))
# print(os.listdir(base))
# extract_state(file)
