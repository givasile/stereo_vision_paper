from abc import ABC, abstractmethod
import PIL
from PIL import Image
from typing import List, Dict, Union
import raw_dataset.utils as utils


class Registry(ABC):

    def __init__(self, index: int, identity: str,
                 imL_g: PIL.Image, imR_g: PIL.Image,
                 imL_rgb: PIL.Image, imR_rgb: PIL.Image,
                 gtL: PIL.Image, gtR: PIL.Image,
                 maskL: PIL.Image, maskR: PIL.Image,
                 maxDL: PIL.Image, maxDR: PIL.Image,
                 dataset: str) -> None:
        self.index = index
        self.id = identity
        self.imL_g = imL_g
        self.imR_g = imR_g
        self.imL_rgb = imL_rgb
        self.imR_rgb = imR_rgb
        self.gtL = gtL
        self.gtR = gtR
        self.maskL = maskL
        self.maskR = maskR
        self.maxDL = maxDL
        self.maxDR = maxDR
        self.dataset = dataset
        super().__init__()

    def plot(self, image: str) -> None:
        assert image in ['imL_g', 'imR_g', 'imL_rgb', 'imR_rgb', 'gtL', 'gtR', 'maskL', 'maskR']
        if image in ['imL_g', 'imR_g', 'imL_rgb', 'imR_rgb', 'maskL', 'maskR']:
            im = getattr(self, image)
            im.show()
        elif image in ['gtR', 'gtL']:
            im = getattr(self, image)
            utils.plot_disp_map(im, image)


class SplitDataset(ABC):

    def __init__(self, split: List[int]) -> None:
        self.split: List[int] = split

        # initialise full dataset
        if self._load_dict_with_all_registries() is None:
            self.full_dataset: Dict[str, List[Dict]] = {'training_set': self._create_list('train'),
                                                        'test_set': self._create_list('test')}
        else:
            self.full_dataset: Dict[str, List[Dict]] = self._load_dict_with_all_registries()

        # initialise split dataset values
        self.train, self.val, self.test = utils.split_dataset(self.full_dataset['training_set'],
                                                              self.full_dataset['test_set'],
                                                              split[0], split[1], split[2])

        super().__init__()

    def load_registry_from_index(self, index: int, which_set: str, full_or_split: str) -> Registry:
        assert full_or_split in ['full', 'split']
        if full_or_split == 'full':
            assert which_set in ['train', 'test']
            examples_list: List[Dict] = self.full_dataset['training_set'] if which_set == 'train' else self.full_dataset['test_set']
            assert 0 <= index <= len(
                examples_list), 'Error in stereo pair loading. The index you asked is out of limits!'
            return self.load_registry(examples_list[index])
        else:
            assert which_set in ['train', 'val', 'test']
            examples_list: List[Dict] = getattr(self, which_set)
            assert 0 <= index <= len(
                examples_list), 'Error in stereo pair loading. The index you asked is out of limits!'
            return self.load_registry(examples_list[index])

    @abstractmethod
    def load_registry(self, registry_dict: Dict) -> Registry:
        pass

    @abstractmethod
    def _create_list(self, which_set: str) -> List[Dict]:
        pass

    @abstractmethod
    def _load_dict_with_all_registries(self) -> Union[None, Dict[str, List[Dict]]]:
        pass


