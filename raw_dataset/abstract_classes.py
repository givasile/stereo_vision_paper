from abc import ABC, abstractmethod, abstractproperty
import raw_dataset.utils as utils


class registry(ABC):

    def __init__(self, index, id, imL_g, imR_g, imL_rgb, imR_rgb, gtL, gtR,
                 maskL, maskR, maxDL, maxDR, dataset):
        self.index = index
        self.id = id
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

    def plot(self, which):
        assert which in ['imL_g', 'imR_g', 'imL_rgb', 'imR_rgb',
                         'gtL', 'gtR', 'maskL', 'maskR']
        if which in ['imL_g', 'imR_g', 'imL_rgb', 'imR_rgb', 'maskL', 'maskR']:
            x = getattr(self, which)
            x.show()
        elif which in ['gtR', 'gtL']:
            x = getattr(self, which)
            utils.plot_disp_map(x, which)


class dataset(ABC):

    @abstractproperty
    def train(self):
        pass

    @abstractproperty
    def test(self):
        pass

    @abstractproperty
    def max_disp(self):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def load(self):
        pass


class split_dataset(ABC):

    @abstractmethod
    def _create_list(self):
        pass

    @abstractmethod
    def load_registry(self):
        pass

    # constructor
    def __init__(self, split):
        self.split = split
        self.train, self.val, self.test = utils.split(
            self._create_list('train'), self._create_list('test'),
            split[0], split[1], split[2])
        super().__init__()

    # methods
    def load_from_index(self, reg, tr_te):
        assert tr_te in ['train', 'test']
        message = 'Error in stereo pair loading. The reg you asked is out of limits!'
        if tr_te == 'train':
            assert reg >= 0 and reg <= len(self._create_list('train')), message
            train = self._create_list('train')
            return self.load_registry(train[reg])
        elif tr_te == 'test':
            assert reg >= 0 and reg <= len(self._create_list('test')), message
            test = self._create_list('test')
            return self.load_registry(test[reg])
