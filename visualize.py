import configparser
import numpy as np
import matplotlib.pyplot as plt

# define colorstyle
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = 'gray'


def imshow_3ch(img, name=None):
    npimg = img.numpy().transpose((1, 2, 0))
    npimg = np.clip(npimg, 0, 1)
    plt.figure()
    plt.imshow(npimg)
    if name is not None:
        plt.title(name)
    plt.show(block=False)


def imshow_mch(img, c, name):
    # Tensor -> np
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.figure()
    plt.title(name)
    plt.imshow(npimg[:, :, c])
    plt.show(block=False)


def imshow_1ch(im, name, lim=None):
    im = im.numpy()
    plt.figure()
    plt.title(name)
    plt.imshow(im)
    if lim is not None:
        plt.clim(lim[0], lim[1])
    plt.colorbar(orientation='horizontal')
    plt.show(block=False)


def mean_mae_curves(stats):
    x = np.arange(len(stats['train']['mae'])) + 1
    plt.figure()
    plt.title('mean error curves')

    def foo(arr):
        tmp = []
        for i in range(len(arr)):
            tmp.append(np.mean(arr[i]))
        return tmp
    plt.plot(x, foo(stats['train']['mae']), label='train')
    plt.plot(x, foo(stats['val_crop']['mae']), label='val_crop')
    plt.plot(x, foo(stats['val_full']['mae']), label='val_full')
    plt.legend()
    plt.show(block=False)


def mean_pcg_curves(stats):
    x = np.arange(len(stats['train']['mae'])) + 1
    plt.figure()
    plt.title('mean pcg curves')

    def foo(arr):
        tmp = []
        for i in range(len(arr)):
            tmp.append(np.mean(arr[i]))
        return tmp
    plt.plot(x, foo(stats['train']['pcg']), label='train')
    plt.plot(x, foo(stats['val_crop']['pcg']), label='val_crop')
    plt.plot(x, foo(stats['val_full']['pcg']), label='val_full')
    plt.legend()
    plt.show(block=False)


def scatter_plot(lis):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, len(lis)+1):
        plt.scatter(i, np.array(lis[i-1]).mean())
    plt.title('mae_test_full')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    # for i,j in zip(np.arange(1, array.shape[0]+1), array.mean(1)):
    #     ax.annotate('%.2f' % j,xy=(i,j))
    plt.show(block=False)
    # fig.savefig('/home/givasile/Desktop/figure_1.png')
