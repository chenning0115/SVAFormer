import numpy as np
from operator import truediv
import matplotlib.pyplot as plt

colors = [
    [0, 0, 1],
    [255, 128, 0],
    [127, 255, 0],
    [0, 255, 0],
    [0, 0, 255],
    [46, 139, 87],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
    [160, 82, 45],
    [160, 32, 240],
    [255, 127, 80],
    [218, 112, 214],
    [255, 0, 0],
    [255, 255, 0],
    [127, 255, 212],
    [216, 191, 216],
    [238, 130, 238],
    [0, 128, 128],
    [255, 165, 0],
    [75, 0, 130],
    [255, 99, 71],
    [128, 0, 0],
    [0, 128, 255],
    [128, 255, 0]
]
colors = [
    [0, 0, 1],
    [255, 128, 0],
    [127, 255, 0],
    [0, 255, 0],
    [0, 0, 255],
    [46, 139, 87],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
    [160, 82, 45],
    [160, 32, 240],
    [255, 127, 80],
    [218, 112, 214],
    [255, 0, 0],
    [255, 255, 0],
    [127, 255, 212],
    [216, 191, 216],
    [238, 130, 238],
    [0, 128, 128],
    [255, 165, 0],
    [75, 0, 130],
    [255, 99, 71],
    [128, 0, 0],
    [0, 128, 255],
    [128, 255, 0]
]
def data_to_colormap(data):
    assert len(data.shape)==2
    x_list = data.reshape((-1,))
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        y[index] = np.array(colors[item]) / 255
    return y

def classification_map(name, map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.set_title(name)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def plot_one(name, cls_data, raw_data, save_path, dpi=100):
    cls_data = cls_data.astype(np.int8)
    h,w = raw_data.shape
    indian_color = data_to_colormap(cls_data).reshape((h,w,3))
    classification_map(name, indian_color, raw_data, dpi, save_path)


def plot_all(cls_data, TR, TE, save_path, dpi=100):
    labels = TR + TE
    cls_data = cls_data + 1
    cls_data[labels==0] = 0
    path_label = "%s_label.png" % save_path
    path_pred = "%s_pred.png" % save_path
    plot_one('label', labels, labels, path_label, dpi=dpi)
    plot_one('pred', cls_data, labels, path_pred, dpi=dpi)