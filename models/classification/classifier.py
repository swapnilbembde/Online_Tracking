import numpy as np
import cv2
import torch
from torch.autograd import Variable

from models import net_utils
from models.classification.rfcn import Extractor as ClassifiedModel


def _factor_closest(num, factor, is_ceil=True):
    num = float(num) / factor
    num = np.ceil(num) if is_ceil else np.floor(num)
    return int(num) * factor


def crop_with_factor(im, dest_size, factor=32, pad_val=0, basedon='min'):
    im_size_min, im_size_max = np.min(im.shape[0:2]), np.max(im.shape[0:2])
    im_base = {'min': im_size_min,
               'max': im_size_max,
               'w': im.shape[1],
               'h': im.shape[0]}
    im_scale = float(dest_size) / im_base.get(basedon, im_size_min)
    #print(im_scale,"scale")
    # Scale the image.
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    # Compute the padded image shape. Ensure it's divisible by factor.
    h, w = im.shape[:2]
    new_h, new_w = _factor_closest(h, factor), _factor_closest(w, factor)
    new_shape = [new_h, new_w] if im.ndim < 3 else [new_h, new_w, im.shape[-1]]

    # Pad the image.
    im_padded = np.full(new_shape, fill_value=pad_val, dtype=im.dtype)
    im_padded[0:h, 0:w] = im

    return im_padded, im_scale, im.shape


class Classifier(object):
    def __init__(self, gpu=0):
        self.gpu = gpu

        ckpt = 'data/classifier_base.h5'
        model = ClassifiedModel(extractor='squeezenet')

        net_utils.load_net(ckpt, model)
        model = model.eval()
        self.model = model.cuda(self.gpu)
        print('load cls model from: {}'.format(ckpt))
        self.score_map = None
        self.im_scale = 1.

    @staticmethod
    def im_preprocess(image):

        # resizing and cropping according to the original paper model
        if min(image.shape[0:2]) > 720:
            real_inp_size = 640
        else:
            real_inp_size = 368 
        im_pad, im_scale, real_shape = crop_with_factor(image, real_inp_size, factor=16, pad_val=0, basedon='min')

        # preprocess image
        im_croped = cv2.cvtColor(im_pad, cv2.COLOR_BGR2RGB)
        im_croped = im_croped.astype(np.float32) / 255. - 0.5

        return im_croped, im_pad, real_shape, im_scale

    def update(self, image):
        im_croped, im_pad, real_shape, im_scale = self.im_preprocess(image)
        self.im_scale = im_scale
        im_data = torch.from_numpy(im_croped).permute(2, 0, 1)
        im_data = im_data.unsqueeze(0)

        # forward
	with torch.no_grad():
	    im_var = Variable(im_data).cuda(self.gpu)
	    self.score_map = self.model(im_var)

        return real_shape, im_scale

    def predict(self, rois):
        """
        :param rois: numpy array [N, 4] ( x1, y1, x2, y2)
        :return: scores [N]
        """
        scaled_rois = rois * self.im_scale
        return self.model.get_cls_score_numpy(self.score_map, scaled_rois)
