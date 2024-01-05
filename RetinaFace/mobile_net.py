from __future__ import print_function
import argparse
import time
import numpy as np
import torch
from RetinaFace.data import cfg_mnetv3
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.models.retinaface_g import RetinaFace
from RetinaFace.utils.box_utils import decode, decode_landm
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms


def check_keys(model, pretrained_state_dict, logs: bool = False):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if logs:
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix, logs: bool = False):
    if logs:
        print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu, logs: bool = False):
    if logs:
        print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.', logs=logs)
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.', logs=logs)
    check_keys(model, pretrained_dict, logs=logs)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Net:
    def __init__(self, load2cpu: bool = False, logs: bool = False,
                 modelPath: str = 'RetinaFace/weights/mobilev3_Final.pth'):
        parser = argparse.ArgumentParser(description='Retinaface')
        parser.add_argument('-m', '--trained_model', default=modelPath,
                            type=str, help='Trained state_dict file path to open')
        parser.add_argument('--network', default='mobilev3',
                            help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
        parser.add_argument('--image', type=str, default=r'./curve/face.jpg', help='detect images')
        parser.add_argument('--fourcc', type=int, default=1, help='detect on webcam')
        parser.add_argument('--cpu', action="store_true", default=load2cpu, help='Use cpu inference')
        parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
        parser.add_argument('--top_k', default=5000, type=int, help='top_k')
        parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
        parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
        parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
        parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
        self.args = parser.parse_args()

        torch.set_grad_enabled(False)
        torch.set_grad_enabled(False)

        self.net = RetinaFace(cfg=cfg_mnetv3)
        self.net = load_model(self.net, self.args.trained_model, load_to_cpu=self.args.cpu, logs=logs)
        self.net.eval()
        if logs:
            print('Finished loading model!')

        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.net = self.net.to(self.device)

    def detect(self, img_raw, logs: bool = False):
        img = np.float32(img_raw)
        # print(img.shape)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        if logs:
            print('net forward time: {:.4f}ms'.format((time.time() - tic) * 1000))

        priorbox = PriorBox(cfg_mnetv3, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnetv3['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnetv3['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        landms = landms[:self.args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        return dets
