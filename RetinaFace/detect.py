from __future__ import print_function
import argparse
import torch
import numpy as np
from data import cfg_mnetv3
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
from models.retinaface_g import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect(img_raw):
    img = np.float32(img_raw)
    print(img.shape)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}ms'.format((time.time() - tic) * 1000))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/mobilev3_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobilev3',
                        help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
    parser.add_argument('--image', type=str, default=r'./curve/face.jpg', help='detect images')
    parser.add_argument('--fourcc', type=int, default=1, help='detect on webcam')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_grad_enabled(False)

    cfg = cfg_mnetv3

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, load_to_cpu=False)
    net.eval()
    print('Finished loading model!')

    device = torch.device("cuda")
    net = net.to(device)

    cap = cv2.VideoCapture(0)
    cap.set(3, 720)  # set video width
    cap.set(4, 680)  # set video height

    tm = cv2.TickMeter()
    while True:
        ret, frame = cap.read()
        tm.start()
        dets = detect(frame)
        image_d = frame.copy()
        tm.stop()
        cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        res = {}

        for i, b in enumerate(dets):
            if b[4] < 0.8:
                continue
            b_new = list(map(int, b))
            img_detect = image_d[b_new[1]:b_new[3], b_new[0]:b_new[2]]
            if 0 in img_detect.shape:
                continue
            res[i] = b

        for i, b in res.items():
            text = "{:.4f}".format(b[4])
            b_new = list(map(int, b))
            img_detect = image_d[b_new[1]:b_new[3], b_new[0]:b_new[2]]
            cv2.imshow("detect" + str(i), img_detect)

            cv2.rectangle(frame, (b_new[0], b_new[1]), (b_new[2], b_new[3]), (0, 255, 0), 2)
            cx = b_new[0]
            cy = b_new[1] + 12
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

            cv2.circle(frame, (b_new[5], b_new[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (b_new[7], b_new[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (b_new[9], b_new[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (b_new[11], b_new[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (b_new[13], b_new[14]), 1, (255, 0, 0), 4)

        cv2.imshow('fourcc', frame)
        tm.reset()
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
