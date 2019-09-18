from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from collections import defaultdict
import glob
import os, sys
from caffe2.python import core, workspace, net_drawer
from caffe2.proto import caffe2_pb2

import pycocotools.mask as mask_util
import utils.boxes as box_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
from utils.timer import Timer
from core.nms_wrapper import nms, soft_nms
import utils.blob as blob_utils
import modeling.FPN as fpn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg, cfg_from_list
import logging
from core.test_engine import initialize_model_from_cfg
from core.test import im_conv_body_only, _project_im_rois
import utils.c2
import pprint
from utils.io import robust_pickle_dump
logger = logging.getLogger(__name__)
# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations.
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (list of ndarray): a list of color images in BGR order. In case of
        video it is a list of frames, else is is a list with len = 1.

    Returns:
        blob (ndarray): a data blob holding an image pyramid (or video pyramid)
        im_scale_factors (ndarray): array of image scales (relative to im) used
            in the image pyramid
    """
    all_processed_ims = []  # contains a a list for each frame, for each scale
    all_im_scale_factors = []
    for frame in im:
        processed_ims, im_scale_factors = blob_utils.prep_im_for_blob(
            frame, cfg.PIXEL_MEANS, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE)
        all_processed_ims.append(processed_ims)
        all_im_scale_factors.append(im_scale_factors)
    # All the im_scale_factors will be the same, so just take the first one
    for el in all_im_scale_factors:
        assert(all_im_scale_factors[0] == el)
    im_scale_factors = all_im_scale_factors[0]
    # Now get all frames with corresponding scale next to each other
    processed_ims = []
    for i in range(len(all_processed_ims[0])):
        for frames_at_specific_scale in all_processed_ims:
            processed_ims.append(frames_at_specific_scale[i])
    # Now processed_ims contains
    # [frame1_scale1, frame2_scale1..., frame1_scale2, frame2_scale2...] etc
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)
def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)
def maskrcnn(im_, bboxs = None, scale = 800, cfg_file = None, opts = None):
    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    #orig_maxSize = cfg.TEST.MAX_SIZE
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)
    name_scope = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    utils.c2.import_custom_ops()
    utils.c2.import_detectron_ops()
    utils.c2.import_contrib_ops()
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if opts is not None:
        cfg_from_list(opts)
    assert_and_infer_cfg()
    cfg.TEST.SCALES = (scale, )
    #cfg.TEST.MAX_SIZE = max_size
    logger.info('Extracting features with config:')
    logger.info(pprint.pformat(cfg))
    workspace.ResetWorkspace()
    model = initialize_model_from_cfg()
    #g = net_drawer.GetPydotGraph(model, rankdir="TB")
    #im_graph = g.create_png()
    #g.write_pdf('graph.pdf')
    _, im_H, im_W = im_.shape
    print("input image size is ".format(im_.shape))
    im_ = np.expand_dims(im_, 0)
    with core.NameScope(name_scope):
        with core.DeviceScope(gpu_dev):
            scale_factors = im_conv_body_only(model, im_)
            if bboxs is not None:
                rois, levels = _project_im_rois(bboxs, scale_factors)
            print("image scale factors are:{}".format(scale_factors))
            #ws_blobs = workspace.Blobs()
            print("Current workspace:{}".format(workspace.CurrentWorkspace()))
            ws_blob = workspace.FetchBlob('gpu_0/res5_2_sum')
            # switch the workspace to the ROIAlign. the second argument "True" means creating
            # the workspace if it is missing
            # create a ROI Align operator
            ws_blob = ws_blob[:, :, 0, :, :]
            print(" size of the feature map is:{}".format(ws_blob.shape))
            N, C, H, W = ws_blob.shape
            feature_scale = (1.0 * H)/ im_H
            roi_align = core.CreateOperator(
                "RoIAlign",
                ["featureIn", "rois"],
                ["featureOut"],
                spatial_scale = feature_scale,
                pooled_h = 7,
                pooled_w = 7,
            )
            workspace.FeedBlob("gpu_0/featureIn", ws_blob)
            workspace.FeedBlob("gpu_0/rois", np.array([rois[0]]).astype(np.float32))
            workspace.RunOperatorOnce(roi_align)
            feature_out = workspace.FetchBlob("gpu_0/featureOut")
            #print("test scales are:{}".format(cfg.TEST.SCALES))
            #print("maximum size is:{}".format(cfg.TEST.MAX_SIZE))
            #print("level is:{}".format(levels))
    # write back the original scale size to the config file
    cfg.TEST.SCALES = orig_scales
    #cfg.TEST.MAX_SIZE = orig_maxSize
    return feature_out

