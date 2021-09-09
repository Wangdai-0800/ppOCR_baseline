#import pydevd_pycharm
#pydevd_pycharm.settrace('192.168.2.186`', port=10001, stdoutToServer=True, stderrToServer=True)

import paddlehub as hub
import cv2
import os
from glob import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataPath = '/data/wd_DBnet_data/origin_data/datasets/icdar2015/test_images/'
testImg = glob(dataPath + "*.jpg")
text_detector = hub.Module(name="chinese_text_detection_db_server")
result = text_detector.detect_text(paths=testImg, use_gpu=False,
                                  output_dir='/data/wd_DBnet_data/origin_data/datasets/icdar2015/detection_result',
                                  box_thresh=0.5, visualization=True)

print("Result")
print(result)