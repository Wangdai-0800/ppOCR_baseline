#import pydevd_pycharm
#pydevd_pycharm.settrace('192.168.2.186`', port=10001, stdoutToServer=True, stderrToServer=True)
#Using this code to get the prediction of PaddleOCR_td_db_mobile on DIY dataset

import paddlehub as hub
import os
from glob import glob
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main():
    #dataPath = '/data/wd_DBnet_data/origin_data/datasets/icdar2015/test_images/'
    #dataPath = '/data/wd_DBnet_data/PSEnet/DB-TestData/'
    dataPath = '/data/wd_DBnet_data/PSEnet/DB_TestData2_Straight/'

    testImg = glob(dataPath + "*/*.jpg")
    outputPath = '/data/wd_DBnet_data/ppOCR/DIY_results_TD2_Straight/'
    predPath = outputPath + 'prediction/'

    text_detector = hub.Module(name="chinese_text_detection_db_mobile")
    BOX_THRESH = 0.4

    for i in range(len(testImg)):
        curImg = testImg[i]
        print("Processing Image %d: %s " % (i, curImg))
        result = text_detector.detect_text(paths=[curImg], use_gpu=False,
                                          output_dir=outputPath,
                                          box_thresh=BOX_THRESH, visualization=True)
        result = result[0]
        savePath = result['save_path']
        detResult = np.array(result['data'])

        curName = curImg.split('/')[-1]
        typicalName = outputPath + 'det_' + curName
        os.rename(savePath, typicalName)
        if os.path.exists(predPath):
            pass
        else:
            os.mkdir(predPath)
        predFilePath = "res_" + curName.replace('.jpg', '.txt')
        predFilePath = predPath + predFilePath
        with open(predFilePath, 'w') as fout:
            if len(detResult.shape) == 2:
                dataMat = detResult
                dataMat = np.reshape(dataMat, [8, ])
                dataMat = [str(item) for item in dataMat]
                line = ','.join(dataMat) + '\r\n'
                fout.write(line)
            else:
                for recIndex in range(detResult.shape[0]):
                    dataMat = detResult[recIndex]
                    dataMat = np.reshape(dataMat, [8, ])
                    dataMat = [str(item) for item in dataMat]
                    line = ','.join(dataMat) + '\r\n'
                    fout.write(line)
        print("Image %d Done." % i)

if __name__ == "__main__":
    main()