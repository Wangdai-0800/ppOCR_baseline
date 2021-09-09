import numpy as np
import Polygon as plg
from glob import glob

# Threshold for selecting ignored Detected Polygon
AREA_PRECISION_CONSTRAINT = 0.5
# Threshold for IoU-mat
IOU_CONSTRAINT = 0.5


def createPolygon(gtPath, predPath):
    gtFileList = glob(gtPath + "*.txt")
    predFileList = glob(predPath + "*.txt")
    assert (len(gtFileList) == len(predFileList))

    all_gtPtsList = []
    all_predPtsList = []
    all_gtPolyList = []
    all_predPolyList = []
    all_gtTextList = []

    for file in gtFileList:
        gtPtsList = []
        predPtsList = []
        gtPolyList = []
        predPolyList = []
        gtTextList = []

        with open(file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                params = line.split(',')
                if len(params) == 9:
                    pts = [int(params[_]) for _ in range(8)]
                    transcription = params[-1]
                    gtPtsList.append(pts)
                    gtTextList.append(transcription)
                else:
                    pts = [int(params[_]) for _ in range(8)]
                    gtPtsList.append(pts)

        # Find the corresponding pred file
        pred_file = predPath + file.split('/')[-1]
        assert (pred_file in predFileList)

        with open(pred_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                params = line.split(',')
                if len(params) == 9:
                    pts = [int(params[_]) for _ in range(8)]
                    predPtsList.append(pts)
                else:
                    pts = [int(params[_]) for _ in range(8)]
                    predPtsList.append(pts)

        gtPtsArray = np.array(gtPtsList)
        predPtsArray = np.array(predPtsList)

        for i in range(len(gtPtsArray)):
            gtPolyList.append(plg.Polygon(gtPtsArray[i, :].reshape([4, 2])))
        for i in range(len(predPtsArray)):
            predPolyList.append(plg.Polygon(predPtsArray[i, :].reshape([4, 2])))

        all_gtPtsList.append(gtPtsArray)
        all_gtPolyList.append(gtPolyList)
        all_predPtsList.append(predPtsArray)
        all_predPolyList.append(predPolyList)
        all_gtTextList.append(gtTextList)

    return all_gtPtsList, all_gtPolyList, all_predPtsList, all_predPolyList, all_gtTextList


def get_union(pD, pG):
    areaA = pD.area();
    areaB = pG.area();
    return areaA + areaB - get_intersection(pD, pG);


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG);
    except:
        return 0


def MethodEvaluation(gtPath, predPath):
    # gtPath = '/data/wd_DBnet_data/PSEnet/PSENet-master/outputs/diy/submit_diy/'
    # predPath = "/data/wd_DBnet_data/ppOCR/DIY_results/prediction/"

    all_gtPtsList, all_gtPolyList, all_predPtsList, \
    all_predPolyList, all_gtTextList = createPolygon(gtPath, predPath)

    matchedSum = 0
    numGlobalCareGt = 0
    numGlobalCareDet = 0

    testSampleNum = len(all_gtPtsList)
    for sample in range(testSampleNum):
        print("Testing Sample %d :" % sample)
        gtPoly = all_gtPolyList[sample]
        predPoly = all_predPolyList[sample]
        textList = all_gtTextList[sample]

        recall = 0
        precision = 0
        hmean = 0

        IoU_Mat = np.zeros((len(gtPoly), len(predPoly)))
        matchedNum = 0
        gtDontCareID = []
        detDontCareID = []
        predDontCareID = []
        detMatchedID = []

        # 取gt中的可忽略样本
        for i in range(len(textList)):
            if textList[i] == '###':
                gtDontCareID.append(i)
        print("%d Polygons is ignored." % len(gtDontCareID))

        # 筛选预测结果的可忽略结果
        if len(gtDontCareID) > 0:
            for ignoreIndex in gtDontCareID:
                ignorePoly = gtPoly[ignoreIndex]
                for detPoly in predPoly:
                    intersected_area = get_intersection(ignorePoly, detPoly)
                    pdDimensions = detPoly.area()
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > AREA_PRECISION_CONSTRAINT):
                        detDontCareID.append(predPoly.index(detPoly))
                        # break

        # 检测
        if len(gtPoly) > 0 and len(predPoly) > 0:
            # Calculate IoU and precision matrixs

            gtRectMat = np.zeros(len(gtPoly), np.int8)
            detRectMat = np.zeros(len(predPoly), np.int8)
            for gtNum in range(len(gtPoly)):
                for detNum in range(len(predPoly)):
                    pG = gtPoly[gtNum]
                    pD = predPoly[detNum]
                    IoU_Mat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPoly)):
                for detNum in range(len(predPoly)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                        detNum] == 0 and gtNum not in gtDontCareID and detNum not in detDontCareID:
                        if IoU_Mat[gtNum, detNum] > IOU_CONSTRAINT:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            matchedNum += 1
                            detMatchedID.append(detNum)

        numGtCare = (len(gtPoly) - len(gtDontCareID))
        numDetCare = (len(predPoly) - len(detDontCareID))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            sampleAP = precision
        else:
            recall = float(matchedNum) / numGtCare
            precision = 0 if numDetCare == 0 else float(matchedNum) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += matchedNum
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        localMetrics = {"SampleID": sample, 'recall': round(recall, 2), 'precision': round(precision, 2), 'hmean': round(hmean, 2)}
        #print(localMetrics)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)

    methodMetrics = {'precision': round(methodPrecision, 2), 'recall': round(methodRecall, 2), 'hmean': round(methodHmean, 2)}

    return methodMetrics


def main():
    gtPath = '/data/wd_DBnet_data/PSEnet/PSENet-master/outputs/DB_TestData2/submit_diy/'
    # modelUsed = 'myMV3_mixup_99000_withoutOC'
    modelUsed = "XH_Straight"

    if modelUsed == 'ppOCR_DB_mobile':
        #ppOCR db_mobile
        predPath = "/data/wd_DBnet_data/ppOCR/DIY_results_TD2/prediction/"
    elif modelUsed == 'myMV3_mixup':
        #myMV3—mixup
        predPath = "/data/wd_DBnet_data/Results/DIY_myMV3/labels/"
    elif modelUsed == 'res18':
        #Res18
        predPath = "/data/wd_DBnet_data/Results/DIY_res18/labels/"
    elif modelUsed == 'mv3':
        # MV3 IC15 1200 epochs
        predPath = "/data/wd_DBnet_data/Results/DIY_mv3/labels/"
    elif modelUsed == 'myMV3_mixup_99000':
        #myMV3—mixup
        predPath = "/data/wd_DBnet_data/Results/TestData2/diy_myMV3_99000_withOC/labels/"
    elif modelUsed == "myMV3_mixup_99000_withoutOC":
        #myMV3-mixup-withoutOCoperation
        predPath = "/data/wd_DBnet_data/Results/TestData2/diy_myMV3_99000_withoutOC/labels/"

    methodMetrics = MethodEvaluation(gtPath, predPath)
    print("\n\t####  Result  ####")
    print(methodMetrics)
    print("Model Used: ", modelUsed)
    print("IOU_CONSTRAINT :", IOU_CONSTRAINT)

if __name__ == "__main__":
    main()
