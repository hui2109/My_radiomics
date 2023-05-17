import os
import pickle
import time

import pandas as pd
from radiomics import featureextractor

import myconstant


def extract_features(img, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableImageTypes(
        Original={},
        Wavelet={})
    return extractor.execute(img, mask)


def batch_extract(result_path=None):
    # 提供了excel文件就不用再提取了
    if result_path:
        return result_path

    if not os.path.exists(myconstant.NrrdFiles):
        print('没有数据可供提取，请先运行image_to_nrrd')

    with open(myconstant.PKLFile, 'rb') as f:
        finding_list = pickle.load(f)

    df = pd.DataFrame()
    for results in finding_list:  # results是一个字典
        img = results['img']
        mask = results['mask']
        roi_value = results['roi_value']

        # 特征提取
        featureVector = extract_features(img=img, mask=mask)
        featureVector['label'] = roi_value

        # 去掉以diagnostic开头的特征
        # for key in featureVector.keys():
        #     if key.startswith('diagnostic'):
        #         del featureVector[key]

        # 将提取的特征转换为DataFrame格式
        df_new = pd.DataFrame.from_dict(featureVector.values()).T
        df_new.columns = featureVector.keys()
        df = pd.concat([df, df_new])

    # 将提取的特征结果写入文件
    if not os.path.exists(myconstant.FeatureResults):
        os.makedirs(myconstant.FeatureResults)
    result_path = f'{myconstant.FeatureResults}/results_{int(time.time())}.xlsx'
    with pd.ExcelWriter(result_path) as writer:
        df.to_excel(writer, index=False)

    return result_path
