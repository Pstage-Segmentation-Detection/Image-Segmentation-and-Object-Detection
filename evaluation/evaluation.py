# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
'''
evluation code of segmentation
'''
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import numpy as np
# resize
import albumentations as A
import os
import cv2

# 반드시 아래의 dataset_path를 설정해주세요. (mask resize하는데 test image를 필요로함)
# batch_01_vt, batch_02_vt, batch_03 의 디렉토리
dataset_path = '../input/data'

# public score (대회 진행중인 경우) : '/public.json'
# test_private score (대회 종료 후) : '/test_private.json'

# gt_path = dataset_path + '/test_private.json'
# gt_path = dataset_path + '/public.json'
# gt_path = dataset_path + '/private.json'

# directory of submission.csv
pred_path = './submission/test_256.csv'


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
                  'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def evaluation(gt_path, pred_path):
    '''
    mask, prediction (256 x 256)
    :param gt_path: directory of .json (ground_truth)
    :param pred_path: directory of submission.csv
    :return: mIoU (float)
    '''
    size = 256
    coco = COCO(gt_path)
    pred_df = pd.read_csv(pred_path, index_col=None)

    pred_file_names = pred_df['image_id'].values.tolist()
    predictions = pred_df['PredictionString'].values.tolist()

    json_file_name = [i['file_name'] for i in list(coco.imgs.values())]
    print(len(json_file_name))
    mIoU_list = []

    i_d = 0

    for index, (file_name, pred) in enumerate(zip(pred_file_names, predictions)):

        if file_name in json_file_name:
            image_id = coco.getImgIds(imgIds=i_d)[0]
            image_infos = coco.loadImgs(image_id)[0]
            pred = np.array(pred.split()).astype(int).reshape(size, size)

            if image_infos['file_name'] == file_name:
                ann_ids = coco.getAnnIds(imgIds=image_id)
                anns = coco.loadAnns(ann_ids)

                # segmentation
                # cv2 를 활용하여 image 불러오기 (aug_pipeline의 input을)
                image = cv2.imread(os.path.join(dataset_path, file_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.0

                # Load the categories in a variable
                cat_ids = coco.getCatIds()
                cats = coco.loadCats(cat_ids)

                # masks : size (height x width)
                # pixel = "category id + 1"
                # Background = 0
                masks = np.zeros((512, 512))
                # Unknown = 1, General trash = 2, ... , Cigarette = 11
                for i in range(len(anns)):
                    className = get_classname(anns[i]['category_id'], cats)
                    pixel_value = category_names.index(className)
                    masks = np.maximum(coco.annToMask(anns[i]) * pixel_value, masks)
                masks = masks.astype(np.long)

                # Compose an augmentation pipeline
                aug_pipeline = A.Compose([A.Resize(height=256, width=256)])
                augmented = aug_pipeline(image=image, mask=masks)
                masks = augmented['mask']

                # mIoU
                mIoU = label_accuracy_score(masks, pred, n_class=12)[2]
                mIoU_list.append(mIoU)

                i_d += 1
            else:
                print(image_infos['file_name'], "!=", file_name)
                print('index error')
                raise Exception('error', 'invalid image_id')

        else:
            continue

    return np.mean(mIoU_list)


# if __name__ == '__main__':
#     # # 837 imgages => ok
#     gt_path = '../input/data/test_private.json'
#     # # 417 imgages => ok
#     # gt_path = '../input/data/public.json'
#     # # 420 imgages => ok
#     # private_path = './data/private.json'
#
#     mIoU = evaluation(gt_path=gt_path, pred_path=pred_path)
#     print(mIoU)