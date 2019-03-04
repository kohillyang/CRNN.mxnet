import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
pathes = []
labels = []
for root_dir, _, names in os.walk("/data2/zyx/yks/dianke_ocr/dataset/train_val_dataset"):
    for name in names:
        label = name.replace('.jpg', '')
        pathes.append(name)
        labels.append(label)
        # image = cv2.imread(os.path.join(root_dir, name))
        # print(name)
        # cv2.imshow('', image)
        # cv2.waitKey(0)

content = {"name": pathes, 'content':labels}
cf = pd.DataFrame(content)
train_set, val_set = train_test_split(cf, test_size=.1, random_state=42, shuffle=True)
train_set.to_csv("/data2/zyx/yks/dianke_ocr/dataset/train_set.csv", index=None)
val_set.to_csv("/data2/zyx/yks/dianke_ocr/dataset/val_set.csv", index=None)
print(cf)