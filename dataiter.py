from pycocotools.coco import COCO
import cv2
import os


class COCOCaption(object):
    def __init__(self, anno_path, image_path):
        coco = COCO(anno_path)
        imgIds = coco.getImgIds()
        self.coco = coco
        self.image_ids = imgIds
        self.image_path = image_path

    def __getitem__(self, idx):
        coco_caps = self.coco
        img = self.coco.loadImgs(self.image_ids[idx])[0]
        image_id = img['id']
        annIds = coco_caps.getAnnIds(imgIds=image_id);
        anns = coco_caps.loadAnns(annIds)
        file_name = img["file_name"]
        file_path = os.path.join(self.image_path, file_name)

        return cv2.imread(file_path)[:, :, ::-1], [x["caption"] for x in anns]

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    dataset = COCOCaption("/data3/zyx/yks/coco2017/annotations/captions_train2017.json",
                          "/data3/zyx/yks/coco2017/train2017")
    for x in dataset:
        print(x)
    print(len(dataset))