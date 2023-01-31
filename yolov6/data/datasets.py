#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)
from yolov6.utils.events import LOGGER

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break


class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
        step=16,
        sample_rate=1,
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = data_dict["names"]
        self.step = step
        self.sample_rate = sample_rate
        # NOTE(xiaowk): get images and labels
        self.get_imgs_labels(self.img_dir)
        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))

    def __len__(self):
        """Get the length of dataset"""
        return len(self.snippets)

    def __getitem__(self, path):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        # Load image
        if self.hyp and "test_load_size" in self.hyp:
            img, (h0, w0), (h, w) = self.load_image(path, self.hyp["test_load_size"])
        else:
            img, (h0, w0), (h, w) = self.load_image(path)

        # Letterbox
        shape = self.img_size
        if self.hyp and "letterbox_return_int" in self.hyp:
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, return_int=self.hyp["letterbox_return_int"])
        else:
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling

        # NOTE(xiaowk): 获取对应的标签
        labels = self.img_info[path]["label"].copy()
        if labels.size:
            w *= ratio
            h *= ratio
            # new boxes
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = (
                w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
            )  # top left x
            boxes[:, 1] = (
                h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
            )  # top left y
            boxes[:, 2] = (
                w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
            )  # bottom right x
            boxes[:, 3] = (
                h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            )  # bottom right y
            labels[:, 1:] = boxes

        # NOTE(xiaowk): 注意数据增强方法
        # if self.augment:
        #     img, labels = random_affine(
        #         img,
        #         labels,
        #         degrees=self.hyp["degrees"],
        #         translate=self.hyp["translate"],
        #         scale=self.hyp["scale"],
        #         shear=self.hyp["shear"],
        #         new_shape=(self.img_size, self.img_size),
        #     )

        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

        # NOTE(xiaowk): 注意数据增强方法
        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, path, shapes

    def load_image(self, path, force_load_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        im = cv2.imread(path)
        assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        if force_load_size:
            r = force_load_size / max(h0, w0)
        else:
            r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def get_imgs_labels(self, img_dir):
        
        assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
        valid_img_record = osp.join(
            osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json"
        )
        NUM_THREADS = min(8, os.cpu_count())

        # paths of all videos
        video_paths = []
        with open(img_dir, "r") as f:
            line = f.readline()
            while line:
                line = line.strip("\n").strip("\r")
                video_paths.append(line)
                line = f.readline()
        
        # snippets of videos, labels of snippets, information of images
        snippets = []
        snippets_labels = []
        img_info = {}

        # create snippets from videos
        if self.task.lower() == 'train':
            for video_path in video_paths:
                images_names = os.listdir(video_path)
                images_paths = [os.path.join(video_path, p) for p in images_names]
                images_paths = sorted(p for p in images_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p))
                snippet_nums = int(len(images_paths) / self.step)
                for i in range(snippet_nums):
                    snippet_begin = random.randint(0, snippet_nums)
                    if snippet_begin + self.step <= len(images_paths) and random.random() <= self.sample_rate:
                        snippet = images_paths[snippet_begin: snippet_begin+self.step]
                        snippets.append(snippet)

        elif self.task.lower() == 'val':
            for video_path in video_paths:
                images_names = os.listdir(video_path)
                images_paths = [os.path.join(video_path, p) for p in images_names]
                images_paths = sorted(p for p in images_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p))
                i = 0
                while i + self.step < len(images_paths):
                    snippet = images_paths[i: i+self.step]
                    snippets.append(snippet)
                    i += self.step
        else:
            assert self.task in ['train', 'val'], f"task must be train or val"

        assert snippets, f"No images found in {img_dir}."

        self.check_images = True

        # check images
        nc_snippets = []
        if self.check_images and self.main_process:
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image_list, snippets),
                    total=len(snippets),
                )
                for snippet, snippet_shapes, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        nc_snippets.append(snippet)
                        for i, (img_file, img_shape) in enumerate(zip(snippet, snippet_shapes)):
                            if img_info.__contains__(img_file):
                                img_info[img_file]["shape"] = img_shape
                            else:
                                img_info[img_file] = {"shape": img_shape}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            # if msgs:
            #     LOGGER.info("\n".join(msgs))
        snippets = nc_snippets
        # check and load anns
        # NOTE(xiaowk): load labels
        for snippet in snippets:
            snippet_label_path = []
            for img_path in snippet:
                label_path = img_path.replace("Data", "labels")
                label_path = label_path[:-len(img_path.split('.')[-1])]+"txt"
                # assert osp.isfile(label_path), f"not file {label_path}"
                img_info[img_path]["label_path"] = label_path
                snippet_label_path.append(label_path)
            snippets_labels.append(snippet_label_path)

        # assert label_paths, f"No labels found in {label_dir}."
        assert snippets_labels, f"No labels found."

        self.check_labels = True

        if self.check_labels:
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            nc_snippets = []
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_list, zip(snippets, snippets_labels)
                )
                pbar = tqdm(pbar, total=len(snippets_labels)) if self.main_process else pbar
                for (
                    img_per_snippet,
                    labels_per_snippet,
                    nc_per_snippet,
                    nm_per_snippet,
                    nf_per_snippet,
                    ne_per_snippet,
                    msg,
                ) in pbar:
                    if nc_per_snippet == 0:
                        nc_snippets.append(img_per_snippet)
                        for i, (img_path, img_label) in enumerate(zip(img_per_snippet, labels_per_snippet)):
                            img_info[img_path]["label"] = img_label
                    else:
                        for i, (img_path, img_label) in enumerate(zip(img_per_snippet, labels_per_snippet)):
                            img_info.pop(img_path)
                    nc += nc_per_snippet
                    nm += nm_per_snippet
                    nf += nf_per_snippet
                    ne += ne_per_snippet
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                # with open(valid_img_record, "w") as f:
                #     json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

            snippets = nc_snippets

        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dir) + ".json"
                )
                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        num_labels = 0
        for img_path, info in img_info.items():
            img_label = info["label"]
            if info["label"]:
                num_labels += len(info["label"])
                img_info[img_path]["label"] = np.array(img_label, dtype=np.float32)
            else:
                img_info[img_path]["label"] = np.zeros((0, 5), dtype=np.float32)

        # self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(snippets)*self.step}/ labels: {num_labels}. "
        )

        self.snippets = snippets
        self.img_info = img_info
        return

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        return img, labels


    @staticmethod
    def check_image_list(im_list):
        ncs, msgs = 0, ""
        shapes = []
        for im_file in im_list:
            im_file, shape, nc, msg = TrainValDataset.check_image(im_file)
            ncs += nc
            msgs += msg + "\n"
            shapes.append(shape)
        return im_list, shapes, ncs, msgs

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            im = Image.open(im_file)  # need to reload the image after using verify()
            shape = im.size  # (width, height)
            try:
                im_exif = im._getexif()
                if im_exif and ORIENTATION in im_exif:
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except:
                im_exif = None
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_list(args):
        img_list, label_list = args
        out_label = []
        out_nc, out_nm, out_nf, out_ne, out_msg = 0, 0, 0, 0, ""
        for img_path, label_path in zip(img_list, label_list):
            img, label, nc, nm, nf, ne, msg = TrainValDataset.check_label_files((img_path, label_path))
            out_label.append(label)
            out_nc += nc
            out_nm += nm
            out_nf += nf
            out_ne += ne
            out_msg += msg
        
        return img_list, out_label, out_nc, out_nm, out_nf, out_ne, out_msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                        labels = labels[indices]  # remove duplicates
                        # msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            
            labels, img_shape = info["label"], info["shape"]
            # labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_w, img_h = img_shape
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()


class LoadData:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise FileNotFoundError(f'Invalid path {p}')
        imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
        vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = 'image'
        if any(vidp):
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None
    @staticmethod
    def checkext(path):
        file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type
    def __iter__(self):
        self.count = 0
        return self
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap
    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __len__(self):
        return self.nf  # number of files
