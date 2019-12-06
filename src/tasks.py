# This file implement dataset and tasks.
# dataset loads and preprocess data
# task manages data, create transformerd version, create dataloader, loss and evaluation metrics
import logging as log
import os
import torch
import json
import numpy
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn import functional as F

def get_task(name, args):

    if name == "stl10_un":
        return STL10(name, args, pretrain=True)
    if name.startswith("stl10_fd"):
        return STL10(name, args, fold=int(name.replace("stl10_fd", "")))
    elif name == "cifar10_un":
        return CIFAR10(name, args, pretrain=True)
    elif name.startswith("cifar10_lp"):
        return CIFAR10(name, args, label_pct=float(name.replace("cifar10_lp", "")) / 100)
    elif name == "cifar100_un":
        return CIFAR100(name, args, pretrain=True)
    elif name.startswith("cifar100_lp"):
        return CIFAR100(name, args, label_pct=float(name.replace("cifar100_lp", "")) / 100)
    elif name.startswith("mnist_un"):
        return MNIST(name, args, pretrain=True)
    elif name.startswith("mnist_lp"):
        return MNIST(name, args, label_pct=float(name.replace("mnist_lp", "")) / 100)
    elif name.startswith("imagenet_un"):
        return ImageNet(name, args, pretrain=True)
    elif name.startswith("imagenet_lp"):
        return ImageNet(name, args, label_pct=float(name.replace("imagenet_lp", "")) / 100)
    else:
        raise NotImplementedError


def task_num_class(name):
    if name.startswith("imagenet"):
        return 1000
    elif name.startswith("cifar100"):
        return 100
    else:
        return 10


class RandomTranslateWithReflect:
    """
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = numpy.random.randint(
            -self.max_translation, self.max_translation + 1, size=2
        )
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop(
            (
                xpad - xtranslation,
                ypad - ytranslation,
                xpad + xsize - xtranslation,
                ypad + ysize - ytranslation,
            )
        )
        return new_image


class DupTransform:
    def __init__(self, num_dup, transform=lambda x: x):
        self.num_dup = num_dup
        self.transform = transform

    def __call__(self, inp):
        output = torch.stack([self.transform(inp) for _ in range(self.num_dup + 1)], dim=0)
        return output


class RandZero:
    def __init__(self, num_patches, num_queries):
        self.num_patches = num_patches
        self.num_queries = num_queries

    def __call__(self, query):
        mask = torch.randperm(self.num_patches) < self.num_queries
        return query * mask


class ToPatches:
    def __init__(self, num_patches):
        self.num_div = int(numpy.sqrt(num_patches))

    def __call__(self, inp):
        channel, height, width = inp.size()
        out = (
            inp.view(
                channel, self.num_div, height // self.num_div, self.num_div, width // self.num_div
            )
            .transpose(2, 3)
            .flatten(1, 2)
            .transpose(0, 1)
        )
        return out


class TransformDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, transform, tensors):
        self.transform = transform
        self.tensors = tensors
        for key in self.tensors:
            if key not in self.transform:
                self.transform[key] = lambda x: x

    def __getitem__(self, index):
        return {key: self.transform[key](tensor[index]) for key, tensor in self.tensors.items()}

    def __len__(self):
        return len(list(self.tensors.values())[0])


class Task(object):
    def __init__(self, name, args, pretrain=False):
        """
        inputs:
            name: str, dataset name
            args: args, global arguments
            pretrain: bool, load pretrain (self-supervised) data or finetune (supervised) data
        """
        self.name = name
        self.args = args
        self.pretrain = pretrain
        self.data_iterators = {}
        self.reset_scorers()
        self.path = os.path.join(args.data_dir, self.name.split("_")[0])
        if pretrain:
            self.eval_metric = "jigsaw_acc"
        else:
            self.eval_metric = "cls_acc"

    def _get_transforms(self):
        """
        outputs:
            train_transform: ...
            eval_transform: ...
        """
        raise NotImplementedError

    def _load_raw_data(self):
        """
        outputs:
            raw_data: dict[str, list[(image, label)]]: from split to list of data
                image: pil (*), raw image
                label: long (*), class label of the image, set to 0 when unavailable
        """
        raise NotImplementedError

    def _preprocess_data(self, data):
        output = {}
        for split, dataset in data.items():
            idx, image, label = zip(
                *[(idx, img, label) for idx, (img, label) in enumerate(dataset)]
            )
            output[split] = {
                "idx": torch.LongTensor(idx),
                "image": image,
                "query": torch.ones((len(image), self.args.num_patches), dtype=torch.bool),
                "label": torch.LongTensor(label),
            }
            if self.pretrain:
                del output[split]["label"]
            else:
                del output[split]["query"]

        return output

    def make_data_split(self, train_data, pct=1.0):
        split_filename = os.path.join(self.path, "%s.json" % self.name)
        if os.path.exists(split_filename):
            with open(split_filename, "r") as f:
                split = json.loads(f.read())
        else:
            full_size = len(train_data)
            train_size = int(full_size * pct * 0.9)
            val_size = int(full_size * pct * 0.1) + train_size
            full_idx = numpy.random.permutation(full_size).tolist()
            split = {"train": full_idx[:train_size], "val": full_idx[train_size:val_size]}
            with open(split_filename, "w") as f:
                f.write(json.dumps(split))
        val_data = [train_data[idx] for idx in split["val"]]
        train_data = [train_data[idx] for idx in split["train"]]
        return train_data, val_data

    def load_data(self):
        """
        load data, create data iterators. use cached data when available.
        """
        log.info("Loading %s data" % self.name)
        data = self._load_raw_data()
        data = self._preprocess_data(data)

        train_transform, eval_transform = self._get_transforms()
        if self.pretrain:
            data["train"] = TransformDataset(train_transform, data["train"])
            data["val"] = TransformDataset(eval_transform, data["val"])
        else:
            data["train"] = TransformDataset(train_transform, data["train"])
            data["val"] = TransformDataset(eval_transform, data["val"])
            data["test"] = TransformDataset(eval_transform, data["test"])

        for split, dataset in data.items():
            self.data_iterators[split] = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=(split == "train"),
                pin_memory=True,
                drop_last=(split == "train"),
                num_workers=self.args.num_workers,
            )

    def reset_scorers(self):
        self.scorers = {"count": 0}
        if self.pretrain:
            self.scorers.update({"loss": [], "jigsaw_acc": []})
            # TODO: Update this when new auxiliary losses are introduced
        else:
            self.scorers.update({"loss": [], "cls_acc": []})

    def update_scorers(self, batch_input, batch_output):
        count = len(batch_input["idx"])
        self.scorers["count"] += count
        for key in self.scorers.keys():
            if key != "count":
                self.scorers[key].append(batch_output[key].cpu().sum() * count)

    def report_scorers(self, reset=False):
        avg_scores = {
            key: sum(value) / self.scorers["count"]
            for key, value in self.scorers.items()
            if key != "count" and value != []
        }
        if reset:
            self.reset_scorers()
        return avg_scores


class nce_loss(Module):
    def __init__(self, size_average = True):
        super(nce_loss, self).__init__()
        self.size_average = size_average

    def forward(self, input):
        logits, nce_target = input
        N, Kp1 = logits.size()  # num true x (num_noise+1)
        loss = F.binary_cross_entropy_with_logits(logits, nce_target, reduce = False)
        loss = torch.sum(loss)
        if self.size_average:
            loss /= N
        return loss

class CIFAR10(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    def _get_transforms(self):
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.244, 0.262],)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        if self.pretrain:
            train_transform = eval_transform = {
                "idx": DupTransform(self.args.dup_pos),
                "image": DupTransform(
                    self.args.dup_pos,
                    transforms.Compose(
                        [
                            flip_lr,
                            img_jitter,
                            col_jitter,
                            rnd_gray,
                            transforms.ToTensor(),
                            normalize,
                            ToPatches(self.args.num_patches),
                        ]
                    ),
                ),
                "query": DupTransform(
                    self.args.dup_pos, RandZero(self.args.num_patches, self.args.num_queries)
                ),
            }
        else:
            train_transform = {
                "image": transforms.Compose(
                    [
                        flip_lr,
                        img_jitter,
                        col_jitter,
                        rnd_gray,
                        transforms.ToTensor(),
                        normalize,
                       # ToPatches(self.args.num_patches),
                    ]
                ),
            }
            eval_transform = {
                "image": transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        normalize, 
                        #ToPatches(self.args.num_patches)
                    ]
                ),
            }
        return train_transform, eval_transform

    def _load_raw_data(self):
        cifar10_train = datasets.CIFAR10(root=self.path, train=True, download=True)
        if self.pretrain:
            cifar10_train, cifar10_val = self.make_data_split(cifar10_train, 1.0)
            raw_data = {"train": cifar10_train, "val": cifar10_val}
        else:
            cifar10_test = datasets.CIFAR10(root=self.path, train=False, download=True)
            cifar10_train, cifar10_val = self.make_data_split(cifar10_train, self.label_pct)
            raw_data = {"train": cifar10_train, "val": cifar10_val, "test": cifar10_test}
        return raw_data


class CIFAR100(CIFAR10):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain, label_pct)

    def _load_raw_data(self):
        cifar100_train = datasets.CIFAR100(root=self.path, train=True, download=True)
        if self.pretrain:
            cifar100_train, cifar100_val = self.make_data_split(cifar100_train, 1.0)
            raw_data = {"train": cifar100_train, "val": cifar100_val}
        else:
            cifar100_test = datasets.CIFAR100(root=self.path, train=False, download=True)
            cifar100_train, cifar100_val = self.make_data_split(cifar100_train, self.label_pct)
            raw_data = {"train": cifar100_train, "val": cifar100_val, "test": cifar100_test}
        return raw_data


class STL10(Task):
    def __init__(self, name, args, pretrain=False, fold=0):
        super().__init__(name, args, pretrain)
        self.fold = fold

    def _get_transforms(self):
        flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = transforms.RandomResizedCrop(
            64, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3
        )
        center_crop = transforms.Compose(
            [transforms.Resize(70, interpolation=3), transforms.CenterCrop(64)]
        )
        if self.pretrain:
            train_transform = eval_transform = {
                "idx": DupTransform(self.args.dup_pos),
                "image": DupTransform(
                    self.args.dup_pos,
                    transforms.Compose(
                        [
                            rand_crop,
                            col_jitter,
                            rnd_gray,
                            transforms.ToTensor(),
                            normalize,
                            ToPatches(self.args.num_patches),
                        ]
                    ),
                ),
                "query": DupTransform(
                    self.args.dup_pos, RandZero(self.args.num_patches, self.args.num_queries)
                ),
            }
        else:
            train_transform = {
                "image": transforms.Compose(
                    [
                        flip_lr,
                        rand_crop,
                        col_jitter,
                        rnd_gray,
                        transforms.ToTensor(),
                        normalize,
                        ToPatches(self.args.num_patches),
                    ]
                ),
            }
            eval_transform = {
                "image": transforms.Compose(
                    [
                        center_crop,
                        transforms.ToTensor(),
                        normalize,
                        ToPatches(self.args.num_patches),
                    ]
                ),
            }
        return train_transform, eval_transform

    def _load_raw_data(self):
        stl10_unlabeled = datasets.STL10(root=self.path, split="unlabeled", download=True)
        if self.pretrain:
            stl10_unlabeled_f = datasets.STL10(
                root=self.path,
                split="unlabeled",
                transform=transforms.RandomHorizontalFlip(p=1.0),
                download=True,
            )
            raw_data = {"train": stl10_unlabeled + stl10_unlabeled_f}
        else:
            stl10_train = datasets.STL10(
                root=self.path, split="train", folds=self.fold, download=True
            )
            stl10_test = datasets.STL10(root=self.path, split="test", download=True)
            stl10_train, stl10_val = self.make_data_split(stl10_train)
            raw_data = {"train": stl10_train, "val": stl10_val, "test": stl10_test}
        return raw_data


class MNIST(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    def _get_transforms(self):
        return train_transform, eval_transform

    def _load_raw_data(self):
        mnist_train = datasets.MNIST(root=self.path, train=True, download=True)
        if self.pretrain:
            mnist_train_f = datasets.MNIST(
                root=self.path,
                train=True,
                transform=transforms.RandomHorizontalFlip(p=1.0),
                download=True,
            )
            raw_data = {"train": mnist_train + mnist_train_f}
        else:
            mnist_test = datasets.MNIST(root=self.path, train=False, download=True)
            mnist_train, mnist_val = self.make_data_split(mnist_train, self.label_pct)
            raw_data = {"train": mnist_train, "val": mnist_val, "test": mnist_test}
        return raw_data


class ImageNet(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    @staticmethod
    def _get_transforms():
        raise NotImplementedError
        train_transform = eval_transform = None
        return train_transform, eval_transform

    def _load_raw_data(self):
        raise NotImplementedError
        raw_data = None
        return raw_data
