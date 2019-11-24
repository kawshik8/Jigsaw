# This file implement dataset and tasks.
# dataset loads and preprocess data
# task manages data, create transformerd version, create dataloader, loss and evaluation metrics
import logging as log
import os
import torch
import json
import numpy
from torchvision import datasets, transforms


def get_task(name, args):

    if name == "stl10-un":
        return STL10(name, args, pretrain=True)
    if name.startswith("stl10-fd"):
        return STL10(name, args, fold=int(name.replace("stl10-fd", "")))
    elif name == "cifar10-un":
        return CIFAR10(name, args, pretrain=True)
    elif name.startswith("cifar10-lp"):
        return CIFAR10(name, args, label_pct=float(name.replace("cifar10-lp", "")) / 100)
    elif name == "cifar100-un":
        return CIFAR100(name, args, pretrain=True)
    elif name.startswith("cifar100-lp"):
        return CIFAR100(name, args, label_pct=float(name.replace("cifar100-lp", "")) / 100)
    elif name.startswith("mnist-un"):
        return MNIST(name, args, pretrain=True)
    elif name.startswith("mnist-lp"):
        return MNIST(name, args, label_pct=float(name.replace("mnist-lp", "")) / 100)
    elif name.startswith("imagenet-un"):
        return ImageNet(name, args, pretrain=True)
    elif name.startswith("imagenet-lp"):
        return ImageNet(name, args, label_pct=float(name.replace("imagenet-lp", "")) / 100)
    else:
        raise NotImplementedError


def task_num_class(name):
    if name.startswith("imagenet"):
        return 1000
    elif name.startswith("cifar100"):
        return 100
    else:
        return 10


class TransformDataset(torch.utils.data.dataset):
    def __init__(self, transform, tensors):
        super().__init__()
        self.transform = transform
        self.tensors = tensors
        for key in self.tensors:
            if key not in self.transform:
                self.transform[key] = lambda x: x

    def __getitem__(self, index):
        return {key: self.transform[key](tensor[index]) for key, tensor in self.tensors.items()}

    def __len__(self):
        return self.tensors.values[0].size(0)


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
        self.path = os.path.join(args.data_dir, self.name.split("-")[0])
        if pretrain:
            self.eval_metric = "jigsaw_acc"
        else:
            self.eval_metric = "cls_acc"

    @staticmethod
    def _get_transforms():
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

    def make_data_split(self, train_data, pct=1.0):
        split_filename = os.path.join(self.path, "%s.json" % self.name)
        if os.path.exist(split_filename):
            with open(split_filename, "r") as f:
                split = json.loads(f.read(split_filename))
        else:
            full_size = len(train_data)
            train_size = int(full_size * pct * 0.9)
            val_size = int(full_size * pct * 0.1) + train_size
            full_idx = numpy.random.permutation(full_size)
            split = {"train": full_idx[:train_size], "val": full_idx[train_size:val_size]}
            with open(split_filename, "w") as f:
                f.write(json.dumps(split))
        train_data = [train_data[idx] for idx in split["train"]]
        val_data = [train_data[idx] for idx in split["val"]]
        return train_data, val_data

    def load_data(self):
        """
        load data, create data iterators. use cached data when available.
        """
        log.info("Loading %s data" % self.name)
        data = self._load_raw_data()

        train_transform, eval_transform = self._get_transforms()
        if self.pretrain:
            data["train"] = TransformDataset(train_transform, data["train"])
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

    def updata_scorers(self, batch_input, batch_output):
        count = len(batch_input["ids"])
        self.scorers["count"] += count
        for key in self.scorers.keys():
            if key != "count":
                self.scorers[key].append(batch_output[key] * count)

    def report_scorers(self, reset=False):
        avg_scores = {
            key: sum(value) / self.scorers["count"]
            for key, value in self.scorers.items()
            if key != "count" and value != []
        }
        if reset:
            self.reset_scorers()
        return avg_scores


class CIFAR10(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    @staticmethod
    def _get_transforms():
        return train_transform, eval_transform

    def _load_raw_data(self):
        cifar10_train = datasets.CIFAR10(root=self.path, train=True, download=True)
        if self.pretrain:
            cifar10_train_f = datasets.CIFAR10(
                root=self.path,
                train=True,
                transform=transforms.RandomHorizontalFlip(p=1.0),
                download=True,
            )
            raw_data = {"train": cifar10_train + cifar10_train_f}
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
            cifar100_train_f = datasets.CIFAR100(
                root=self.path,
                train=True,
                transform=transforms.RandomHorizontalFlip(p=1.0),
                download=True,
            )
            raw_data = {"train": cifar100_train + cifar100_train_f}
        else:
            cifar100_test = datasets.CIFAR100(root=self.path, train=False, download=True)
            cifar100_train, cifar100_val = self.make_data_split(cifar100_train, self.label_pct)
            raw_data = {"train": cifar100_train, "val": cifar100_val, "test": cifar100_test}
        return raw_data


class STL10(Task):
    def __init__(self, name, args, pretrain=False, fold=0):
        super().__init__(name, args, pretrain)
        self.fold = fold

    @staticmethod
    def _get_transforms():
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

    @staticmethod
    def _get_transforms():
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
