# This file implement dataset and tasks.
# dataset loads and preprocess data
# task manages data, create transformerd version, create dataloader, loss and evaluation metrics
import logging as log
import os
import torch
import torchvision as V


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
            raw_data: dict[str, dict[str, tensor]]: from split to tensors of each field
                "idx": long (*), index of the image instance
                "image": float (*, channels, height, width), pixels from image
                "label": long (*), class label of the image, set to 0 when unavailable
        """
        raise NotImplementedError

    def _preprocess_data(self, raw_data):
        """
        inputs:
            raw_data
        outputs:
            preproc_data: dict[str, dict[str, tensor]]: from split to tensors of each field
                "idx": long (bs), index of the image instance
                "image": float (bs, num_patches, channels, height, width), pixels from raw and
                transformed image
                "query": bool (bs, num_patches), which patches are queried, only in pretrain
                "label": long (bs), class label of the image, only in fine-tune
                (if cfgs.dup_pos > 0, each image instance in minibatch will have (1 + dup_pos)
                transformed versions.)
        """
        raise NotImplementedError

    def load_data(self):
        """
        load and preprocess data, create data iterators. use cached data when available.
        """
        log.info("Loading %s data" % self.name)
        preprocessed_cache = os.path.join(
            self.args.exp_dir, "data_cache", "%s_preproc.data" % self.name
        )
        if os.path.exist(preprocessed_cache):
            preproc_data = torch.load(preprocessed_cache)
        else:
            raw_cache = os.path.join(self.args.exp_dir, "data_cache", "%s_raw.data" % self.name)
            if os.path.exist(raw_cache):
                raw_data = torch.load(raw_cache)
            else:
                raw_data = self._load_raw_data()
            preproc_data = self._preprocess_data(raw_data)

        train_transform, eval_transform = self._get_transforms()
        if self.pretrain:
            preproc_data["train"] = TransformDataset(train_transform, preproc_data["train"])
        else:
            preproc_data["train"] = TransformDataset(train_transform, preproc_data["train"])
            preproc_data["val"] = TransformDataset(eval_transform, preproc_data["val"])
            preproc_data["test"] = TransformDataset(eval_transform, preproc_data["test"])

        for split, dataset in preproc_data.items():
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
        return raw_data

    def _preprocess_data(self, raw_data):
        return preproc_data


class CIFAR100(CIFAR10):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain, label_pct)

    def _load_raw_data(self):
        return raw_data


class STL10(Task):
    def __init__(self, name, args, pretrain=False, fold=0):
        super().__init__(name, args, pretrain)
        self.fold = fold

    @staticmethod
    def _get_transforms():
        return train_transform, eval_transform

    def _load_raw_data(self):
        return raw_data

    def _preprocess_data(self, raw_data):
        return preproc_data


class MNIST(Task):
    def __init__(self, name, args, pretrain=False, label_pct=0.0):
        super().__init__(name, args, pretrain)
        self.label_pct = label_pct

    @staticmethod
    def _get_transforms():
        return train_transform, eval_transform

    def _load_raw_data(self):
        return raw_data

    def _preprocess_data(self, raw_data):
        return preproc_data


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

    def _preprocess_data(self, raw_data):
        raise NotImplementedError
        return preproc_data
