# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.model.resnet as resnet


def get_model(name, args):
    if name == "selfie":
        return SelfieModel(args)
    else:
        raise NotImplementedError


class JigsawModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = None
        self.shared_params = []
        self.pretrain_params = []
        self.finetune_params = []

    def forward(self, batch_input, task=None):
        """
        inputs:
            batch_input: dict[str, tensor]: inputs in one minibatch
                "idx": long (bs), index of the image instance
                "image": float (bs, num_patches, channels, height, width), pixels from raw and
                transformed image
                "query": bool (bs, num_patches), which patches are queried, only in pretrain
                "label": long (bs), class label of the image, only in fine-tune
                (if cfgs.dup_pos > 0, each image instance in minibatch will have (1 + dup_pos)
                transformed versions.)
            task: task object
        outputs:
            batch_output: dict[str, tensor]: outputs in one minibatch
                "loss": float (1), full loss of a batch
                "loss_*": float (1), one term of the loss (if more than one SSL tasks are used)
                "jigsaw_acc": float (1), jigsaw puzzle accuracy, only when pretrain
                "cls_acc": float (1), classification accuracy, only when finetune
                "predict": float (bs), class prediction, only when finetune
        """
        raise NotImplementedError

    def config_stage(self, stage):
        """
        switch between pretrain and finetune stages
        inputs:
            stage: str, "pretrain" or "finetune"
        outputs:
            optimizer: ...
            scheduler: ...
        """
        self.stage = stage
        if stage == "pretrain":
            param_groups = [
                {
                    "params": self.shared_params + self.pretrain_params,
                    "max_lr": self.args.pretrain_learning_rate,
                    "weight_decay": self.args.pretrain_weight_decay,
                }
            ]
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                anneal_strategy="cosine",
                total_steps=self.args.pretrain_total_iters,
                pct_start=self.args.warmup / self.args.pretrain_total_iters,
                cycle_momentum=False,
            )
        elif stage == "finetune":
            param_groups = [
                {
                    "params": self.finetune_params,
                    "max_lr": self.args.pretrain_learning_rate,
                    "weight_decay": self.args.pretrain_weight_decay,
                }
            ]
            if self.args.transfer_paradigm in ["tunable", "bound"]:
                param_groups.append(
                    {
                        "params": self.shared_params,
                        "max_lr": self.args.finetune_learning_rate,
                        "weight_decay": self.args.finetune_weight_decay
                        if self.args.transfer_paradigm == "tunable"
                        else 0.0,
                    }
                )
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                anneal_strategy="cosine",
                total_steps=self.args.finetune_total_iters,
                pct_start=self.args.warmup / self.args.finetune_total_iters,
                cycle_momentum=False,
            )

        return optimizer, scheduler


class SelfieModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries
        self.d_model = 256

        full_resnet = resnet.resnet50()
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
        )

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        layer_norm = nn.LayerNorm(d_model=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()

        from task import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Sequential(
                nn.AvgPool1d(self.num_patches), nn.Linear(self.d_model, task_num_class(taskname))
            )

        self.shared_params = list(self.patch_network.parameters())
        self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.position_embedding.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        device = batch_input["image"].device
        bs = batch_input["image"].size(0)
        patches = self.patch_network(batch_input["image"].flatten(0, 1)).view(
            bs, self.num_patches, -1
        )  # (bs, num_patches, d_model)
        if self.stage == "pretrain":
            query_patch = torch.masked_select(patches, batch_input["query"].unsqueeze(2)).view(
                bs, self.num_queries, self.d_model
            )  # (bs, num_queries, d_model)
            visible_patch = (
                torch.masked_select(patches, batch_input["query"].unsqueeze(2) == 0)
                .view(bs, 1, self.num_context, self.d_model)
                .repeat(1, self.num_queries, 1, 1)
                .flatten(0, 1)
            )  # (bs * num_queries, num_context, d_model)
            pos_embeddings = self.position_embedding(
                torch.nonzero(batch_input["query"])[:, 1]
            ).unsqueeze(
                1
            )  # (bs * num_queries, 1, d_model)
            query_return = self.attention_pooling(
                torch.cat([pos_embeddings, visible_patch], dim=1)
            )[:, 0, :].view_as(
                query_patch
            )  # (bs, num_queries, d_model)
            similarity = torch.bmm(
                query_patch, query_return.transpose(1, 2)
            )  # (bs, num_queries, num_queries)
            jigsaw_pred = F.log_softmax(similarity, 2).flatten(
                0, 1
            )  # (bs * num_queries, num_queries)
            jigsaw_label = (
                torch.arange(0, self.num_queries, device=device).repeat(bs).long()
            )  # (bs * num_queries)
            batch_output["loss"] = F.nllloss(jigsaw_pred, jigsaw_label)
            batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=2) == jigsaw_label).mean()

        elif self.stage == "finetune":
            hidden = self.attention_pooling(patches)
            cls_pred = self.cls_classifier[task.name](hidden)
            batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
            batch_output["predict"] = cls_pred.max(dim=1)[1]
            batch_output["cls_acc"] = (batch_output["predict"] == batch_input["label"]).mean()
        return batch_output
