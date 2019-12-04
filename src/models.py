# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet as resnet
import itertools

def get_model(name, args):
    if name == "selfie":
        return SelfieModel(args)
    elif name == "selfie1":
        return SelfieModel_revised(args)
    elif name == "Allp":
        return AllPatchModel(args)
    elif name == "Exp":
        return ExchangePatchModel(args)
    elif name == "baseline":
        return BaselineModel(args)
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
            #print(self.args.pretrain_learning_rate)
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                anneal_strategy="cos",
                total_steps=self.args.pretrain_total_iters,
                pct_start=self.args.warmup_iters / self.args.pretrain_total_iters,
                cycle_momentum=False,
                max_lr=self.args.pretrain_learning_rate,
            )
        elif stage == "finetune":
            param_groups = [
                {
                    "params": self.finetune_params,
                    "max_lr": self.args.finetune_learning_rate,
                    "weight_decay": self.args.finetune_weight_decay,
                }
            ]
            if self.args.transfer_paradigm == "tunable":
                param_groups.append(
                    {
                        "params": self.shared_params,
                        "max_lr": self.args.finetune_learning_rate,
                        "weight_decay": self.args.finetune_weight_decay,
                    }
                )
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                anneal_strategy="cos",
                total_steps=self.args.finetune_total_iters,
                pct_start=self.args.warmup_iters / self.args.finetune_total_iters,
                cycle_momentum=False,
                max_lr=self.args.finetune_learning_rate,
            )

        return optimizer, scheduler


def masked_select(inp, mask):
    return inp.flatten(0, len(mask.size()) - 1)[mask.flatten().nonzero()[:, 0]]

class BaselineModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.d_model = 1024

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

        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Linear(self.d_model, task_num_class(taskname))

        self.avg_pool = nn.AvgPool1d(self.num_patches)
        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.patch_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.cls_classifiers.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        inp = batch_input["image"]

        device = inp.device#batch_input["aug"].device
        bs = inp.size(0)

        patches = self.patch_network(inp.flatten(0, 1)).view(
            bs, self.num_patches, -1
        )  # (bs, num_patches, d_model)
        # pool = self.attention_pooling(patches)# (bs, aug_patches, d_model)
        final = self.avg_pool(patches.transpose(1,2)).view(bs,self.d_model) # (bs, d_model)
        #print(final.shape)
        cls_pred = self.cls_classifiers[task.name](final)
        batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
        batch_output["predict"] = cls_pred.max(dim=1)[1]
        batch_output["cls_acc"] = (batch_output["predict"] == batch_input["label"]).float().mean()

        return batch_output


class SelfieModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries

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

        self.d_model = 1024
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32)
        layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Linear(self.d_model, task_num_class(taskname))

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
            query_patch = masked_select(patches, batch_input["query"]).view(
                bs, self.num_queries, self.d_model
            )  # (bs, num_queries, d_model)
            visible_patch = (
                masked_select(patches, ~batch_input["query"])
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
            )/(self.d_model**(1/2.0))  # (bs, num_queries, num_queries)
            #print(similarity[0])
            jigsaw_pred = F.log_softmax(similarity, 2).flatten(
                0, 1
            )  # (bs * num_queries, num_queries)
            jigsaw_label = (
                torch.arange(0, self.num_queries, device=device).repeat(bs).long()
            )  # (bs * num_queries)
            batch_output["loss"] = F.nll_loss(jigsaw_pred, jigsaw_label)
            batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=1)[1] == jigsaw_label).float().mean()

        elif self.stage == "finetune":
            hidden = self.attention_pooling(patches).mean(dim=1)
            cls_pred = self.cls_classifiers[task.name](hidden)
            batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
            batch_output["predict"] = cls_pred.max(dim=1)[1]
            batch_output["cls_acc"] = (
                (batch_output["predict"] == batch_input["label"]).float().mean()
            )
        return batch_output

class SelfieModel_revised(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        full_resnet = resnet.resnet50()
        self.patch_network = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
            full_resnet.layer1,
            full_resnet.layer2,
            full_resnet.layer3,
            self.avg_pool,
        )

        self.d_model = 1024

        self.attention_pool_u0 = nn.Parameter(torch.rand(size = (self.args.batch_size, self.d_model), dtype = torch.float, requires_grad=True))

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32, dropout=0.1, dim_feedforward=640, activation='gelu')
        layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Linear(self.d_model, task_num_class(taskname))

        self.shared_params = list(self.patch_network.parameters())
        self.shared_params += list(self.attention_pooling.parameters())
        self.shared_params += [self.attention_pool_u0]
        self.pretrain_params = list(self.position_embedding.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        device = batch_input["image"].device
        bs = batch_input["image"].size(0)
        self.attention_pool_u0 = self.attention_pool_u0.to(device)
        patches = self.patch_network(batch_input["image"].flatten(0, 1)).view(
            bs, self.num_patches, -1
        )

        if self.stage == "pretrain":
            query_patch = masked_select(patches, batch_input["query"]).view(
                bs, self.num_queries, self.d_model
            )  # (bs, num_queries, d_model)
            visible_patch = (
                masked_select(patches, ~batch_input["query"])
                .view(bs, 1, self.num_context, self.d_model)
                .repeat(1, self.num_queries, 1, 1)
                .flatten(0, 1)
            )  # (bs * num_queries, num_context, d_model)
            pos_embeddings = self.position_embedding(
                torch.nonzero(batch_input["query"])[:, 1]
            ).view(
                bs, self.num_queries, self.d_model
            ) # (bs, num_queries, d_model)
            u0 = self.attention_pool_u0.view(bs,1,1,self.d_model).repeat(1,self.num_queries,1,1).flatten(0,1) # (bs * num_queries, 1, d_model)
            global_vector = self.attention_pooling(
                torch.cat([u0, visible_patch], dim=1)
            )[:, 0, :].view_as(
                query_patch
            )  # (bs, num_queries, d_model)

            query_return = global_vector + pos_embeddings

            similarity = torch.bmm(
                query_patch, query_return.transpose(1, 2)
            )/(self.d_model**(1/2.0))  # (bs, num_queries, num_queries)
            #print(similarity[0])
            jigsaw_pred = F.log_softmax(similarity, 2).flatten(
                0, 1
            )  # (bs * num_queries, num_queries)
            jigsaw_label = (
                torch.arange(0, self.num_queries, device=device).repeat(bs).long()
            )  # (bs * num_queries)
            batch_output["loss"] = F.nll_loss(jigsaw_pred, jigsaw_label)
            batch_output["jigsaw_acc"] = (jigsaw_pred.max(dim=1)[1] == jigsaw_label).float().mean()

        elif self.stage == "finetune":
            u0 = self.attention_pool_u0.view(
                bs,1,self.d_model
            ) # (bs, 1, d_model)
            hidden = self.attention_pooling(torch.cat([u0, patches], dim=1))[:,0,:]
            cls_pred = self.cls_classifiers[task.name](hidden)
            batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
            batch_output["predict"] = cls_pred.max(dim=1)[1]
            batch_output["cls_acc"] = (
                (batch_output["predict"] == batch_input["label"]).float().mean()
            )
        return batch_output

class AllPatchModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.dup_pos = args.dup_pos
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries
        self.d_model = 1024

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

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32)
        layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=3, norm=layer_norm
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Linear(self.d_model, task_num_class(taskname))

        self.avg_pool = nn.AvgPool1d(self.num_patches)
        self.sigmoid = nn.Sigmoid()
        self.shared_params = list(self.patch_network.parameters())
        self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.sigmoid.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}
        
        inp = batch_input["image"]

        device = inp.device#batch_input["aug"].device
        bs = inp.size(0)

        patches = self.patch_network(inp.flatten(0, 1)).view(
            bs, self.num_patches, -1
        )  # (bs, num_patches, d_model)
        attn_pool = self.attention_pooling(patches)# (bs, aug_patches, d_model)
        final = self.avg_pool(attn_pool.transpose(1,2)).view(bs,self.d_model) # (bs, d_model)

        if self.stage == "pretrain":
            
            similarity = torch.mm(
                final, final.transpose(0,1)
            )/(self.d_model**(1/2.0))  # (bs, bs)
            
            jigsaw_pred = self.sigmoid(similarity) # (bs , bs)

            jigsaw_label = torch.zeros(size=(bs,bs),dtype=torch.float).to(device)
            for i in range(bs):
                
                indices = torch.arange(int((i/self.dup_pos))*self.dup_pos,int(((i/self.dup_pos))+1)*self.dup_pos).type(torch.long).to(device)
                #### Creates an array of size self.dup_pos_patches 
                jigsaw_label[i] = jigsaw_label[i].scatter_(dim=0, index=indices, value=1.)
                #### Makes the indices of jigsaw_labels (array of zeros) 1 based on the labels in indices

            batch_output["loss"] = F.binary_cross_entropy(jigsaw_pred, jigsaw_label)#F.cross_entropy(jigsaw_pred, jigsaw_label)
            jigsaw_pred1 = jigsaw_pred.clone()
            jigsaw_pred1[jigsaw_pred>0.5] = 1
            jigsaw_pred1[jigsaw_pred<=0.5] = 0
            batch_output["jigsaw_acc"] = ((jigsaw_pred1) == jigsaw_label).float().mean()

        elif self.stage == "finetune":
            
            cls_pred = self.cls_classifiers[task.name](final)
            batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
            batch_output["predict"] = cls_pred.max(dim=1)[1]
            batch_output["cls_acc"] = (batch_output["predict"] == batch_input["label"]).float().mean()
        return batch_output

class ExchangePatchModel(JigsawModel):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.dup_pos = args.dup_pos
        self.num_patches = args.num_patches
        self.num_queries = args.num_queries
        self.num_context = self.num_patches - self.num_queries
        self.f1 = 256
        self.f2 = 1024
        self.f3 = 512
        self.d_model = 1024

        full_resnet = resnet.resnet50()
        self.initial_layers = nn.Sequential(
            full_resnet.conv1,
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool,
        )

        transformer_layer1 = nn.TransformerEncoderLayer(d_model=self.f1, nhead=32)
        layer_norm1 = nn.LayerNorm(normalized_shape=self.f1)
        self.attention_exchange1 = nn.TransformerEncoder(
            encoder_layer=transformer_layer1, num_layers=3, norm=layer_norm1
        )

        transformer_layer2 = nn.TransformerEncoderLayer(d_model=self.f2, nhead=32)
        layer_norm2 = nn.LayerNorm(normalized_shape=self.f2)
        self.attention_exchange2 = nn.TransformerEncoder(
            encoder_layer=transformer_layer2, num_layers=3, norm=layer_norm2
        )

        transformer_layer3 = nn.TransformerEncoderLayer(d_model=self.f3, nhead=32)
        layer_norm3 = nn.LayerNorm(normalized_shape=self.f3)
        self.attention_exchange3 = nn.TransformerEncoder(
            encoder_layer=transformer_layer3, num_layers=3, norm=layer_norm3
        )

        self.res_block1  =  full_resnet.layer1
        self.res_block2  =  full_resnet.layer2
        self.res_block3  =  full_resnet.layer3

        final_transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=32)
        final_layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attention_pooling = nn.TransformerEncoder(
            encoder_layer=final_transformer_layer, num_layers=3, norm=final_layer_norm
        )

        #self.position_embedding = nn.Embedding(self.num_patches, self.d_model)
        self.cls_classifiers = nn.ModuleDict()

        from tasks import task_num_class

        for taskname in args.finetune_tasks:
            self.cls_classifiers[taskname] = nn.Linear(self.d_model, task_num_class(taskname))

        self.avg_pool = nn.AvgPool1d(self.num_patches)

        self.Exchange_network = nn.Sequential(
            self.initial_layers,
            self.attention_exchange1,
            self.res_block1,
            self.attention_exchange2,
            self.res_block2,
            self.attention_exchange3,
            self.res_block3,
            self.attention_pooling,
        )
        self.sigmoid = nn.Sigmoid()
        
        self.shared_params = list(self.Exchange_network.parameters())
        #self.shared_params += list(self.attention_pooling.parameters())
        self.pretrain_params = list(self.avg_pool.parameters())
        self.finetune_params = list(self.cls_classifiers.parameters())

    def forward(self, batch_input, task=None):
        batch_output = {}

        inp = batch_input["image"]

        device = inp.device
        bs = inp.size(0)

        #print(inp.flatten(0, 1).size())
        input_attn1 = self.initial_layers(inp.flatten(0, 1)) # (bs, num_aug_patches, f1, h1, w1)
        #print(input_attn1.size())
        output_attn1 = self.attention_exchange1(input_attn1.view(bs,self.num_patches,-1)).view_as(input_attn1)

        input_attn2 = self.res_block1(output_attn1)# (bs, num_aug_patches, f2, h2, w2)
        #print(input_attn2.size())
        output_attn2 = self.attention_exchange2(input_attn2.view(bs,self.num_patches,-1)).view_as(input_attn2) 

        input_attn3 = self.res_block2(output_attn2)# (bs, num_aug_patches, f3, h3, w3)
        #print(input_attn3.size())
        output_attn3 = self.attention_exchange3(input_attn3.view(bs,self.num_patches,-1)).view_as(input_attn3)

        input_attn_pool = self.res_block3(output_attn3)# (bs, num_aug_patches, d_model)
        output_attn_pool = self.attention_pooling(input_attn_pool.view(bs,self.num_patches,-1)).view_as(input_attn_pool)
        final = self.avg_pool(output_attn_pool.view(bs,self.num_patches,-1).transpose(1,2)).view(bs,self.d_model)
        
        if self.stage == "pretrain":

            similarity = torch.mm(
                final, final.transpose(0,1)
            )/(self.d_model**(1/2.0))  # (bs, bs)
            
            jigsaw_pred = self.sigmoid(similarity) # (bs , bs)

            jigsaw_label = torch.zeros(size=(bs,bs),dtype=torch.float).to(device)
            for i in range(bs):
                #print((i/self.dup_pos)*self.dup_pos,((i/self.dup_pos)+1)*self.dup_pos)
                indices = torch.arange(int((i/self.dup_pos))*self.dup_pos,int(((i/self.dup_pos))+1)*self.dup_pos).type(torch.long).to(device)
                #### Creates an array of size self.dup_pos_patches 
                jigsaw_label[i] = jigsaw_label[i].scatter_(dim=0, index=indices, value=1.)
                #### Makes the indices of jigsaw_labels (array of zeros) 1 based on the labels in indices

            batch_output["loss"] = F.binary_cross_entropy(jigsaw_pred, jigsaw_label)#F.cross_entropy(jigsaw_pred, jigsaw_label)
            jigsaw_pred1 = jigsaw_pred.clone()
            jigsaw_pred1[jigsaw_pred>0.5] = 1
            jigsaw_pred1[jigsaw_pred<=0.5] = 0
            batch_output["jigsaw_acc"] = ((jigsaw_pred1) == jigsaw_label).float().mean()

        elif self.stage == "finetune":
            #hidden = self.attention_pooling(patches)
            cls_pred = self.cls_classifiers[task.name](final)
            batch_output["loss"] = F.cross_entropy(cls_pred, batch_input["label"])
            batch_output["predict"] = cls_pred.max(dim=1)[1]
            batch_output["cls_acc"] = (batch_output["predict"] == batch_input["label"]).float().mean()
        return batch_output
