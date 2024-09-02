import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageModel(nn.Module):
    def __init__(
        self,
        timm_params,
    ):
        super().__init__()
        self.model = timm.create_model(**timm_params)

        self.model_name = timm_params["model_name"]
        if self.model_name == "tf_efficientnet_b0.ns_jft_in1k":
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            self.pooling = GeM()
            self.linear = nn.Linear(in_features, 1)
        elif self.model_name == "eva02_small_patch14_336.mim_in22k_ft_in1k":
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, 1)
        elif self.model_name == "swin_base_patch4_window7_224.ms_in22k_ft_in1k":
        # elif self.model_name == "swin_base_patch4_window12_384.ms_in22k":
            self.model.head.fc = nn.Linear(1024, 1)
        # elif self.model_name == "convnextv2_base.fcmae_ft_in22k_in1k_384":
        elif self.model_name == "convnext_base.fb_in22k_ft_in1k":
            self.model.head.fc = nn.Linear(1024, 1)
        elif self.model_name == "seresnextaa101d_32x8d.sw_in12k_ft_in1k_288":
            self.model.fc = nn.Linear(2048, 1)
        elif self.model_name == "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k": # "maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k":
            self.model.head = nn.Linear(512, 1)
        elif self.model_name == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k":  # "coatnet_rmlp_2_rw_224.sw_in1k":
            self.model.head.fc = nn.Linear(1024, 1)
        elif self.model_name == "twins_svt_large.in1k":
            self.model.head = nn.Linear(1024, 1)
        elif self.model_name == "hiera_base_224.mae_in1k_ft_in1k":
            self.model.head.fc = nn.Linear(768, 1)
        else:
            raise Exception(f"unknown model: {self.model_name}")

        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        if self.model_name == "tf_efficientnet_b0.ns_jft_in1k":
            features = self.model(images)
            pooled_features = self.pooling(features).flatten(1)
            return self.sigmoid(self.linear(pooled_features))
        elif self.model_name == "eva02_small_patch14_336.mim_in22k_ft_in1k":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "swin_base_patch4_window7_224.ms_in22k_ft_in1k":
        # elif self.model_name == "swin_base_patch4_window12_384.ms_in22k":
            output = self.sigmoid(self.model(images))
            return output
        # elif self.model_name == "convnextv2_base.fcmae_ft_in22k_in1k_384":
        elif self.model_name == "convnext_base.fb_in22k_ft_in1k":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "seresnextaa101d_32x8d.sw_in12k_ft_in1k_288":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k": #"maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k":  #"coatnet_rmlp_2_rw_224.sw_in1k":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "twins_svt_large.in1k":
            output = self.sigmoid(self.model(images))
            return output
        elif self.model_name == "hiera_base_224.mae_in1k_ft_in1k":
            output = self.sigmoid(self.model(images))
            return output
        else:
            raise Exception(f"unknown model: {self.model_name}")


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'
