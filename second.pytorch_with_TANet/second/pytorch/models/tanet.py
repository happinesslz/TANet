import torch
from torch import nn
from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
from second.pytorch.models.rpn import register_rpn
from torchplus.tools import change_default_args
from torchplus.nn import Empty, GroupNorm, Sequential
from second.pytorch.models.pointpillars import PFNLayer
import numpy as np

import yaml
from easydict import EasyDict as edict

filename = './configs/tanet/tanet.yaml'
with open(filename, 'r') as f:
    cfg = edict(yaml.load(f))
    if cfg.Dataset == 'Kitti':
        cfg.TA = cfg.KITTI_TA
    elif cfg.Dataset == 'Nuscenes':
        cfg.TA = cfg.NUSCENES_TA


# Point-wise attention for each voxel
class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )

    def forward(self, x):
        b, w, _ = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)
        out1 = self.fc(y).view(b, w, 1)
        return out1


# Channel-wise attention for each voxel
class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        b, _, c = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(b, c)
        y = self.fc(y).view(b, 1, c)
        return y


# Point-wise attention for each voxel
class PACALayer(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r):
        super(PACALayer, self).__init__()
        self.pa = PALayer(dim_pa,  dim_pa // reduction_r)
        self.ca = CALayer(dim_ca,  dim_ca // reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        pa_weight = self.pa(x)
        ca_weight = self.ca(x)
        paca_weight = torch.mul(pa_weight, ca_weight)
        paca_normal_weight = self.sig(paca_weight)
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight

# Voxel-wise attention for each voxel
class VALayer(nn.Module):
    def __init__(self, c_num, p_num):
        super(VALayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_num + 3, 1),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(p_num, 1),    ########################
            nn.ReLU(inplace=True)
        )

        self.sigmod = nn.Sigmoid()

    def forward(self, voxel_center, paca_feat):
        '''
        :param voxel_center: size (K,1,3)
        :param SACA_Feat: size (K,N,C)
        :return: voxel_attention_weight: size (K,1,1)
        '''
        voxel_center_repeat = voxel_center.repeat(1, paca_feat.shape[1], 1)
        # print(voxel_center_repeat.shape)
        voxel_feat_concat = torch.cat([paca_feat, voxel_center_repeat], dim=-1)  # K,N,C---> K,N,(C+3)

        feat_2 = self.fc1(voxel_feat_concat)  # K,N,(C+3)--->K,N,1
        feat_2 = feat_2.permute(0, 2, 1).contiguous()  # K,N,1--->K,1,N

        voxel_feat_concat = self.fc2(feat_2)  # K,1,N--->K,1,1

        voxel_attention_weight = self.sigmod(voxel_feat_concat)  # K,1,1

        return voxel_attention_weight



class VoxelFeature_TA(nn.Module):
    def __init__(self,dim_ca=cfg.TA.INPUT_C_DIM,dim_pa=cfg.TA.NUM_POINTS_IN_VOXEL,
                 reduction_r = cfg.TA.REDUCTION_R,boost_c_dim = cfg.TA.BOOST_C_DIM,
                 use_paca_weight = cfg.TA.USE_PACA_WEIGHT):
        super(VoxelFeature_TA, self).__init__()
        self.PACALayer1 = PACALayer(dim_ca=dim_ca, dim_pa=dim_pa, reduction_r=reduction_r)
        self.PACALayer2 = PACALayer(dim_ca=boost_c_dim, dim_pa=dim_pa, reduction_r=reduction_r)
        self.voxel_attention1 = VALayer(c_num=dim_ca, p_num=dim_pa)
        self.voxel_attention2 = VALayer(c_num=boost_c_dim, p_num=dim_pa)
        self.use_paca_weight = use_paca_weight
        self.FC1 = nn.Sequential(
            nn.Linear(2*dim_ca, boost_c_dim),
            nn.ReLU(inplace=True),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(boost_c_dim, boost_c_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_center, x):
        paca1,paca_normal_weight1 = self.PACALayer1(x)
        voxel_attention1 = self.voxel_attention1(voxel_center, paca1)
        if self.use_paca_weight:
            paca1_feat = voxel_attention1 * paca1 * paca_normal_weight1
        else:
            paca1_feat = voxel_attention1 * paca1
        out1 = torch.cat([paca1_feat, x], dim=2)
        out1 = self.FC1(out1)

        paca2,paca_normal_weight2 = self.PACALayer2(out1)
        voxel_attention2 = self.voxel_attention2(voxel_center, paca2)
        if self.use_paca_weight:
            paca2_feat = voxel_attention2 * paca2 * paca_normal_weight2
        else:
            paca2_feat = voxel_attention2 * paca2
        out2 = out1 + paca2_feat
        out = self.FC2(out2)

        return out



## PillarFeature_TANet is modified from pointpillars.PillarFeatureNet
# by introducing Triple Attention
@register_vfe
class PillarFeature_TANet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net with Tripe attention.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super(PillarFeature_TANet, self).__init__()
        self.name = 'PillarFeature_TANet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_input_features = cfg.TA.BOOST_C_DIM

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)

        self.VoxelFeature_TA = VoxelFeature_TA()
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):

        # Find distance of x, y, and z from cluster center
        ## ref: https://github.com/traveller59/second.pytorch/issues/144
        num_voxels_set_0_to_1 = num_voxels.clone()
        num_voxels_set_0_to_1[num_voxels_set_0_to_1 == 0] = 1
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels_set_0_to_1.type_as(features).view(-1, 1, 1)
        #points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)

        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  ## (N,60,9) ##(N,100,9)

        features = self.VoxelFeature_TA(points_mean, features)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()



###### PSA
class Two_RPNNoHeadBase_PSA(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(Two_RPNNoHeadBase_PSA, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        ### Refine:
        self.bottle_conv = nn.Conv2d(sum(num_upsample_filters), sum(num_upsample_filters)//3, 1)

        self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
        self.block1_dec4x = nn.MaxPool2d(kernel_size=4)   ### C=64

        self.block2_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=128

        self.block2_inc2x = ConvTranspose2d(num_filters[1], num_filters[0]//2, 2, stride=2)
        self.block3_inc2x = ConvTranspose2d(num_filters[2], num_filters[1]//2, 2, stride=2)


        self.block3_inc4x = ConvTranspose2d(num_filters[2], num_filters[0]//2, 4,stride=4)  #### C=32

        if upsample_strides[0] < 1:
            stride0 = np.round(1.0 / upsample_strides[0]).astype(np.int64)
            self.refine_up1 = Sequential(
                Conv2d(num_filters[0], num_upsample_filters[0], stride0,stride=stride0),
                BatchNorm2d(num_upsample_filters[0]),
                nn.ReLU(),
            )
        else:
            stride0 = np.round(upsample_strides[0]).astype(np.int64)
            self.refine_up1 = Sequential(
                ConvTranspose2d(num_filters[0], num_upsample_filters[0], stride0,
                                stride=stride0),
                BatchNorm2d(num_upsample_filters[0]),
                nn.ReLU(),
            )

        if upsample_strides[1]<1:
            stride1 = np.round(1.0 / upsample_strides[1]).astype(np.int64)
            self.refine_up2 = Sequential(
                Conv2d(num_filters[1], num_upsample_filters[1], stride1,stride=stride1),
                BatchNorm2d(num_upsample_filters[1]),
                nn.ReLU(),
            )

        else:
            stride1 =np.round(upsample_strides[1]).astype(np.int64)
            self.refine_up2 = Sequential(
                ConvTranspose2d(num_filters[1], num_upsample_filters[1], stride1,stride=stride1),
                BatchNorm2d(num_upsample_filters[1]),
                nn.ReLU(),
            )

        if upsample_strides[2] < 1:
            stride2 = np.round(1.0 / upsample_strides[2]).astype(np.int64)
            self.refine_up3 = Sequential(
                ConvTranspose2d(num_filters[2], num_upsample_filters[2], stride2,
                                stride=stride2),
                BatchNorm2d(num_upsample_filters[2]),
                nn.ReLU(),
            )

        else:
            stride2 = np.round(upsample_strides[2]).astype(np.int64)
            self.refine_up3 = Sequential(
                ConvTranspose2d(num_filters[2], num_upsample_filters[2], stride2,
                                stride=stride2),
                BatchNorm2d(num_upsample_filters[2]),
                nn.ReLU(),
            )

        self.fusion_block1 = nn.Conv2d(num_filters[0]+num_filters[0]//2+num_filters[0]//2, num_filters[0], 1)
        self.fusion_block2 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[1]//2, num_filters[1], 1)
        self.fusion_block3 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[2], num_filters[2], 1)


        #######
        C_Bottle = cfg.PSA.C_Bottle
        C = cfg.PSA.C_Reudce

        self.RF1 = Sequential(  # 3*3
            Conv2d(C_Bottle*2, C, kernel_size=1, stride=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle*2, kernel_size=3, stride=1, padding=1, dilation=1),
            BatchNorm2d(C_Bottle*2),
            nn.ReLU(inplace=True),
        )

        self.RF2 = Sequential(  # 5*5
            Conv2d(C_Bottle, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle, kernel_size=3, stride=1, padding=1, dilation=1),
            BatchNorm2d(C_Bottle),
            nn.ReLU(inplace=True),
        )

        self.RF3 = Sequential(  # 7*7
            Conv2d(C_Bottle//2, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C),
            nn.ReLU(inplace=True),
            Conv2d(C, C_Bottle//2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(C_Bottle//2),
            nn.ReLU(inplace=True),
        )

        self.concat_conv1 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)  ## kernel_size=3
        self.concat_conv2 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(num_filters[1], num_filters[1], kernel_size=3, padding=1)


    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x


        #####
        blottle_conv = self.bottle_conv(x)

        x1,x2,x3 = stage_outputs[0], stage_outputs[1], stage_outputs[2]
        up1,up2,up3 = ups[0], ups[1],ups[2]

        x1_dec2x = self.block1_dec2x(x1)
        x1_dec4x = self.block1_dec4x(x1)

        x2_dec2x = self.block2_dec2x(x2)
        x2_inc2x = self.block2_inc2x(x2)

        x3_inc2x = self.block3_inc2x(x3)
        x3_inc4x = self.block3_inc4x(x3)

        # print('x1,x2,x3:', x1.shape, x2.shape, x3.shape)
        # print('up1,up2,up3:', up1.shape, up2.shape, up3.shape)
        # print('x1_dec2x,x2_dec2x,x3_inc2x:', x1_dec2x.shape, x2_dec2x.shape, x3_inc2x.shape)
        # print('x1_dec4x,x2_inc2x,x3_inc4x:', x1_dec4x.shape, x2_inc2x.shape, x3_inc4x.shape)

        concat_block1 = torch.cat([x1,x2_inc2x,x3_inc4x], dim=1)
        fusion_block1 = self.fusion_block1(concat_block1)

        concat_block2 = torch.cat([x1_dec2x,x2,x3_inc2x], dim=1)
        fusion_block2 = self.fusion_block2(concat_block2)

        concat_block3 = torch.cat([x1_dec4x,x2_dec2x,x3], dim=1)
        fusion_block3 = self.fusion_block3(concat_block3)

        refine_up1 = self.RF3(fusion_block1)
        refine_up1 = self.refine_up1(refine_up1)
        refine_up2 = self.RF2(fusion_block2)
        refine_up2 = self.refine_up2(refine_up2)
        refine_up3 = self.RF1(fusion_block3)
        refine_up3 = self.refine_up3(refine_up3)


        branch1_sum_wise = refine_up1 + blottle_conv
        branch2_sum_wise = refine_up2 + blottle_conv
        branch3_sum_wise = refine_up3 + blottle_conv

        concat_conv1 = self.concat_conv1(branch1_sum_wise)
        concat_conv2 = self.concat_conv2(branch2_sum_wise)
        concat_conv3 = self.concat_conv3(branch3_sum_wise)

        PSA_output = torch.cat([concat_conv1,concat_conv2,concat_conv3], dim=1)
        res["refine_out"] = PSA_output
        return res





@register_rpn
class PSA(Two_RPNNoHeadBase_PSA):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(PSA, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)
            self.refine_conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

        self.refine_conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.refine_conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)

    def forward(self, x):
        res = super().forward(x)
        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds


        ##refine:
        refine_x = res["refine_out"]
        refine_box_preds = self.refine_conv_box(refine_x)
        refine_cls_preds = self.refine_conv_cls(refine_x)
        refine_box_preds = refine_box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        refine_cls_preds = refine_cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict["Refine_loc_preds"] = refine_box_preds
        ret_dict["Refine_cls_preds"] = refine_cls_preds

        if self._use_direction_classifier:
            refine_dir_cls_preds = self.refine_conv_dir_cls(x)
            refine_dir_cls_preds = refine_dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["Refine_dir_preds"] = refine_dir_cls_preds


        return ret_dict
