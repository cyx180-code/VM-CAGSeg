import torch
import torch.nn as nn

from .vmamba import VSSM, PatchEmbed2D
from . import bottlenecks as NK
from . import vepe as PE
from . import attentions as AM
from . import common as C
# from .common import *
from . import multiscalefeaturefusion as MSFF


class CAGSegVSSM(VSSM):
    def __init__(
            self, patch_size=4, in_chans=3, num_classes=1000, 
            depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
            dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], 
            d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, patch_norm=True,
            use_checkpoint=False, **kwargs
        ):
        super(CAGSegVSSM, self).__init__(
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes, 
            depths=depths, 
            depths_decoder=depths_decoder,
            dims=dims, 
            dims_decoder=dims_decoder, 
            d_state=d_state, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, 
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, 
            **kwargs
        )

        print("CAGSegVSSM", kwargs)
        """patch embed"""
        self.create_patch_embed(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim, 
            norm_layer=norm_layer if patch_norm else None, 
            **kwargs
        )
        
        """bottleneck"""
        self.create_bottleneck(**kwargs)
        
        """skip connection"""
        self.create_skip_connection(dims=dims_decoder, **kwargs)

        self.create_multi_scale_vessel_enhancement_module(
            dims=dims, 
            norm_layer=norm_layer,
            channel_first=False,
            **kwargs
        )

        self.create_msff(dims_decoder=dims_decoder, **kwargs)

    def create_patch_embed(
            self, 
            patch_size=4, 
            in_chans=3, 
            embed_dim=96, 
            norm_layer=None, 
            **kwargs
        ):
        vepe = kwargs.get('vepe', "no")
        att_ver = kwargs.get('vepe_att_ver', "sea")
        _patch_embed = PE.get_patch_embeding(vepe)
        self.patch_embed = _patch_embed(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            norm_layer=norm_layer, 
            att_ver=att_ver
        )
    
    def create_bottleneck(self, **kwargs):
        neck_ver = kwargs.get('neck_ver', 'no')
        _bottleneck = NK.get_neck(neck_ver)
        if _bottleneck is not None:
            self.bottleneck = nn.Sequential(
            C.Permute(0, 3, 1, 2),
            _bottleneck(self.num_features, self.num_features),
            C.Permute(0, 2, 3, 1),
        )
        else:
            self.bottleneck = None
    
    def create_skip_connection(self, dims=[768, 384, 192, 96], **kwargs):        
        skip_conn_ver = kwargs.get('skip_conn_ver', "no")
        _skip_conn = AM.get_attention(skip_conn_ver)

        self.enable_skip_conn = _skip_conn is not None
        if self.enable_skip_conn:
            self.skip_connections = nn.ModuleList()
            for i_layer in range(self.num_layers - 1):
                conn = nn.Sequential(
                    C.Permute(0, 3, 1, 2),
                    _skip_conn(channel=dims[i_layer]),
                    C.Permute(0, 2, 3, 1),
                )
                self.skip_connections.append(conn)
    
    def create_msff(self, dims_decoder=[768, 384, 192, 96], **kwargs):
        msff_ver = kwargs.get('msff', 'no')
        msff_att_ver = kwargs.get('msff_att_ver', 'ca')
        if -1 != msff_ver.find("csif"):
            self.skip_conn_fusion = True
            self.enable_msff = True
            msff_neck_ver = kwargs.get('msff_neck_ver', 'spm')
            _create_msff = MSFF.get_msff(msff_ver)
            self.msff = _create_msff(
                in_dim=dims_decoder[0], 
                out_dim=dims_decoder[0], 
                msf_dims=dims_decoder[:-1],
                att_ver=msff_att_ver,
                neck_ver=msff_neck_ver,
            )
        else:
            self.enable_msff = False
    
    def forward_features_up(self, x, skip_list):
        interaction_fused_feas = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                if self.enable_msff:
                    msf = skip_list[1:]
                    msf.reverse()
                    interaction_fused_feas = self.msff(x, msf)
                    x = interaction_fused_feas[0]
                elif self.bottleneck is not None:
                    x = self.bottleneck(x).contiguous()
                x = layer_up(x)
            else:
                x_skip = skip_list[-inx]
                if self.enable_msff and self.skip_conn_fusion:
                    x_skip = interaction_fused_feas[inx]
                elif self.enable_skip_conn:
                    x_skip = self.skip_connections[inx-1](x_skip).contiguous()
                x = layer_up(x+x_skip)
        return x
    

    def create_multi_scale_vessel_enhancement_module(
            self, 
            dims=[96, 192, 384, 768], 
            norm_layer=nn.LayerNorm, 
            channel_first=False, 
            **kwargs
        ):
        msve_ver = kwargs.get('msve', "no")
        self.num_msve_layers = int(kwargs.get('msve_layer', "2"))
        if msve_ver != 'no':
            self.msvems =  nn.ModuleList()
            _create_msvem = MSVE.get_msve(msve_ver)
            for i_layer in range(self.num_msve_layers):
                msvem = _create_msvem(
                    in_dim=3,
                    out_dim=dims[i_layer],
                    ve_dim=96,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    att_ver='sea',
                    ratio='scale_1_2' if (i_layer !=0) else 'scale_1_4'
                )
                self.msvems.append(msvem)
        else:
            self.msvems = None
    
    def forward_features(self, x):
        skip_list = []

        """downsample"""
        all_ratios = [4, 8, 16]
        x_multiscale = C.downsample(x, ratios=[all_ratios[i] for i in range(self.num_msve_layers - 1)])

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            skip_list.append(x)

            """multi scale Vessel Enhancement module"""
            if self.msvems is not None and idx < self.num_msve_layers:
                x = self.msvems[idx](x_multiscale[idx], x)

            x = layer(x)
        return x, skip_list



class CAGSegMamba(nn.Module):
    def __init__(
            self, 
            input_channels=3, 
            num_classes=1,
            depths=[2, 2, 9, 2], 
            depths_decoder=[2, 9, 2, 2],
            drop_path_rate=0.2,
            load_ckpt_path=None,
            **kwargs
        ):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.cagsegnet = CAGSegVSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
            **kwargs
        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.cagsegnet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self, logger = None):
        if self.load_ckpt_path is not None:
            model_dict = self.cagsegnet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            logs = 'Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict))
            print(logs)
            if logger:
                logger.info(logs)
            self.cagsegnet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            logs = f'Not loaded keys: {not_loaded_keys}'
            print(logs)
            if logger:
                logger.info(logs)
            print("encoder loaded finished!")
            logs = 'encoder loaded finished!'
            if logger:
                logger.info(logs)

            model_dict = self.cagsegnet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']     
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            logs = 'Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict))
            print(logs)
            if logger:
                logger.info(logs)
            self.cagsegnet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            logs = f'Not loaded keys: {not_loaded_keys}'
            print(logs)
            if logger:
                logger.info(logs)
            print("decoder loaded finished!")
            logs = 'decoder loaded finished!'
            if logger:
                logger.info(logs)