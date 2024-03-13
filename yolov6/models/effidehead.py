import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox
from yolov6.models.seghead import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag

from yolov6.models.clshead import ClassificationHead

class Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=10, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        super().__init__()
        print('Initializing Detect head...')
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        print('initial...')
        print('#'*80)
        self.spp1 = PAPPM(192, 96, 256)
#         self.seg_head1 = segmenthead(256, 128, 6)
        self.spp2 = PAPPM(384, 96, 256)
#         self.seg_head2 = segmenthead(256, 128, 6)
        self.spp3 = PAPPM(768, 96, 256)
        self.seg_head = segmenthead(256, 128, 20) # self.seg_head = segmenthead(256, 128, 6)

        # add cls branch
        self.cls1 = ClassificationHead(192 , self.nc)
        self.cls2 = ClassificationHead(384 , self.nc)
        self.cls3 = ClassificationHead(256 , self.nc)

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        


        # Efficient decoupled head layers
        for i in range(num_layers):
            # print("="*100)
            idx = i*5
            # print("idx : " , idx)
            self.stems.append(head_layers[idx])
            # print("*"*80)
            # print("head_layers[idx] : " , head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            # print("*"*80)
            # print("head_layers[idx+1] : " , head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            # print("*"*80)
            # print("head_layers[idx+2] : " , head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            # print("*"*80)
            # print("head_layers[idx+3] : " , head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])
            # print("*"*80)
            # print("head_layers[idx+4] : " , head_layers[idx+4])

        # assert False

        # print("="*100)
        # print("view the detail of head_layers")
        # print("num_layers : " , num_layers)
        # print("head_layers : " , type(head_layers) , "|" , len(head_layers) , "|" , head_layers)

        # print("="*100)
        # assert False

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        # det & seg : self.training == "True"
        if self.training:
            cls_score_list = []
            reg_distri_list = []
#             print(self.nl)

            # add clshead here
            # try :
            # for i in range(len(x)) :
            #     print(type(x[i]) , "|" , x[i].shape)
            # print(type(x) , len(x))
            # assert False
            # x2 = self.cls_head(x)
                # print("x2 ok")
            # except :
            #     x2 = None   
            #     print("x2 not ok") 
            
            # [8, 96, 80, 80]
            # [8, 192, 40, 40]
            # [8, 384, 40, 40]7654321
            
            # [8, 768, 10, 10]
            
            for i in range(self.nl):
                # print("shape of x[{}] : ".format(i) , x[i].shape)
                if i==1:
                    # print("spp1 + x[{i}]  : " , self.spp1(x[i]).shape)
                    # print("x[{i}] : " , x[i].shape , "\n" , x[i])
                    x1 = self.spp1(x[i])
                    # print("x1 after spp1 : " , x1.shape , "\n" , x1)
                    # print("@"*80)
                elif i==2:
                    # print("spp2 + x[{i}]  : " , self.spp2(x[i]).shape)
                    # print("x[{i}] : " , x[i].shape , "\n" , x[i])
                    # print("x1 before spp2 : " , x1.shape , "\n" , x1)
                    x1 = x1 + F.interpolate(self.spp2(x[i]), scale_factor=2, mode='bilinear',align_corners=True)
                    # print("interpolate after spp2 : " , x1.shape , "\n" , x1)
                    # print("@"*80)
                elif i==3:
                    # print("spp3 + x[{i}] : " , self.spp3(x[i]).shape)
                    # print("x[{i}] : " , x[i].shape , "\n" , x[i])
                    # print("x1 before spp3 : " , x1.shape , "\n" , x1)
                    x1 = x1 + F.interpolate(self.spp3(x[i]), scale_factor=4, mode='bilinear',align_corners=True)
                    # print("interpolate after spp3 : " , x1.shape , "\n" , x1)
                    # print("@"*80)
                    
                x[i] = self.stems[i](x[i])
#                 print(x[i].shape)
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)


                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            
            # assert False

            # print("@"*150)
            # print("after for loop :")
            # for num , item in enumerate(cls_score_list) :
            #     print("cls_score_list {} : ".format(num) , item.shape)
            # for num , item in enumerate(reg_distri_list) :
            #     print("reg_distri_list {} : ".format(num) , item.shape)
            # print("reg_distri_list : " , len(reg_distri_list) , "\n" , reg_distri_list)
            # print("@"*150)

            # assert False

            # print("="*100)
            # print("view the detail of decouple head")
            # print("self.stem : " , type(self.stems) , "|" , self.stems)
            # print("self.cls_convs : " , type(self.cls_convs) , "|" , self.cls_convs)
            # print("self.cls_preds : " , type(self.cls_preds) , "|" , self.cls_preds)
            # print("self.reg_convs : " , type(self.reg_convs) , "|" , self.reg_convs)
            # print("self.reg_preds : " , type(self.reg_preds) , "|" , self.reg_preds)

            # print("="*100)
            # assert False
            
#             import pdb
#             pdb.set_trace()
            x1 = self.seg_head(x1)
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            # print("after cat :")
            # print("cls_score_list : " , cls_score_list.shape)
            # print("cls_score_list[0][0] : " , cls_score_list[0][0])
            # print("reg_distri_list : " , reg_distri_list.shape)
            # print("@"*150)

            # assert False

            x2 = None

            # return x, cls_score_list, reg_distri_list, x1
            return x, cls_score_list, reg_distri_list, x1, x2
        
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')
#             print(self.nl)
            for i in range(self.nl):
                b, _, h, w = x[i].shape
#                 print(x[i].shape)
                if i==1:
                    x1 = self.spp1(x[i])  
                elif i==2:
                    x1 = x1 + F.interpolate(self.spp2(x[i]), scale_factor=2, mode='bilinear',align_corners=True)
                elif i==3:
                    x1 = x1 +  F.interpolate(self.spp3(x[i]), scale_factor=4, mode='bilinear',align_corners=True)
                    
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            x1 = self.seg_head(x1)
#             print(x1.shape)
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1), x1


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )

    return head_layers