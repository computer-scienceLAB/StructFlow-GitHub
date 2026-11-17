import torch
import torch.nn as nn
import torch.nn.functional as F
from .warp import flow_warp
from .correlation import CorrTorch as Correlation #ROKIA, pure pytorch implementation of correlation instead of cuda
from .net_blocks import FeaturePyramidExtractor,  ContextNetwork, FeatureFusion, FlowEstimatorFullDeform, extract_gradient_map


class SharedCorrelation(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        super(SharedCorrelation, self).__init__()
        self.corr = Correlation(pad_size=pad_size, kernel_size=kernel_size,
                    max_displacement=max_displacement, stride1=stride1,
                    stride2=stride2, corr_multiply=corr_multiply)

    def forward(self, x1_pyramid, x2_pyramid, x_r_pyramid):
        corr_flow = self.corr(x1_pyramid[0], x2_pyramid[0])
        corr_depth = self.corr(x1_pyramid[0], x_r_pyramid[0])
        return torch.cat([corr_flow, corr_depth], dim=1)


class PWCFlowDepth(nn.Module):
    def __init__(self, lv_chs=(3, 16, 32, 64, 96, 128, 192), n_out=5, n_context=1, search_range=4, bn=False):#, use_structure_aware=False, use_context_aware=False
        super(PWCFlowDepth, self).__init__()
        
        
        self.lv_chs = lv_chs
        self.n_pyramid = len(lv_chs) - 1
        self.n_out = n_out
        self.n_context = n_context
        self.bn = bn
        
        self.feature_extractor = FeaturePyramidExtractor(self.lv_chs, bn=self.bn)#, use_structure_aware=use_structure_aware
        self.shared_corr = SharedCorrelation(pad_size=search_range, kernel_size=1, 
                    max_displacement=search_range, stride1=1, 
                    stride2=1, corr_multiply=1)
                    
        self.fusion_layers_flow = nn.ModuleList()
        self.fusion_layers_depth = nn.ModuleList()
        self.context_networks = nn.ModuleList()  # Added context networks
        self.context_networks_depth = nn.ModuleList()  # Added context networks Depth
        self.shared_decoders = nn.ModuleList()
        
        
        for i, ch in enumerate(self.lv_chs[::-1][:n_out]):
            l = self.n_pyramid - i
            ch_in = (search_range * 2 + 1) ** 2  # corr
            ch_in_depth = (search_range * 2 + 1) ** 2  # corr
            if i > 0:
                ch_in += ch + 2  # corr, x1, up_flow
                ch_in_depth += ch + 1  # corr, x1, up_depth
                
            # Add Fusion layers for bidirectional fusion.  Adjust output channels.
            in_channels_flow = (search_range * 2 + 1) ** 2 + ch + 2 # Input channels for flow fusion
            in_channels_depth = (search_range * 2 + 1) ** 2 + ch  + 1 # Input channels for depth fusion
            self.fusion_layers_flow.append(FeatureFusion(in_channels_flow, 1, ch_in)) #flow fusion
            self.fusion_layers_depth.append(FeatureFusion(in_channels_depth, 2, ch_in_depth)) #depth fusion +1 for gradient map
            if n_out - i <= self.n_context:  # Add context networks
                c_layer = ContextNetwork(32 + 2, bn=self.bn)
                d_layer = ContextNetwork(32 + 1, ch_out=1, bn=self.bn)
                self.add_module(f'ContextNetwork(Lv{len(lv_chs) - 1 - i})', c_layer)
                self.add_module(f'ContextNetworkDepth(Lv{len(lv_chs) - 1 - i})', d_layer)
                self.context_networks.append(c_layer)
                self.context_networks_depth.append(d_layer)
            else:
                self.context_networks.append(None)
                self.context_networks_depth.append(None)
                
            # Create and store Shared Decoder instance for this level
            if i==0:
                 shared_decoder = FlowEstimatorFullDeform(162, bn=self.bn)#, use_context_aware=use_context_aware
            else:
                 shared_decoder = FlowEstimatorFullDeform(ch_in + ch_in_depth , bn=self.bn)# 2 is for gradient map, use_context_aware=use_context_aware
            self.shared_decoders.append(shared_decoder)
            self.add_module(f'SharedDecoder(Lv{len(lv_chs) - 1 - i})', shared_decoder)
        
            
                
    def init_weights(self):
        for m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _forward(self, x1_pyramid, x2_pyramid, x_r_pyramid, base_image, neg=False):
        flow_predictions = []
        depth_predictions = []
        gradient_maps = []
        device = next(self.parameters()).device
        shared_corr_volume = self.shared_corr(x1_pyramid, x2_pyramid, x_r_pyramid)
        for i, ( context_flow, context_depth, shared_decoder, flow_fusion, depth_fusion) in \
        enumerate(zip(self.context_networks, self.context_networks_depth, self.shared_decoders,self.fusion_layers_flow,self.fusion_layers_depth)): 
        
            if i == 0:
                 feats, flow, depth = shared_decoder(shared_corr_volume, base_image=base_image)
                 if neg:
                    depth = -F.relu(-depth)
                 else:
                    depth = F.relu(depth)

            else:
                up_flow = F.interpolate(flow * 2, scale_factor=2, mode='bilinear', align_corners=True)
                up_depth = F.interpolate(depth * 2, scale_factor=2, mode='bilinear', align_corners=True)
                #x2_warp = flow_warp(x2_pyramid[i], up_flow)
                #for _ in range(2):  # Rokia, Iterative refinement
                x2_warp = flow_warp(x2_pyramid[i], up_flow)#, device = device)#.to(up_flow.device)#.transpose(2,3)
                corr_flow = self.shared_corr.corr(x1_pyramid[i], x2_warp)
                F.leaky_relu_(corr_flow)

                
                x2_depth_warp = flow_warp(x_r_pyramid[i], up_depth)#,device = device).to(up_depth.device)#.transpose(2, 3)
                corr_depth = self.shared_corr.corr(x1_pyramid[i], x2_depth_warp)
                F.leaky_relu_(corr_depth)
                

                flow_input = torch.cat([corr_flow, x1_pyramid[i], up_flow], dim=1) # add gradient features
                depth_input = torch.cat([corr_depth, x1_pyramid[i], up_depth], dim=1)

                # Bidirectional Fusion using Feature Fusion
                fused_flow_feat = flow_fusion(flow_input, up_depth)
                fused_depth_feat = depth_fusion(depth_input, up_flow)
                
                    # Use shared decoders
                feats, flow, depth= shared_decoder(torch.cat([fused_flow_feat, fused_depth_feat], dim=1), base_image=base_image)


                flow = flow + up_flow
                depth = depth + up_depth
                if neg:
                    depth = -F.relu(-depth)
                else:
                    depth = F.relu(depth)


                if context_flow is not None:
                    flow = flow + context_flow(torch.cat([flow, feats], dim=1))
                if context_depth is not None:
                    depth = depth + context_depth(torch.cat([depth, feats], dim=1))#depth_feats
                    if neg:
                        depth = -F.relu(-depth)
                    else:
                        depth = F.relu(depth)

            flow_predictions.append(flow)
            if neg:
                depth_predictions.append(-depth)
            else:
                depth_predictions.append(depth)
            if len(flow_predictions) == self.n_out:
                break

        flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flow_predictions]
        depths = [F.interpolate(depth * 4, scale_factor=4, mode='bilinear', align_corners=True) for depth in depth_predictions]
        
        return flows[::-1], depths[::-1]
        
        
        
    def forward(self, x, base_image, with_bk=True):
        # We extract gradient maps once at the beginning
        # We need the gradient of the first image (x1) and second image (x2 for backward pass)
        Gm1 = extract_gradient_map(x[:, :3])
        Gm2 = extract_gradient_map(x[:, 3:6])

        # Pass the corresponding gradient map to the feature extractor
        x1_pyramid = self.feature_extractor(x[:, :3], Gm1) 
        x2_pyramid = self.feature_extractor(x[:, 3:6], Gm2) 
        x_r_pyramid = self.feature_extractor(x[:, 6:9], Gm1) # Right image uses gradient from left image
        x_r2_pyramid = self.feature_extractor(x[:, 9:], Gm2) # Right image t+1 uses gradient from left image t+1
        
        # The base_image parameter is still passed to _forward as it's used by the context-aware GSPN
        flows_f, disp_lr = self._forward(x1_pyramid, x2_pyramid, x_r_pyramid, base_image=x[:, :3], neg=False)
     
        if with_bk:
            flows_b, disp_rl = self._forward(x2_pyramid, x1_pyramid, x_r2_pyramid, base_image=x[:, 3:6], neg=True)
            flows = [torch.cat((flow_f, flow_b), 1)
                     for flow_f, flow_b in zip(flows_f, flows_b)]
            disps = [torch.cat((lr + 1e-6, rl + 1e-6), 1) 
                    for lr, rl in zip(disp_lr, disp_rl)]
           
            return flows, disps
        else:
            return flows_f, disp_lr

