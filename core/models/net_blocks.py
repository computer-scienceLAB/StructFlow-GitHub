import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform import DeformableConv2d


def Dconv(inC, outC, kernel_size =3 , stride=1,padding=1, dilation=1, isReLU=True):
      if isReLU:
        return nn.Sequential(DeformableConv2d(inC, outC, kernel_size=kernel_size, stride=stride,
                 padding=padding, dilation=dilation),
                 nn.LeakyReLU(0.1, inplace=True))
      else:
        return nn.Sequential(DeformableConv2d(inC, outC, kernel_size=kernel_size, stride=stride,
             padding=padding, dilation=dilation))


class CustomConv2D(nn.Conv2d):
    # add `pad_same` option to native conv2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, pad_same=False):
        self.pad_same = pad_same
        super(CustomConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def _conv2d_same_padding(self, input, weight, bias, stride, dilation, groups):

        def _compute_padding(i, k, s, d):
            len_o = (i + s - 1) // s
            return max(0, (len_o - 1) * s + (k - 1) * d + 1 - i)

        pad_row = _compute_padding(input.size(2), weight.size(2), stride[0], dilation[0])
        pad_col = _compute_padding(input.size(3), weight.size(3), stride[1], dilation[1])

        if pad_row % 2:
            input = F.pad(input, (0, 0, 0, 1))  # padding the last row
        if pad_col % 2:
            input = F.pad(input, (0, 1, 0, 0))  # padding the last column

        return F.conv2d(input, weight, bias, stride,
                        padding=(pad_row // 2, pad_col // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input):
        if self.pad_same:
            return self._conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                             self.dilation, self.groups)
        else:
            return super(CustomConv2D, self).forward(input)


def conv(batch_norm, in_planes, out_planes,
         kernel_size=3, stride=1, dilation=1, pad_same=True):
    if batch_norm:
        return nn.Sequential(
            CustomConv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=((kernel_size - 1) * dilation) // 2,
                         dilation=dilation, bias=True, pad_same=pad_same),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            CustomConv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=((kernel_size - 1) * dilation) // 2,
                         dilation=dilation, bias=True, pad_same=pad_same),
            nn.LeakyReLU(0.1, inplace=True)
        )


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class StructureAwareSpatialAttention(nn.Module):
    """ The new, novel Spatial Attention that IS structure-aware. """
    def __init__(self, kernel_size=7):
        super(StructureAwareSpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        # Input channels are now 3 (avg, max, gradient)
        self.conv = nn.Conv2d(3, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gm_downsampled):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Concatenate the gradient map as a third cue
        y = torch.cat([max_pool, avg_pool, gm_downsampled], dim=1)
        y = self.conv(y)
        attention_map = self.sigmoid(y)
        return x * attention_map


class SharedFeatureTransform(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
       super(SharedFeatureTransform, self).__init__()
       self.transform = nn.Sequential(
           conv(bn, in_channels, out_channels),
           conv(bn, out_channels, out_channels)
        )
    def forward(self, x):
       return self.transform(x)


class FeaturePyramidExtractor(nn.Module):
    # --- Added use_structure_aware flag ---
    def __init__(self, lv_chs, bn=False):#, use_structure_aware=False
        super(FeaturePyramidExtractor, self).__init__()
        #self.use_structure_aware = use_structure_aware

        self.convs = []
        self.se_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        self.feature_transform_blocks = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(lv_chs[:-1], lv_chs[1:])):
            layer = nn.Sequential(
                conv(bn, ch_in, ch_out, stride=2),
                conv(bn, ch_out, ch_out)
            )
            self.add_module('Feature(Lv{})'.format(l + 1), layer)
            self.convs.append(layer)
            self.se_blocks.append(SEBlock(ch_out))

            # Use if/else to select the correct attention module
            #if self.use_structure_aware:
            self.spatial_blocks.append(StructureAwareSpatialAttention())
            #else:
            #    self.spatial_blocks.append(OriginalSpatialAttention())
            
            self.feature_transform_blocks.append(SharedFeatureTransform(ch_out, ch_out))


    def forward(self, x, Gm=None):
        feature_pyramid = [x]
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            x = self.se_blocks[i](x)

            # se if/else to call the correct attention forward pass ---
            #if self.use_structure_aware:
            if Gm is None:
                raise ValueError("Structure-aware attention requires a gradient map 'Gm'.")
            gm_downsampled = F.interpolate(Gm, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = self.spatial_blocks[i](x, gm_downsampled)
            #else:
            #    x = self.spatial_blocks[i](x)

            x = self.feature_transform_blocks[i](x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class ContextNetwork(nn.Module):

    def __init__(self, ch_in, ch_out=2, bn=False):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(
            conv(bn, ch_in, 128, 3, 1, 1),
            conv(bn, 128, 128, 3, 1, 2),
            conv(bn, 128, 128, 3, 1, 4),
            conv(bn, 128, 96, 3, 1, 8),
            conv(bn, 96, 64, 3, 1, 16),
            conv(bn, 64, 32, 3, 1, 1),
        )
        self.predict_flow = CustomConv2D(32, ch_out, kernel_size=3, stride=1,
                                         padding=1)

    def forward(self, x):
        x = self.convs(x)
        return self.predict_flow(x)


class FeatureFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d( (out_channels // 2) + (out_channels // 2), out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.relu(self.conv1(x1))
        #print('x1',x1.shape)
        x2 = self.relu(self.conv2(x2))
        #print('x2',x2.shape)
        fused = torch.cat([x1, x2], dim=1)
        #print('fused',fused.shape)
        return self.relu(self.conv3(fused))


def extract_gradient_map(image):
    """
    Extracts the gradient magnitude map from an image.
    Args:
        image (torch.Tensor): Input image (B x C x H x W).
    Returns:
        torch.Tensor: Gradient magnitude map (B x 1 x H x W).
    """
    # Convert to grayscale 
    if image.shape[1] == 3:
        image = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image = image.unsqueeze(1)

    # Calculate gradients
    gx = image[:, :, :, :-1] - image[:, :, :, 1:]
    gy = image[:, :, :-1, :] - image[:, :, 1:, :]

    # Pad gradients to match original image size
    gx = F.pad(gx, (0, 1, 0, 0), mode='replicate') # Pad the last column
    gy = F.pad(gy, (0, 0, 0, 1), mode='replicate') # Pad the last row

    # Calculate gradient magnitude
    gradient_magnitude = torch.sqrt(gx**2 + gy**2)

    return gradient_magnitude


class FlowEstimatorFullDeform(nn.Module):
    # Added use_context_aware flag ---
    def __init__(self, ch_in, ch_out=2, bn=False):#, use_context_aware=False
        super(FlowEstimatorFullDeform, self).__init__()
        #self.use_context_aware = use_context_aware
        
        # --- All conv and dcn layers remain the same ---
        self.conv1 = conv(bn, ch_in, 128)
        self.conv2 = conv(bn, 128 + 128, 128)
        self.conv3 = conv(bn, 128 + 128 + 128, 96)
        self.conv4 = conv(bn, 128 + 96 + 96, 64)
        self.conv5 = conv(bn, 96 + 64 + 64, 32)

        self.dcn1 = Dconv(128, 128)
        self.dcn2 = Dconv(128, 128)
        self.dcn3 = Dconv(96, 96)
        self.dcn4 = Dconv(64, 64)
        self.dcn5 = Dconv(32, 32)
        
        # Use if/else to initialize the correct guidance network ---
        #if self.use_context_aware:
            # New, novel Context-Aware GSPNs
        self.gspn_x4 = GSPN(feature_channels=64)
        self.gspn_x5d5 = GSPN(feature_channels=32)
        #else:
            # Original, baseline "blind" network
        self.gradient_scale_network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

        self.predict_flow = CustomConv2D(64 + 32 + 32, ch_out, kernel_size=3, stride=1, padding=1)
        self.predict_depth = CustomConv2D(64 + 32 + 32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, base_image=None):
        # --- Feature extraction part is the same ---
        x1 = self.conv1(x)
        d1 = self.dcn1(x1)
        x2 = self.conv2(torch.cat([x1, d1], dim=1))
        d2 = self.dcn2(x2)
        x3 = self.conv3(torch.cat([x1, x2, d2], dim=1))
        d3 = self.dcn3(x3)
        x4 = self.conv4(torch.cat([x2, x3, d3], dim=1))
        d4 = self.dcn4(x4)
        x5 = self.conv5(torch.cat([x3, x4, d4], dim=1))
        d5 = self.dcn5(x5)
        
        if base_image is not None:
            gradient_map = extract_gradient_map(base_image)
            gm_interpolated = F.interpolate(gradient_map, size=x4.shape[2:], mode='bilinear', align_corners=False)

            # Use if/else to apply the correct guidance logic ---
            #if self.use_context_aware:
                # New, novel context-aware logic
            scale_map_x4 = self.gspn_x4(x4, gm_interpolated)
            scale_map_x5d5 = self.gspn_x5d5(x5, gm_interpolated)
                
            x4 = x4 * (1 + 0.1 * scale_map_x4 * gm_interpolated)
            x5 = x5 * (1 + 0.1 * scale_map_x5d5 * gm_interpolated)
            d5 = d5 * (1 + 0.1 * scale_map_x5d5 * gm_interpolated)
            #else:
                # Original, baseline "blind" logic
                #scale_map = self.gradient_scale_network(gm_interpolated)
                #x4 = x4 * (1 + 0.1 * scale_map * gm_interpolated)
                #x5 = x5 * (1 + 0.1 * scale_map * gm_interpolated)
                #d5 = d5 * (1 + 0.1 * scale_map * gm_interpolated)

        flow = self.predict_flow(torch.cat([x4, x5, d5], dim=1))
        depth = self.predict_depth(torch.cat([x4, x5, d5], dim=1))

        return x5, flow, depth


class GSPN(nn.Module):
    """ Context-Aware Gradient Scale Prediction Network """
    def __init__(self, feature_channels):
        super(GSPN, self).__init__()
        # Input channels are the features + 1 (for the gradient map)
        self.conv1 = nn.Conv2d(feature_channels + 1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, gm_interpolated):
        # Concatenate features and gradients to provide context
        combined_input = torch.cat([features, gm_interpolated], dim=1)
        x = self.relu(self.conv1(combined_input))
        scale_map = self.sigmoid(self.conv2(x))
        return scale_map
