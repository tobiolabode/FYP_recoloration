import torch
from utils.util import *
import time

def warp_color(IA_l, IB_lab,
    cluster_value_current,cluster_value_ref,
    cluster_preds_current,cluster_preds_ref, 
    features_B, vggnet, nonlocal_net, feature_noise=0, temperature=0.01):
    print('calling wrap_color')
    IA_rgb_from_gray = gray2rgb_batch(IA_l)
    print('nonlocal_net: ', nonlocal_net)
    print('IA_rgb_from_gray: ', IA_rgb_from_gray.shape)
    print('cluster_value_current: ', cluster_value_current.shape)
    print('cluster_value_ref: ', cluster_value_ref.shape)
    print('features_B: ', cluster_value_ref.shape)
    with torch.no_grad():
        ablation_time = time.time()
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
            IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        end_time=time.time()
        _t_resnet = end_time-ablation_time
        B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

    # NOTE: output the feature before normalization
    features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

    A_relu2_1 = feature_normalize(A_relu2_1)
    A_relu3_1 = feature_normalize(A_relu3_1)
    A_relu4_1 = feature_normalize(A_relu4_1)
    A_relu5_1 = feature_normalize(A_relu5_1)
    B_relu2_1 = feature_normalize(B_relu2_1)
    B_relu3_1 = feature_normalize(B_relu3_1)
    B_relu4_1 = feature_normalize(B_relu4_1)
    B_relu5_1 = feature_normalize(B_relu5_1)

    ablation_time = time.time()
    
    # operations break around here
    # Error occurred in epoch 0, iteration 0: Given groups=1, weight of size [256, 256, 1, 1], expected input[1, 1, 4, 256] to have 256 channels, but got 1 channels instead
    # B_lab_map: [batch_size, 3, height, width]
    # cluster_value_current and cluster_value_ref: [batch_size, 16, 16]
    # cluster_preds_current and cluster_preds_ref: [batch_size, 16, 16]
    # A_relu2_1 and B_relu2_1: [batch_size, 128, height/2, width/2]
    # A_relu3_1 and B_relu3_1: [batch_size, 256, height/4, width/4]
    # A_relu4_1 and B_relu4_1: [batch_size, 512, height/8, width/8]
    # A_relu5_1 and B_relu5_1: [batch_size, 512, height/16, width/16]
    out_tensor_warp = nonlocal_net(
        IB_lab,
        cluster_value_current,
        cluster_value_ref,
        cluster_preds_current,
        cluster_preds_ref,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=temperature,
    )
    end_time=time.time()
    _t_nonlocal = end_time-ablation_time
    print('out_tensor_warp,_t_resnet,_t_nonlocal', [out_tensor_warp.shape,_t_resnet.shape,_t_nonlocal.shape])

    return out_tensor_warp,_t_resnet,_t_nonlocal


def frame_colorization(
    IA_lab,
    IB_lab,
    cluster_value_current,
    cluster_value_ref,
    cluster_preds_current,
    cluster_preds_ref,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):
    print('calling frame_colorization')
    IA_l = IA_lab[:, 0:1, :, :]
    print('IA_l', IA_l.shape)
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        out_tensor_warp,_t_resnet,_t_nonlocal = warp_color(        # bs 7 256 256
            IA_l, IB_lab,
            cluster_value_current,cluster_value_ref,
            cluster_preds_current,cluster_preds_ref,
            features_B, vggnet, nonlocal_net, feature_noise, temperature=temperature
        )
        #color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map), dim=1)     #这里是colornet的输入，
        out_tensor_warp[:,3:4,:,:] = IA_l     # ab c l
        print("out_tensor_warp[:,3:4,:,:]: ", out_tensor_warp[:,3:4,:,:].shape)
        print("out_tensor_warp", out_tensor_warp.shape)
        print("_t_nonlocal", _t_nonlocal)
        # S2 = out_tensor_warp[:,2:3,:,:].clone()
        # out_tensor_warp[:,0:1,:,:][S2<0.1]=0
        # out_tensor_warp[:,1:2,:,:][S2<0.1]=0
        # out_tensor_warp[:,2:3,:,:][S2<0.1]=0
        ablation_time=time.time()
        IA_ab_predict = colornet(out_tensor_warp)
        end_time=time.time()
        _t_colornet = end_time-ablation_time
        print(f"IA_ab_predict, out_tensor_warp,[_t_resnet,_t_nonlocal,_t_colornet]: {IA_ab_predict.shape, out_tensor_warp.shape, [_t_resnet.shape,_t_nonlocal.shape, _t_colornet.shape]}")

    return IA_ab_predict, out_tensor_warp,[_t_resnet,_t_nonlocal,_t_colornet]

