# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root for full license information.

if __name__ == '__main__':
    import os
    import argparse
    from PIL import Image
    import numpy as np
    import cv2

    import torch
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    from lib.models.ExampleColorNet import ExampleColorNet

    from utils.util import mkdir_if_not, lab2rgb_transpose_mc

    from lib.TestDataset import TestDataset
    import lib.TestTransform as transforms

    # keep folder combo path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='Custom_Models\Deep-Exemplar-based-Colorization\colorization_subnet\lib\examplar-based\\', type=str)
    parser.add_argument('--test_model', type=str, default='Custom_Models\Deep-Exemplar-based-Colorization\colorization_subnet\lib\\models\\example_net.pth')
    parser.add_argument('--gpu_id',type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='Custom_Models\Deep-Exemplar-based-Colorization\colorization_subnet\lib\examplar-based\\output\\')
    parser.add_argument('--short_size', type=int, default=256)
    # Custom_Models\Deep-Exemplar-based-Colorization\colorization_subnet\lib\pairs.txt    
    opt = parser.parse_args()
    mkdir_if_not(opt.out_dir)
    print(opt.data_root)
    print(opt.out_dir)
    if opt.gpu_id >= 0:
        torch.cuda.set_device(opt.gpu_id)
    if opt.short_size > 0:
        test_dataset = TestDataset(opt.data_root,
                                   transform=transforms.Compose([transforms.Resize(opt.short_size),
                                                                 transforms.RGB2Lab(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(),
                                                                 transforms.Concatenate()]))
    else:
        test_dataset = TestDataset(opt.data_root,
                                   transform=transforms.Compose([transforms.RGB2Lab(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(),
                                                                 transforms.Concatenate()]))

    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True)

    color_net = ExampleColorNet(ic=13)
    assert os.path.exists(opt.test_model), 'cannot find the test model: %s ' % opt.test_model
    color_net.load_state_dict(torch.load(opt.test_model, map_location=lambda storage, loc: storage))
    color_net.eval()

    if opt.gpu_id >= 0:
        color_net.cuda()

    size_unit = 8
    for iter, data in enumerate(data_loader):
        orig_im_l, orig_err_ba, orig_warped_ba, orig_warped_aba, orig_im_ab, orig_err_ab = data
        print('-'*10)
        print('PRE OPS: ')
        print(f"orig_im_l shape: {orig_im_l.shape}")
        print(f"orig_warped_ba[:, 1:, ...] shape: {orig_warped_ba[:, 1:, ...].shape}")
        out_name = test_dataset.get_out_name(iter)
        print(f'out_name: {out_name}')
        basename, ext = os.path.splitext(os.path.basename(out_name))
        print('testing for [%d/%d] %s ' % (iter, len(test_dataset), out_name))

        orig_h, orig_w = orig_im_l.size(2), orig_im_l.size(3)
        unit_h = int(orig_h/size_unit) * size_unit
        unit_w = int(orig_w/size_unit) * size_unit
        print(f"unit_h shape: {unit_h}")
        print(f"unit_w shape: {unit_w}")
        print(f"orig_h shape: {orig_h}")
        print(f"orig_w shape: {orig_w}")

        print(f'orig_warped_ba.size(2) :  {orig_warped_ba.size(2)}')
        print(f" VS unit_h shape: {unit_h}")

        if (unit_h != orig_h) or (unit_w != orig_w) or (orig_warped_ba.size(2) != unit_h):
            print('unit_h != orig_h or unit_w != orig_w')
            im_l = torch.from_numpy(cv2.resize(orig_im_l[0].numpy().transpose((1,2,0)), (unit_w, unit_h))[np.newaxis,np.newaxis, ...])
            err_ba = torch.from_numpy(cv2.resize(orig_err_ba[0].numpy().transpose((1, 2, 0)), (unit_w, unit_h)).transpose((2, 0, 1))[np.newaxis, ...])
            warped_ba = torch.from_numpy(cv2.resize(orig_warped_ba[0].numpy().transpose((1, 2, 0)), (unit_w, unit_h)).transpose((2, 0, 1))[np.newaxis, ...])
            warped_aba = torch.from_numpy(cv2.resize(orig_warped_aba[0].numpy().transpose((1, 2, 0)), (unit_w, unit_h)).transpose((2, 0, 1))[np.newaxis, ...])
            im_ab = torch.from_numpy(cv2.resize(orig_im_ab[0].numpy().transpose((1, 2, 0)), (unit_w, unit_h)).transpose((2, 0, 1))[np.newaxis, ...])
            err_ab = torch.from_numpy(cv2.resize(orig_err_ab[0].numpy().transpose((1, 2, 0)), (unit_w, unit_h)).transpose((2, 0, 1))[np.newaxis, ...])
        else:
            print('No Trigger')
            im_l, err_ba, warped_ba, warped_aba, im_ab, err_ab = orig_im_l, orig_err_ba, orig_warped_ba, orig_warped_aba, orig_im_ab, orig_err_ab

        # if warped_ba.size(2) != unit_h:

        # def resize_tensor(input_tensor, new_height, new_width):
        #     numpy_tensor = input_tensor.numpy().transpose((1, 2, 0))
        #     resized = cv2.resize(numpy_tensor, (new_width, new_height)).transpose((2, 0, 1))
        #     return torch.from_numpy(resized[np.newaxis, ...])

        # expected_h, expected_w = 256, 320 # Assuming these are your model's expected dimensions
        # if im_l.size(2) != expected_h or im_l.size(3) != expected_w:
        #     im_l = resize_tensor(im_l[0], expected_h, expected_w)
        # if warped_ba.size(2) != expected_h or warped_ba.size(3) != expected_w:
        #     warped_ba = resize_tensor(warped_ba[0], expected_h, expected_w)

        print('POST OPs :')
        print(f"im_l shape:  {im_l.shape}")
        print(f"warped_ba shape:  {warped_ba.shape}")
        # add a assert or expection here if dims don't match
        colornet_input = torch.cat((im_l, warped_ba[:, 1:, ...], err_ba, err_ab), dim=1) # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 256 but got size 285 for tensor number 1 in the list.
        # channels 3-4 seem swapped:    im_l.shape
                                        # torch.Size([1, 1, 256, 320])
                                        # warped_ba[:, 1:, ...].shape
                                        # torch.Size([1, 2, 285, 256])

        if opt.gpu_id >= 0:
            colornet_input_v = Variable(colornet_input.cuda())
        else:
            colornet_input_v = Variable(colornet_input)

        pred_ab_v = color_net(colornet_input_v)
        if unit_h != orig_h or unit_w != orig_w:
            pred_orig_ab = torch.from_numpy(cv2.resize(pred_ab_v.data[0].cpu().numpy().transpose((1, 2, 0)), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))[np.newaxis, ...])
        else:
            pred_orig_ab = pred_ab_v

        warpba_color_im = lab2rgb_transpose_mc(orig_im_l[0], pred_orig_ab[0])
        Image.fromarray(warpba_color_im).save(os.path.join(opt.out_dir, out_name))