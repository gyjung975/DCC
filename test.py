import argparse
import os
import time
import copy
import numpy as np
from scipy.ndimage.interpolation import rotate as scipyrotate
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torchvision.utils import save_image
from putils import get_loops, get_dataset, get_network, get_eval_pool, get_daparam, \
    match_loss, get_time, TensorDataset, evaluate_synset, epoch, evaluate

import clip
from te import text_embedding


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='FashionMNIST')

    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--test_model', type=str, default='ConvNet_eval')
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--epoch_eval_train', type=int, default=300)
    parser.add_argument('--lr_img', type=float, default=0.1)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--init', type=str, default='noise')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--dis_metric', type=str, default='ours')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(50, args.Iteration+1, 50).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]
    print('eval_it_pool: ', eval_it_pool)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path,
                                                                                                         args)
    print(class_names)
    # model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    model_eval_pool = ['ConvNet_eval']

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        # sen = ['A photo of airplane', 'A photo of automobile', 'A photo of bird', 'A photo of catt', 'A photo of deer',
        #        'A photo of dog', 'A photo of frog', 'A photo of horse', 'A photo of ship', 'A photo of truck']
        # sen = ['airplane', 'automobile', 'bird', 'catt', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # sen = ['A photo of zero', 'A photo of one', 'A photo of two', 'A photo of three', 'A photo of four',
        #        'A photo of five', 'A photo of six', 'A photo of seven', 'A photo of eight', 'A photo of nine']
        # sen = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        sen = ['A photo of T-shirt/top', 'A photo of Trouser', 'A photo of Pullover', 'A photo of Dress',
               'A photo of Coat', 'A photo of Sandal', 'A photo of Shirt', 'A photo of Sneaker', 'A photo of Bag', 'A photo of Ankle boot']
        text_feature = text_embedding(sen).detach()
        # text_feature = text_embedding(sen)
        # [10, 512] / False

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        images_all = torch.cat(images_all, dim=0).to(args.device)
        # [50000, 3, 32, 32] / False
        labels_all = torch.tensor(labels_all, dtype=torch.float, device=args.device)
        # [50000] / False

        labels_int = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.int, device=args.device)
        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_int):
            indices_class[lab].append(i)

        labels_emb = copy.deepcopy(labels_all.detach())
        labels_emb = labels_emb.unsqueeze(dim=1).repeat(1, 512)
        for c in range(num_classes):
            for i in indices_class[c]:
                labels_emb[i] = text_feature[c]
        # [50000, 512] / False

        # for c in range(num_classes):
        #     print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f' % (ch,
        #                                                                torch.mean(images_all[:, ch]),
        #                                                                torch.std(images_all[:, ch])))

        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float32, requires_grad=True, device=args.device)
        # [10, 3, 32, 32] / True
        label_syn = torch.tensor(np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
                                 dtype=torch.float32, requires_grad=False, device=args.device).view(-1)
        # [10] / False

        #####################################################################
        label_syn_emb = copy.deepcopy(label_syn.detach())
        label_syn_emb = label_syn_emb.unsqueeze(dim=1).repeat(1, 512)
        for c in range(num_classes):
            label_syn_emb[c * args.ipc:(c + 1) * args.ipc] = text_feature[c]
        # [10, 512]
        ######################################################################

        print('initialize synthetic data from random noise')

        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s TRAINING BEGINS' % get_time())

        for it in range(args.Iteration + 1):
            if it in eval_it_pool:
                # model_pool = ['ConvNet_eval']
                for model_eval in model_eval_pool:
                    print('------------------------------------------------')
                    print('Evaluation')
                    print('model_train = %s, model_eval = %s, iteration = %d' % (args.model,
                                                                                 model_eval,
                                                                                 it))
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)

                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        # [10 * ipc, 3, 32, 32]
                        label_syn_eval = copy.deepcopy(label_syn.detach())
                        # [10 * ipc]
                        # label_syn_eval = copy.deepcopy(label_syn_emb.detach())
                        # [10, 512]

                        _, acc_train, acc_test = evaluate(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                          testloader, args, text_feature=text_feature)

                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs),
                                                                             model_eval,
                                                                             np.mean(accs),
                                                                             np.std(accs)))
                    print('------------------------------------------------')

                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                # args.model_path = os.path.join(args.save_path, 'pra_%s_%s_%dipc_%dexp_%diter' % (args.dataset,
                #                                                                                  args.model,
                #                                                                                  args.ipc,
                #                                                                                  args.num_exp,
                #                                                                                  args.Iteration))
                # if not os.path.exists(args.model_path):
                #     os.mkdir(args.model_path)
                #
                # save_name = os.path.join(args.model_path, 'exp%d_iter%d.png' % (exp, it))
                #
                # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                #
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                #
                # image_syn_vis[image_syn_vis < 0] = 0.0
                # image_syn_vis[image_syn_vis > 1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc)



            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()

            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()

            loss_avg = 0
            args.dc_aug_param = None

            for ol in range(args.outer_loop):
                BN_flag = False
                BNSizePC = 16

                for module in net.modules():
                    if 'BatchNorm' in module._get_name():
                        BN_flag = True

                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()                     # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)     # get running mu, sigma

                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()           # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(args.device)

                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    # [batch, 3, 32, 32]
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    # [batch]

                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    # [1, 3, 32, 32]
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    # [1]
                    lab_syn_emb = copy.deepcopy(label_syn_emb.detach())
                    # lab_syn_emb = copy.deepcopy(text_feature.detach())
                    # [10, 512]

                    output_real = net(img_real)
                    # [batch, 512]
                    similarity_real = output_real @ lab_syn_emb.T
                    # [batch, 10]
                    loss_real = criterion(similarity_real, lab_real)

                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    # [1, 512]
                    similarity_syn = output_syn @ lab_syn_emb.T
                    # [1, 10]
                    loss_syn = criterion(similarity_syn, lab_syn)

                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

                loss_avg += loss.item()

                image_syn_train = copy.deepcopy(image_syn.detach())     # [10, 3, 32, 32]
                label_syn_train = copy.deepcopy(label_syn.detach())     # [10]
                label_syn_eval_train = copy.deepcopy(label_syn_emb.detach())     # [10, 512]

                dst_syn_train = TensorDataset(image_syn_train, label_syn_eval_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train,
                                                          shuffle=True, num_workers=args.num_worker)

                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=False, text_feature=text_feature)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            # if it == args.Iteration:
            #     data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, },
            #                os.path.join(args.model_path, 'model.pt'))

            args.model_path = os.path.join(args.save_path, 'pra_%s_%s_%dipc_%dexp_%diter' % (args.dataset,
                                                                                             args.model,
                                                                                             args.ipc,
                                                                                             args.num_exp,
                                                                                             args.Iteration))
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)

            save_name = os.path.join(args.model_path, 'exp%d_iter%d.png' % (exp, it))

            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())

            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]

            image_syn_vis[image_syn_vis < 0] = 0.0
            image_syn_vis[image_syn_vis > 1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc)

    print('\n==================== Final Results ====================\n')

    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp,
                                                                                                        args.model,
                                                                                                        len(accs),
                                                                                                        key,
                                                                                                        np.mean(accs) * 100,
                                                                                                        np.std(accs) * 100))


if __name__ == '__main__':
    main()
