import argparse
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch

from networks import Generator
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--lr_gen', type=float, default=0.0002)

    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--method', type=str, default='DC')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    # S: the same to training model, M: multi architectures,  W: net width, D: net depth,
    # A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20,
                        help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300,
                        help='num_eval 마다의 train epoch')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations; 하나의 synthetic data *번 update')
    parser.add_argument('--lr_img', type=float, default=0.1,
                        help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01,
                        help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    args = parser.parse_args()

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' else [args.Iteration]
    # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path,
                                                                                                         args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    # synthetic data로 train하고 eval할 model

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        # images_all = []
        # labels_all = []
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        # len(images_all) = 60,000 / [[1, 1, 28, 28], [1, 1, 28, 28], ..., [1, 1, 28, 28]]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        # len(labels_all) = 60,000 / ['class', 'class', ..., 'class']

        images_all = torch.cat(images_all, dim=0).to(args.device)                       # [60000, 1, 28, 28]
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)     # [60000]

        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):       # class c images 중 n개 random get
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (ch,
                                                                       torch.mean(images_all[:, ch]),
                                                                       torch.std(images_all[:, ch])))

        ''' initialize the synthetic data  -->  latent vector '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float, requires_grad=True, device=args.device)
        # [num_class * ipc, channel, H, W]
        # label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)],
        #                          dtype=torch.long, requires_grad=False, device=args.device).view(-1)
        label_syn = torch.tensor(np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
                                 dtype=torch.long, requires_grad=False, device=args.device).view(-1)
        # [num_class * ipc] : [0,0, ..., 1,1, ..., 9,9]

        ######################################################################################
        # image_syn = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
        #                         dtype=torch.float, requires_grad=False, device=args.device)
        # label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)],
        #                          dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        # generator = Generator(args).to(args.device)
        # optimizer_gen = torch.optim.SGD(generator.parameters(), lr=args.lr_gen, momentum=0.5)
        ######################################################################################

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c+1) * args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins' % get_time())

        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data  -->  Generator '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('------------------------------------------------')
                    print('Evaluation')
                    print('model_train = %s, model_eval = %s, iteration = %d' % (args.model,
                                                                                 model_eval,
                                                                                 it))
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)

                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        label_syn_eval = copy.deepcopy(label_syn.detach())

                        ###############################################
                        # generator.eval()
                        # with torch.no_grad():
                        #     image_syn_eval = generator(image_syn)
                        #     label_syn_eval = copy.deepcopy(label_syn)
                        ###############################################

                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                                 testloader, args)

                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs),
                                                                             model_eval,
                                                                             np.mean(accs),
                                                                             np.std(accs)))
                    print('------------------------------------------------')

                    if it == args.Iteration:        # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                args.model_path = os.path.join(args.save_path, '%s_%s_%dipc_%dexp_%diter' % (args.dataset,
                                                                                             args.model,
                                                                                             args.ipc,
                                                                                             args.num_exp,
                                                                                             args.Iteration))
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)

                save_name = os.path.join(args.model_path, 'exp%d_iter%d.png' % (exp, it))

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())     # [num_class * ipc, channel, H, W]

                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]

                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc)
                # [num_class * ipc, channel, H, W]
                # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data  -->  Generator'''
            ######################################################################
            # model = models.resnet34(pretrained=False)
            # num_features = model.fc.in_features
            # model.fc = nn.Linear(num_features, num_classes)
            # model = model.to(args.device)

            # model.train()
            # optimizer_model = torch.optim.SGD(model.parameter(), lr=args.lr_model)

            # generator.train()
            #######################################################################

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()

            net_parameters = list(net.parameters())     # net.parameters() : generator
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()

            loss_avg = 0
            args.dc_aug_param = None
            # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function)

            for ol in range(args.outer_loop):
                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16       # for batch normalization

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

                ''' update synthetic data  -->  Generator '''
                loss = torch.tensor(0.0).to(args.device)

                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c

                    ##################################################################################################
                    # gen_img_syn = generator(image_syn)
                    # img_syn = gen_image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    # lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    ##################################################################################################

                    img_syn = image_syn[c * args.ipc:(c+1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    output_real = net(img_real)

                    ##############################
                    # output_real = model(img_real)
                    ##############################

                    loss_real = criterion(output_real, lab_real)

                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)

                    ############################
                    # output_syn = model(img_syn)
                    ############################

                    loss_syn = criterion(output_syn, lab_syn)

                    ###################################################
                    # loss_sim = torch.multiply(output_syn, output_real)
                    # loss_syn = criterion(output_syn, lab_syn)
                    ###################################################

                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                    #############################
                    # loss += -loss_sim + loss_syn
                    #############################

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

                ##########################
                # optimizer_gen.zero_grad()
                # loss.backward()
                # optimizer_gen.step()
                ###########################

                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train = copy.deepcopy(image_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())

                print(image_syn_train.shape)

                #############################################
                # image_syn_train = generator(image_syn)
                # label_syn_train = copy.deepcopy(label_syn)
                #############################################

                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train,
                                                          shuffle=True, num_workers=args.num_worker)

                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=False)

                    ################################################################################
                    # epoch('train', trainloader, model, optimizer_model, criterion, args, aug=False)
                    #################################################################################

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration:        # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, },
                           os.path.join(args.model_path, 'model.pt'))

                #####################################################################################
                # torch.save(generator.state_dict(),
                #            os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.t7' % (args.method,
                #                                                                    args.dataset,
                #                                                                    args.model,
                #                                                                    args.ipc)))
                ######################################################################################

    print('\n==================== Final Results ====================\n')

    for key in model_eval_pool:
        accs = accs_all_exps[key]
        # print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()
