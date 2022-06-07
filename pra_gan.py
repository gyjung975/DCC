import argparse
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch

from networks import Generator
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description='DDGAN')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--lr_gen', type=float, default=0.0002)
    parser.add_argument('--lr_model', type=float, default=0.01)

    parser.add_argument('--ipc', type=int, default=1)
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=2)
    parser.add_argument('--Iteration', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--epoch_eval_train', type=int, default=300)
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    args = parser.parse_args()

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist()
    print('eval_it_pool: ', eval_it_pool)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path,
                                                                                                         args)
    args.batch_real = args.ipc

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('\nEvaluation model pool: ', model_eval_pool)

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]


        for ch in range(channel):
            print('Real images : channel %d, mean = %.4f, std = %.4f' % (ch,
                                                                         torch.mean(images_all[:, ch]),
                                                                         torch.std(images_all[:, ch])))

        print('\nInitialize latent vector from random noise')

        # latent_vector = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
        #                             dtype=torch.float, device=args.device)
        label_syn = torch.tensor(np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
                                 dtype=torch.long, device=args.device).view(-1)

        generator = Generator(args).to(args.device)
        optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_gen, weight_decay=0.2)
        optimizer_gen.zero_grad()

        criterion = nn.CrossEntropyLoss().to(args.device)
        criterion_mse = nn.MSELoss().to(args.device)
        print('\n%s TRAINING BEGINS' % get_time())

        for it in range(args.Iteration + 1):
            ''' Evaluation '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('------------------------------------------------')
                    print('Evaluation')
                    print('Train_model = %s, Eval_model = %s, iteration = %d' % (args.model,
                                                                                 model_eval,
                                                                                 it))
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
                    # print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)

                        generator.eval()

                        latent_vector = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
                                                    dtype=torch.float, device=args.device)
                        image_syn_eval = generator(latent_vector).detach()
                        label_syn_eval = copy.deepcopy(label_syn.detach())

                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                                 testloader, args)

                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs),
                                                                             model_eval,
                                                                             np.mean(accs),
                                                                             np.std(accs)))
                    print('------------------------------------------------')

                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                args.model_path = os.path.join(args.save_path, 'GAN_%s_%s_%dipc_%dexp_%diter' % (args.dataset,
                                                                                                 args.model,
                                                                                                 args.ipc,
                                                                                                 args.num_exp,
                                                                                                 args.Iteration))
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)

                save_name = os.path.join(args.model_path, 'exp%d_iter%d.png' % (0, it))

                image_syn_vis = generator(torch.randn(size=(num_classes * args.ipc, args.latent_dim),
                                                      dtype=torch.float, device=args.device).detach()).cpu()

                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]

                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc)

            ''' Train '''
            # model = models.resnet34(pretrained=False)
            # num_features = model.fc.in_features
            # model.fc = nn.Linear(num_features, num_classes)
            # model = model.to(args.device)
            #
            # model.train()
            # optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=0.2)

            generator.train()

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()

            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()

            loss_avg = 0
            args.dc_aug_param = None

            for ol in range(args.outer_loop):
                BN_flag = False
                BNSizePC = 16  # for batch normalization

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

                ''' Update Generator '''
                ### outer loop 밖?? 안??? ###
                latent_vector = torch.randn(size=(num_classes * args.ipc, args.latent_dim),
                                            dtype=torch.float, device=args.device)
                gen_img_syn = generator(latent_vector)

                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c

                    img_syn = gen_img_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    output_real = net(img_real)
                    # output_real = model(img_real)
                    output_syn = net(img_syn)
                    # output_syn = model(img_syn)

                    loss_real = criterion(output_real, lab_real)
                    loss_syn = criterion(output_syn, lab_syn)
                    loss_mse = criterion_mse(output_syn, output_real)
                    loss += loss_mse + loss_syn

                optimizer_gen.zero_grad()
                loss.backward()
                optimizer_gen.step()

                loss_avg += loss.item()

                ''' Update Net '''
                # image_syn_train = generator(latent_vector).detach()
                image_syn_train = copy.deepcopy(gen_img_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())

                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, batch_size=args.batch_train,
                                         shuffle=True, num_workers=args.num_worker)

                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=False)
                    # epoch('train', trainloader, model, optimizer_net, criterion, args, aug=False)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration:
                torch.save(generator.state_dict(), os.path.join(args.model_path, 'generator.t7'))

        print('\n==================== Final Results ====================\n')

        for key in model_eval_pool:
            accs = accs_all_exps[key]
            print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (args.num_exp,
                                                                                                            args.model,
                                                                                                            len(accs),
                                                                                                            key,
                                                                                                            np.mean(accs) * 100,
                                                                                                            np.std(accs) * 100))


if __name__ == "__main__":
    main()