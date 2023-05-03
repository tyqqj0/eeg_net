import loss_reweighting as loss_expect
import torch
import torch.nn as nn
from torch.autograd import Variable

from schedule import lr_setter


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())  # TODO:这里的weight是什么 取0维应该是样本数
    # print('weight size', weight.size())
    # print('cfeatures size', cfeatures.size())
    # print('pre_features size', pre_features.size())
    # print('pre_weight1 size', pre_weight1.size())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)  # 样本数一样，所以dim=0
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)  # 这个是SGD优化器，用来优化weight

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum)  # TODO:搞懂这块
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        # print(".......................................")
        # print("lossb", lossb)
        # print("lossp", lossp)
        # print("lambdap", lambdap)
        # print("lossg", lossg)
        # print(".......................................")
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)
    # print('pre_features size, pre_weight1 size\n\n----', pre_features.size(), pre_weight1.size())
    softmax_weight = softmax(weight)

    return softmax_weight, pre_features, pre_weight1
