from torch import optim
from modules import deeplabv3_mobilenetv2


def get_scheduler(opts, optimizer, max_iter=None):
    if opts.lr_policy == 'poly':
        assert max_iter is not None, "max_iter necessary for poly LR scheduler"
        return optim.lr_scheduler.LambdaLR(optimizer,
                                           lr_lambda=lambda cur_iter: (1 - cur_iter / max_iter) ** opts.lr_power)
    if opts.lr_policy == 'warmuppoly':
        assert max_iter is not None, "max_iter necessary for poly LR scheduler"
        return WarmupPolyLrScheduler(optimizer, power=0.9, max_iter=max_iter + opts.warmup_iters,
                                     warmup_iter=opts.warmup_iters, warmup_ratio=0.1, warmup='exp', last_epoch=-1)

    if opts.lr_policy == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    return None


class WarmupLrScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        ratio = self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        else:
            raise NotImplementedError
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(self, optimizer, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1):
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio


def get_optimizer(opt_name, params, lr, weight_decay, momentum, nesterov):
    if opt_name == 'SGD':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    if opt_name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise NotImplementedError


def get_optimizer_and_scheduler(args, model_parameters, max_iter, lr=None):
    params = [{"params": filter(lambda p: p.requires_grad, model_parameters),
               'weight_decay': args.weight_decay}]
    optimizer = get_optimizer(args.optimizer, params, args.lr if lr is None else lr, args.weight_decay, args.momentum,
                              args.nesterov)
    scheduler = get_scheduler(args, optimizer, max_iter=max_iter)
    return optimizer, scheduler


def make_model(args, augm_model=False):
    dict_model = {
        'deeplabv3': {'model': deeplabv3_mobilenetv2, 'kwargs': {}},
    }

    if args.hp_filtered and augm_model:
        dict_model[args.model]['kwargs']['in_channels'] = 4

    return dict_model[args.model]['model'](args.num_classes, **dict_model[args.model]['kwargs'])


def schedule_cycling_lr(r, c, lr1, lr2):
    t = 1 / c * (r % c + 1)
    lr = (1 - t) * lr1 + t * lr2
    return lr
