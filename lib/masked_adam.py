import os
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
# from jt.utils.cpp_extension import load
#TODO:
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# sources=['cuda/adam_upd.cpp', 'cuda/adam_upd_kernel.cu']
# adam_upd_cuda = load(
#         name='adam_upd_cuda',
#         sources=[os.path.join(parent_dir, path) for path in sources],
#         verbose=True)

from .jit_cuda import adam_upd
''' Extend Adam optimizer
1. support per-voxel learning rate
2. masked update (ignore zero grad) which speeduping training
'''
class MaskedAdam(jt.optim.Adam):
# class MaskedAdam(jt.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.per_lr = None
        super(MaskedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()
    # TODO: state_dict not same with pytorch
    # def state_dict(self):

    #     state = {"defaults": self.defaults,self.param_groups}
    #     return state

    # def load_state_dict(self, state):

    #     for k,v in state["defaults"].items():
    #         setattr(self, k, v)

    
    def step(self):
        n = self.n_step
        jt.flags.node_order = 1
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            eps = group.get("eps", self.eps)
            skip_zero_grad=group["skip_zero_grad"]
            # weight_decay = group.get("weight_decay", self.weight_decay)
            beta1, beta2 = group.get("betas", self.betas)

            for param,grad, v, m in zip(group["params"], group["grads"], group["values"], group["m"]):
                if param.is_stop_grad(): 
                    print("It has grad but  is set stop grad")
                    continue
                    # Lazy state initialization
                    # if len(state) == 0:
                    #     state['step'] = 0
                    #     # # Exponential moving average of gradient values
                    #     # state['exp_avg'] = jt.zeros_like(param, memory_format=jt.preserve_format)
                    #     # # Exponential moving average of squared gradient values
                    #     # state['exp_avg_sq'] = jt.zeros_like(param, memory_format=jt.preserve_format)
              

                    # state['step'] += 1
                    #TODO: grad
                if self.per_lr is not None and param.shape == self.per_lr.shape:
                    adam_upd.adam_upd_with_perlr(
                            param, grad, m, v, self.per_lr,
                            n, beta1, beta2, lr, eps)
                elif skip_zero_grad:
                    #TODO:
                    adam_upd.masked_adam_upd(
                            param, grad, m, v,
                            n, beta1, beta2, lr, eps)
                else:
                    adam_upd.adam_upd(
                            param, grad, m, v,
                            n, beta1, beta2, lr, eps)
                param.requires_grad=True
        self.post_step()
