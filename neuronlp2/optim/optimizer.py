import torch

class AdamOptimizer:
    def __init__(self, parameter, lr=.0012,betas=(.9,.98),eps=1e-12,
                    weight_decay=1e-5 ,decay=.75, decay_step=5000):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=betas,
                                      eps=eps,weight_decay=weight_decay)
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()