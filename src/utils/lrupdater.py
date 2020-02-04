class LrUpdater():
    def __init__(self, lr, optimizer, epochs_text, gammas_text):
        self.optimizer = optimizer
        self.current_lr = lr
        self.epochs = [int(i) for i in epochs_text.split(',')]
        self.gammas = [float(i) for i in gammas_text.split(',')]

    def set_lr(self, lr):
        print(f"Changing LR to: {lr:0.8f}")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def next(self, epoch_number):
        lr = self.current_lr
        change = False

        for eidx, e in enumerate(self.epochs):
            if epoch_number == e:
                change = True
                lr = lr * self.gammas[eidx]

        if change:
            self.set_lr(lr)
            self.current_lr = lr