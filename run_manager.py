from tqdm import tqdm


class RunManager(object):
    def __init__(self, model_manager, optimizer, loaders, criterion, device):
        self.model_manager = model_manager
        self.optimizer = optimizer
        self.loaders = loaders
        self.criterion = criterion
        self.device = device

    def train(self):
        loss_ = 0.0
        acc = 0.0
        loader = self.loaders['train']
        self.model_manager.set_train_mode()
        self.optimizer.zero_grad()
        for idx, (feat, _, label) in enumerate(tqdm(loader)):

            output, loss = self.model_manager.forward(feat, label, self.criterion)

            loss.backward()

            loss_ += loss.item()

            acc += 1 if output.argmax().item() == label.item() else 0

            if (idx + 1) % int(len(loader)/3) == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                tqdm.write("Train [{}]/[{}]: {:.2f} -- {:.2f}".format(idx, len(loader), loss_ / idx, acc / idx))

    def val(self, test):
        loss_ = 0.0
        acc = 0.0
        if test:
            loader = self.loaders['test']
        else:
            loader = self.loaders['val']
        self.model_manager.set_eval_mode()

        for idx, (feat, _, label) in enumerate(tqdm(loader)):

            output, loss = self.model_manager.forward(feat, label, self.criterion)

            loss_ += loss.item()

            acc += 1 if output.argmax().item() == label.item() else 0

        if test:
            print("=" * 50)
            print("=" * 10, "Test: {:.2f} -- {:.5f}".format(loss_ / len(loader), acc / len(loader)), "=" * 10)
            print("=" * 50)
        else:
            tqdm.write("Validation: {:.2f} -- {:.5f}".format(loss_ / len(loader), acc / len(loader)))

        return acc / len(loader)
