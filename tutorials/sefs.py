import numpy as np
import scipy
import torch
from torch import optim, nn

from compfs.models.sefs import CompFS
from tutorials.adni_data import adni_data, ADNIDataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_data, val_data = adni_data()
m_mb = []
for i in range(2):
    x = train_data[i].data
    cov = np.corrcoef(x.T)
    remove_idx = []

    cov_new = np.delete(cov, remove_idx, axis=0)
    cov_new = np.delete(cov_new, remove_idx, axis=1)
    max_iterations = 100  # 最大迭代次数
    epsilon = 1e-8  # 收敛判据

    # 获取矩阵a的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_new)

    # 开始迭代修正过程
    for i in range(max_iterations):
        if np.all(eigenvalues > 0):
            break

        # 计算修正项
        delta = -np.min(eigenvalues) + epsilon

        # 构造修正后的矩阵
        cov_new = cov_new + delta * np.eye(cov_new.shape[0])

        # 重新计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_new)
    L = scipy.linalg.cholesky(cov_new, lower=True)
    x_dim = np.shape(x)[1]
    def mask_generation(mb_size_, pi_=5):
        '''
            Phi(x; mu, sigma) = 1/2 * (1 + erf( (x-mu)/(sigma * sqrt(2)) ))
            --> Phi(x; 0,1)   = 1/2 * (1 + erf( x/sqrt(2) ))
        '''
        if len(remove_idx) == 0:
            epsilon = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], mb_size_])
            g = np.matmul(L, epsilon)
        else:
            present_idx = [i for i in range(x_dim) if i not in remove_idx]
            epsilon = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], mb_size_])
            g2 = np.random.normal(loc=0., scale=1., size=[len(remove_idx), mb_size_])
            g1 = np.matmul(L, epsilon)
            g = np.zeros([x_dim, mb_size_])

            g[present_idx, :] = g1
            g[remove_idx, :] = g2

        m = (1 / 2 * (1 + scipy.special.erf(g / np.sqrt(2))) < pi_).astype(float).T
        return m
    m_mb.append(mask_generation(mb_size_=128))


def accuracy(x, y):
    # Accuracy.
    acc = 100 * torch.sum(torch.argmax(x, dim=-1) == y) / len(y)
    return acc.item()


def mse(x, y):
    # MSE for regression.
    return 0.5 * torch.mean((x - y) ** 2).item()


def make_lambda_threshold(l):
    # If the value is above a certain value l (lambda) return 1, otherwise 0.
    l = float(l)

    def l_func(p):
        return p >= torch.full_like(p, l)

    return l_func


def make_std_threshold(nsigma):
    # Choose which features are relevant in p relative to other features,
    # if value of feature is above mean + n standard deviations.
    nsigma = float(nsigma)

    def std_dev_func(p):
        mean = torch.mean(p)
        std = torch.std(p)
        return p >= torch.full_like(p, (mean + nsigma * std).item())

    return std_dev_func


def make_top_k_threshold(k):
    # Choose top k features.
    k = int(k)

    def top_k(p):
        ids = torch.topk(p, k)[1]
        out = torch.zeros_like(p)
        out[ids] = 1.0
        return out.int()

    return top_k



def accuracy(x, y):
    # Accuracy.
    acc = 100 * torch.sum(torch.argmax(x, dim=-1) == y) / len(y)
    return acc.item()


def mse(x, y):
    # MSE for regression.
    return 0.5 * torch.mean((x - y) ** 2).item()


class CompFSShell:
    def __init__(self, model_config):
        self.device = device
        self.model = CompFS(model_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_config["lr"])
        self.num_epochs = model_config["num_epochs"]
        self.lr_decay = model_config["lr_decay"]
        self.batchsize = model_config["batchsize"]
        self.val_metric = model_config["val_metric"]
        self.nlearners = model_config["nlearners"]
        super().__init__()

    def train(self, train_data, val_data, m, batch_size):
        for i in range(self.nlearners):
            self.model.x_bar.append(train_data[i].get_x_bar())
        train_data = self.model.preprocess(train_data)
        val_data = self.model.preprocess(val_data)
        train_loader = ADNIDataloader(train_data[0], train_data[1], batch_size=batch_size)
        val_loader = ADNIDataloader(val_data[0], val_data[1], batch_size=batch_size)
        print("\n\nTraining for {} Epochs:\n".format(self.num_epochs))

        for epoch in range(1, self.num_epochs + 1):
            # Train an epoch.
            epoch_loss = self.train_encoder(train_loader, m)

            # Evaluate the model and save values.
            val = self.val_encoder(val_loader, m)

            # Print information.
            print(
                "Epoch: {}, Average Loss: {:.3f}, Val Metric: {:.1f}, nfeatures: {}".format(
                    epoch, epoch_loss, val
                )
            )

            # Update learning rate.
            for g in self.optimizer.param_groups:
                g["lr"] *= self.lr_decay

        for epoch in range(1, self.num_epochs + 1):
            # Train an epoch.
            epoch_loss = self.train_epoch(train_loader)

            # Evaluate the model and save values.
            val = self.calculate_val_metric(val_loader)
            nfeatures = self.model.count_features()
            # overlap = self.model.get_overlap()[0]

            # Print information.
            print(
                "Epoch: {}, Average Loss: {:.3f}, Val Metric: {:.1f}, nfeatures: {}".format(
                    epoch, epoch_loss, val, nfeatures
                )
            )

            # Update learning rate.
            for g in self.optimizer.param_groups:
                g["lr"] *= self.lr_decay

    def train_encoder(self, train_loader, m):
        avg_loss = 0
        for data1, (j, data2) in train_loader:
            # Train a model
            data2, labels = data2
            data1 = data1[1]
            data1[0] = data1[0].to(self.device)
            data1.append(data2.to(self.device))
            m_ = [torch.tensor(m[0]).to(self.device), torch.tensor(m[1]).to(self.device)]
            data = data1
            self.optimizer.zero_grad()
            loss = self.model.ss_phase(data, m_)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        self.model.update_after_epoch()
        return avg_loss / len(train_loader)

    def val_encoder(self, val_loader, m):
        avg_loss = 0
        with torch.no_grad():
            for data1, (j, data2) in val_loader:
                # Train a model
                data2, labels = data2
                data1 = data1[1]
                data1[0] = data1[0].to(self.device)
                data1.append(data2.to(self.device))
                m_ = [torch.tensor(m[0]).to(self.device), torch.tensor(m[1]).to(self.device)]
                data = data1
                loss = self.model.ss_phase(data, m_)
                avg_loss += loss.item()
        return avg_loss / len(val_loader)

    def train_epoch(self, train_loader):
        avg_loss = 0
        for data1, (j, data2) in train_loader:
            # Train a model
            data2, labels = data2
            data1 = data1[1]
            data1[0] = data1[0].to(self.device)
            data1.append(data2.to(self.device))
            labels = labels.to(self.device)
            data = data1
            self.optimizer.zero_grad()
            loss = self.model.get_loss(data, labels)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        self.model.update_after_epoch()
        return avg_loss / len(train_loader)

    def calculate_val_metric(self, val_loader):
        metric = 0
        for data1, (j, data2) in val_loader:
            # Train a model
            data2, labels = data2
            data1 = data1[1]
            data1[0] = data1[0].to(self.device)
            data1.append(data2.to(self.device))
            labels = labels.to(self.device)
            data = data1
            out = self.model.predict(data)
            metric += self.val_metric(out, labels)
        return metric / len(val_loader)

    def get_groups(self):
        return self.model.get_groups()

    def print_evaluation_info(self, val_loader):
        for data1, (j, data2) in val_loader:
            # Train a model
            data2, labels = data2
            data1 = data1[1]
            data1[0] = data1[0].to(self.device)
            data1.append(data2.to(self.device))
            labels = labels.to(self.device)
            data = data1
        self.model.print_evaluation_info(data, labels, self.val_metric)


compfs_config = {
    "lr": 0.003,
    "lr_decay": 0.99,
    "batchsize": 50,
    "num_epochs": 100,
    "loss_func": nn.CrossEntropyLoss(),
    "val_metric": accuracy,
    "in_dim": [2000, 328],
    "h_dim": 20,
    "out_dim": 3,
    "nlearners": 2,
    "threshold_func": make_lambda_threshold(0.7),
    "temp": 0.1,
    "beta_s": 4.5,
    "beta_s_decay": 0.99,
    "beta_d": 1.2,
    "beta_d_decay": 0.99,
}

if __name__ == '__main__':
    model = CompFSShell(compfs_config)
    model.train(train_data, val_data, m_mb, batch_size=128)
