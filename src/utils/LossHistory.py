import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, log_dir):
        """
        以日志文件夹保存路径初始化
        """
        import datetime
        curr_time               = datetime.datetime.now()
        time_str                = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir            = log_dir
        self.time_str           = time_str
        self.save_path          = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses             = []
        self.val_loss           = []
        self.iter_loss          = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        """
        添加训练集与测试集损失函数值,并绘图保存
        """
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        """
        绘制训练集与测试集损失函数曲线
        """
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'blue', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'cornflowerblue', linewidth = 2, label='val loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def append_iter_loss(self,loss):
        self.iter_loss.append(loss)
        with open(os.path.join(self.save_path, "iter_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

    def iter_loss_plot(self):
        """
        绘制训练集与测试集损失函数曲线
        """
        iters = range(len(self.iter_loss))

        plt.figure()
        plt.plot(iters, self.iter_loss, 'blue', linewidth = 2, label='train loss')
        plt.grid(True)
        plt.xlabel('iteration')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, "iteration_loss.png"))
        plt.cla()
        plt.close("all")
        print('save'+os.path.join(self.save_path, "iteration_loss.png"))