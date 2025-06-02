import traceback
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import os
import pickle
from deducer import PPODeducer
from yaml_parser import YamlParser


# 决策补偿模块
class Compensator:
    def __init__(self, config):
        """初始化补偿器
        Args:
            config: 配置参数
        """
        self.config = config
        self.device = torch.device("cuda")  # 使用GPU设备
        # 初始化为标准正态分布(均值0, 标准差1)
        self.motor_error = (0, 1)  # 电机控制误差分布参数(均值,标准差)
        self.steer_error = (0, 1)  # 转向控制误差分布参数(均值,标准差)
        self.action_delay_step = 0  # 动作延迟步数

    def update_compensator(self, motor_error, steer_error, action_delay_step):
        """更新补偿器参数
        Args:
            motor_error: 新的电机误差参数(均值,标准差)
            steer_error: 新的转向误差参数(均值,标准差)
            action_delay_step: 新的动作延迟步数
        """
        self.motor_error = motor_error
        self.steer_error = steer_error
        self.action_delay_step = action_delay_step

    def get_enhanced_action(self, action):
        """获取添加噪声后的增强动作
        Args:
            action: 原始动作向量
        Returns:
            增强后的动作
        """
        enhanced_action = action.copy()  # 创建副本避免修改原动作
        # 从当前误差分布中采样噪声
        motor = np.random.normal(*self.motor_error)  # 电机噪声
        steer = np.random.normal(*self.steer_error)  # 转向噪声
        # 添加噪声到动作
        enhanced_action[0] += motor  # 电机控制量添加噪声
        enhanced_action[1] += steer  # 转向控制量添加噪声
        return enhanced_action


class TrajectoryDataset(Dataset):
    """轨迹数据集类，用于处理和加载轨迹数据"""

    def __init__(self, trojectoties):
        """初始化轨迹数据集
        Args:
            trojectoties: 轨迹列表，每条轨迹包含多个时间步的状态-动作序列
        """
        self.sasasass = []  # 存储处理后的轨迹片段

        # 遍历每条轨迹
        for trajectory in trojectoties:
            # 以步长2滑动窗口处理轨迹，生成5元组片段(a,a,a,a,a)
            for i in range(2, len(trajectory) - 6, 2):
                # 提取5个连续时间步的数据
                sasasas = (
                    trajectory[i],    # 状态t
                    trajectory[i+1],  # 动作t
                    trajectory[i+2],  # 状态t+1
                    trajectory[i+3],  # 动作t+1
                    trajectory[i+4],  # 状态t+2
                )

                # 将5元组拼接为1维数组
                sasasas_array = np.hstack(sasasas)
                self.sasasass.append(sasasas_array)

        print("处理后轨迹片段数量:", len(self.sasasass))

    def __len__(self):
        """返回数据集大小"""
        return len(self.sasasass)

    def __getitem__(self, idx):
        """获取指定索引的轨迹片段
        Args:
            idx: 数据索引
        Returns:
            对应索引的轨迹片段数组
        """
        return self.sasasass[idx]


# 基于高斯过程的参数生成器，用于生成和优化强化学习中的环境参数
class Generator:
    def __init__(self):
        """初始化高斯过程生成器"""
        # 定义高斯过程核函数：常数核 * RBF核
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # 创建高斯过程回归器
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,  # 噪声水平
            n_restarts_optimizer=10  # 优化器重启次数
        )

        # 初始化样本数据
        self.X_init = np.array([[1.0, 1.0]])  # 初始输入(2维参数)
        self.Y_init = np.array([1.98]).reshape(-1, 1)  # 初始输出(性能得分)
        # 拟合高斯过程模型
        self.gp.fit(self.X_init, self.Y_init)

    def generate_param(self):
        """生成新的参数组合
        Returns:
            np.array: 最优参数组合(2维)
        """
        # 定义获取函数(acquisition function)
        def acquisition(x):
            x = np.atleast_2d(x)
            mu, sigma = self.gp.predict(x, return_std=True)
            return -(mu + 1.96 * sigma)  # 置信上限(UCB)

        # 定义参数搜索空间边界
        bounds = np.array([[0, 10.0], [0, 10.0]])
        best_res = None
        # 随机初始化10次，寻找最优参数
        for _ in range(10):
            x0 = np.random.uniform(0, 4.0, size=(2,))  # 随机初始点
            res = minimize(acquisition, x0=x0, bounds=bounds)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        return best_res.x

    def update(self, x, y):
        """用新数据更新高斯过程模型
        Args:
            x: 新参数(2维)
            y: 对应性能得分
        """
        # 添加新样本数据
        self.X_init = np.vstack((self.X_init, x))
        self.Y_init = np.vstack((self.Y_init, y))
        # 重新拟合模型
        self.gp.fit(self.X_init, self.Y_init)


# 判别器模型，用于区分真实轨迹和模拟轨迹
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """初始化判别器
        Args:
            input_dim: 输入特征维度
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim  # 保存输入维度

        # 定义神经网络结构
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),  # 全连接层(输入层)
            nn.LeakyReLU(0.2),  # LeakyReLU激活函数(负斜率0.2)
            nn.Linear(512, 256),  # 全连接层(隐藏层)
            nn.LeakyReLU(0.2),  # LeakyReLU激活函数
            nn.Linear(256, 1),  # 全连接层(输出层)
            nn.Sigmoid(),  # Sigmoid激活函数(输出0-1概率)
        )

    def forward(self, x):
        """前向传播过程
        Args:
            x: 输入特征张量
        Returns:
            判别结果(0-1之间的概率值)
        """
        out = self.model(x)  # 通过神经网络
        return out  # 返回判别结果


def train_GAN(run_id, model_id, tuple_length=3, input_dim=30, num_epochs=500, batch_size=64, collect_sim_episode_num=20,
              discriminator_save_path="pkl/discriminator.pkl"):
    """训练生成对抗网络(GAN)来优化补偿模块参数
    Args:
        run_id: 实验运行的唯一标识符
        model_id: 模型标识符
        tuple_length: 轨迹元组长度，默认为3
        input_dim: 输入特征维度，默认为30
        num_epochs: 训练轮数，默认为500
        batch_size: 批大小，默认为64
        collect_sim_episode_num: 每轮收集的模拟轨迹数量，默认为20
        discriminator_save_path: 判别器模型保存路径
    """
    try:
        # 初始化最佳参数和得分记录
        best_param = None  # 保存最佳参数组合
        best_score = None  # 保存最佳得分

        # 初始化生成器和判别器
        generator = Generator()  # 创建参数生成器
        discriminator = Discriminator(input_dim)  # 创建轨迹判别器

        # 读取已有的参数和分数更新生成器
        param_score_save_dict = dict()  # 参数-得分字典
        param_score_save_path = "pkl/param_score_%s.pkl" % run_id  # 参数保存路径
        if os.path.exists(param_score_save_path):
            print("加载已有参数得分记录...")
            with open(param_score_save_path, "rb") as f:
                param_score_save_dict = pickle.load(f)  # 加载历史记录
        # 用历史数据更新生成器
        for key, value in param_score_save_dict.items():
            param = np.array(key)
            generator.update(param, value["score"])  # 更新生成器模型
            print("加载参数:", param)

        # 加载已有的判别器模型
        if os.path.exists(discriminator_save_path):
            print("加载已有判别器模型...")
            discriminator.load_state_dict(torch.load(discriminator_save_path))

        # 开始训练GAN
        with open("log.txt", "w", encoding="utf-8") as f:
            f.write("")  # 清空日志文件
        for i in range(len(param_score_save_dict), num_epochs):
            # 1. 生成新的环境参数组合
            param_dist = generator.generate_param()  # 从生成器获取新参数

            # 2. 加载真实轨迹数据集
            trajectory_file_path = "trajectories/UGVRace-20-2-125_125_1.pkl"
            with open(trajectory_file_path, "rb") as f:
                realTrojectoties = pickle.load(f)  # 加载真实轨迹数据
            # 创建真实轨迹数据集
            realTrajectoryDataset = TrajectoryDataset(realTrojectoties)
            # 调整数据集大小使其能被batch_size整除
            realTrajectoryDataset.sasasass = realTrajectoryDataset.sasasass[
                                             : len(realTrajectoryDataset) // batch_size * batch_size
                                             ]
            # 创建真实轨迹数据加载器
            realTrajectoryDataLoader = DataLoader(
                realTrajectoryDataset,
                batch_size=batch_size,
                shuffle=True,  # 打乱数据顺序
                num_workers=0,  # 不使用多进程加载
                generator=torch.Generator(device="cuda:0")  # 使用CUDA随机数生成器
            )

            # 3. 使用新参数创建虚拟环境
            config = YamlParser("configs/car1.yaml").get_config()  # 加载环境配置
            device = torch.device("cuda")  # 使用GPU设备
            accelerate = False  # 启用加速模式
            steer_error, motor_error = param_dist[0], param_dist[1]  # 解包参数
            action_delay_step = 1  # 设置动作延迟步数
            # 创建PPO推理器，用于生成模拟轨迹
            deducer = PPODeducer(
                config,
                model_id,
                device=device,
                accelerate=accelerate,
                steer_error=steer_error,  # 设置转向误差参数
                motor_error=motor_error,  # 设置电机误差参数
                action_delay_step=action_delay_step,  # 设置动作延迟
            )

            # 4. 采集模拟轨迹数据
            simTrajectoryDataset = TrajectoryDataset([])  # 初始化空数据集
            # 目标长度设为与真实数据集相同
            simTrajectoryTargetLength = len(realTrajectoryDataLoader.dataset)
            # 持续采集直到达到目标数量
            while len(simTrajectoryDataset) < simTrajectoryTargetLength:
                reward_mean, rewards, _simTrojectoties = deducer.run_deduce(
                    episode_num=collect_sim_episode_num  # 每次采集20条轨迹
                )
                _simTrajectoryDataset = TrajectoryDataset(_simTrojectoties)
                simTrajectoryDataset.sasasass.extend(_simTrajectoryDataset.sasasass)
                print("模拟轨迹采集进度: %d/%d" % (len(simTrajectoryDataset), simTrajectoryTargetLength))
            # 截断到目标长度
            simTrajectoryDataset.sasasass = simTrajectoryDataset.sasasass[
                                            :simTrajectoryTargetLength
                                            ]
            deducer.close()  # 关闭推理器

            # 5. 创建模拟轨迹数据加载器
            simTrajectoryDataLoader = DataLoader(
                simTrajectoryDataset,
                batch_size=batch_size,
                shuffle=True,  # 打乱数据顺序
                num_workers=0,  # 不使用多进程加载
                generator=torch.Generator(device="cuda:0")  # 使用CUDA随机数生成器
            )

            # 定义二元交叉熵损失函数
            criterion = nn.BCELoss()

            # 初始化判别器优化器(Adam优化器，学习率0.001)
            discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=0.001
            )

            generator_total_rewards = []  # 存储生成器的奖励

            # 同时遍历模拟和真实轨迹数据
            for sim_sas_batch, real_sas_batch in zip(
                    simTrajectoryDataLoader, realTrajectoryDataLoader
            ):
                # 合并模拟和真实数据
                sas_batch = torch.cat((sim_sas_batch, real_sas_batch), 0)
                # 创建标签(0表示模拟数据，1表示真实数据)
                label = torch.cat(
                    (
                        torch.zeros(sim_sas_batch.size()[0]),  # 模拟数据标签
                        torch.ones(real_sas_batch.size()[0]),  # 真实数据标签
                    ),
                    0,
                )
                # 随机采样batch_size大小的数据
                rand_idx = torch.randperm(sas_batch.size()[0])[:batch_size]
                sas_batch = sas_batch[rand_idx]
                label = label[rand_idx]

                # 6. 更新判别器
                # 清空梯度
                discriminator_optimizer.zero_grad()
                # 前向传播
                sas_output = discriminator(sas_batch)
                sas_output = sas_output.squeeze()  # 去除多余的维度
                # 计算判别器损失
                discriminator_loss = criterion(sas_output, label)
                # 反向传播和参数更新
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # 记录生成器性能(判别器对模拟数据的输出)
                generator_rewards = sas_output.detach().cpu().numpy().tolist()
                generator_total_rewards.extend(generator_rewards)

                # 保存判别器模型(每个batch都保存)
                torch.save(discriminator.state_dict(), discriminator_save_path)

            # 7. 更新生成器
            # 计算当前epoch生成器的平均得分(判别器对模拟数据的平均输出)
            generator_total_rewards_avg = np.mean(generator_total_rewards)
            # 打印训练进度信息
            print(
                "epoch %d, param_dist %s, score %f"
                % (i, param_dist, generator_total_rewards_avg)
            )

            # 用当前参数和得分更新生成器的高斯过程模型
            generator.update(param_dist, generator_total_rewards_avg)

            # 将当前参数组合转换为元组作为字典键
            key = tuple(param_dist)
            # 记录当前参数的性能得分
            param_score_save_dict[key] = {
                "index": len(param_score_save_dict),  # 记录索引序号
                "score": generator_total_rewards_avg,  # 记录得分
            }
            # 保存参数得分记录到文件
            with open(param_score_save_path, "wb") as f:
                pickle.dump(param_score_save_dict, f)

            # 遍历所有参数记录，寻找当前最佳参数
            for param, value in param_score_save_dict.items():
                score = value["score"]
                # 更新最佳参数记录
                if best_score is None or score > best_score:
                    best_param = param
                    best_score = score

            # 打印当前最佳参数信息
            print("best_param", best_param)
            print("best_score", best_score)
            print("-----------------------------------")

    except:
        # 捕获并打印异常信息
        print(traceback.format_exc())

    # 训练结束后保存最终的判别器模型
    print("save discriminator model")
    torch.save(discriminator.state_dict(), discriminator_save_path)

    # 返回训练得到的最佳参数和对应得分
    return best_param, best_score


def main():
    model_id = "UGVRace-20-2"
    run_id = "UGVRace-20-2-3-重复7次"
    tuple_length = 3
    input_dim = 6 * (tuple_length + 1) + 2 * tuple_length
    num_epochs = 500
    batch_size = 64
    collect_sim_episode_num = 20
    discriminator_save_path = "pkl/discriminator.pkl"
    train_GAN(run_id, model_id, tuple_length, input_dim, num_epochs, batch_size, collect_sim_episode_num,
              discriminator_save_path)


if "__main__" == __name__:
    main()
