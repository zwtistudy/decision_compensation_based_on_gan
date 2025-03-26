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
        """
        初始化补偿器实例，用于调整决策输出以补偿模拟与现实环境之间的差异。

        Args:
            config: 配置文件，包含补偿器所需的参数。
        """
        self.config = config
        self.device = torch.device("cuda")  # 设置计算设备为GPU
        # 初始化线性变换参数（单位矩阵 + 零偏置）
        self.m_w = np.eye(2)  # 电机线性变换的权重矩阵，初始化为单位矩阵 [[1, 0], [0, 1]]
        self.m_b = np.zeros(2)  # 电机线性变换的偏置向量，初始化为零向量 [0, 0]
        self.s_w = np.eye(2)  # 转向线性变换的权重矩阵，初始化为单位矩阵
        self.s_b = np.zeros(2)  # 转向线性变换的偏置向量，初始化为零向量
        self.action_delay_step = 1  # 动作延迟步数，用于模拟现实环境中的延迟

    def update_compensator(self, m_w, m_b, s_w, s_b, action_delay_step):
        """
        更新补偿器的参数，包括电机和转向的线性变换参数以及动作延迟步数。

        Args:
            m_w: 电机线性变换的新权重矩阵。
            m_b: 电机线性变换的新偏置向量。
            s_w: 转向线性变换的新权重矩阵。
            s_b: 转向线性变换的新偏置向量。
            action_delay_step: 新的动作延迟步数。
        """
        self.m_w = m_w
        self.m_b = m_b
        self.s_w = s_w
        self.s_b = s_b
        self.action_delay_step = action_delay_step

    # 获取增强后的决策
    def get_enhanced_action(self, action):
        """
        根据当前误差分布对决策进行补偿，生成增强后的决策。

        Args:
            action: 原始决策，由策略网络生成。

        Returns:
            增强后的决策，考虑了电机和转向的误差，用于实际执行。
        """
        # 计算增强后的决策
        enhanced_action = np.dot(self.m_w, action) + self.m_b  # 应用电机线性变换
        return enhanced_action



class TrajectoryDataset(Dataset):
    def __init__(self, trojectoties):
        """初始化轨迹数据集
        Args:
            trojectoties: 轨迹数据列表，用于构建虚实一致性损失
        """
        self.sasasass = []  # 存储处理后的轨迹数据
        for trajectory in trojectoties:
            # 遍历轨迹，以步长2滑动窗口提取状态-动作序列
            for i in range(2, len(trajectory) - 6, 2):
                # 提取连续的状态-动作序列，用于量化虚实差异
                sasasas = (
                    trajectory[i],  # 状态s_t
                    trajectory[i + 1],  # 动作a_t
                    trajectory[i + 2],  # 状态s_{t+1}
                    trajectory[i + 3],  # 动作a_{t+1}
                    trajectory[i + 4],  # 状态s_{t+2}
                    trajectory[i + 5],  # 动作a_{t+2}
                    trajectory[i + 6],  # 状态s_{t+3}
                )

                # 堆叠为一个array，用于GAN训练
                sasasas_array = np.hstack(sasasas)
                self.sasasass.append(sasasas_array)
        print("len(sasasass)", len(self.sasasass))

    def __len__(self):
        """返回数据集大小"""
        return len(self.sasasass)

    def __getitem__(self, idx):
        """获取指定索引的轨迹数据
        Args:
            idx: 数据索引
        Returns:
            状态-动作序列数组，用于判别器训练
        """
        return self.sasasass[idx]


class Generator:
    def __init__(self):
        """初始化生成器，使用高斯过程回归生成四维补偿参数"""
        # 扩展核函数到四维参数空间
        kernel = (C(1.0, (1e-3, 1e3)) * RBF([10,10,10,10], (1e-2, 1e2)))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-5, n_restarts_optimizer=10
        )

        # 初始化四维参数样本（单位矩阵 + 零偏置）
        self.X_init = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1]])
        self.Y_init = np.array([1.98]).reshape(-1, 1)
        self.gp.fit(self.X_init, self.Y_init)

    def generate_param(self):
        """生成四维补偿参数和延迟步数"""
        def acquisition(x):
            x = np.atleast_2d(x)
            mu, sigma = self.gp.predict(x, return_std=True)
            return -(mu + 1.96 * sigma)

        # 扩展搜索空间到四维参数
        bounds = [
            [0, 10.0], [0, 10.0], [0, 10.0], [0, 10.0],  # m_w
            [-5, 5], [-5, 5],  # m_b
            [0, 10.0], [0, 10.0],  # s_w
            [1, 5]  # action_delay_step (整数步)
        ]

        best_res = None
        for _ in range(10):
            # 随机初始化时保持矩阵对称性
            x0 = [
                np.random.uniform(0, 4.0), 0,  # m_w[0][0], m_w[0][1]
                0, np.random.uniform(0, 4.0),  # m_w[1][0], m_w[1][1]
                np.random.uniform(-2, 2), np.random.uniform(-2, 2),  # m_b
                np.random.uniform(0, 4.0), 0,  # s_w[0][0], s_w[0][1]
                0, np.random.uniform(0, 4.0),  # s_w[1][0], s_w[1][1]
                np.random.randint(1, 5)  # action_delay_step
            ]
            res = minimize(acquisition, x0=x0, bounds=bounds)
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        # 将参数转换为补偿器需要的格式
        params = best_res.x
        return {
            'm_w': [[params[0], params[1]], [params[2], params[3]]],
            'm_b': [params[4], params[5]],
            's_w': [[params[6], params[7]], [params[8], params[9]]],
            's_b': [params[10], params[11]],
            'action_delay_step': int(params[12])
        }

    def update(self, param_dist, generator_total_rewards_avg):
        """更新高斯过程回归模型"""
        self.X_init = np.vstack((self.X_init, param_dist))
        self.Y_init = np.vstack((self.Y_init, generator_total_rewards_avg))
        self.gp.fit(self.X_init, self.Y_init)

    def update_compensator(self, compensator):
        """更新补偿器参数"""
        params = self.generate_param()
        compensator.update_compensator(
            m_w=params['m_w'],
            m_b=params['m_b'],
            s_w=params['s_w'],
            s_b=params['s_b'],
            action_delay_step=params['action_delay_step']
        )


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """初始化判别器，用于区分真实和模拟的决策序列
        Args:
            input_dim: 输入维度，即状态-动作序列的维度
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        # 定义判别器网络结构，用于计算虚实一致性损失
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),  # 全连接层，提取特征
            nn.LeakyReLU(0.2),  # 激活函数，引入非线性
            nn.Linear(512, 256),  # 全连接层，进一步提取特征
            nn.LeakyReLU(0.2),  # 激活函数
            nn.Linear(256, 1),  # 输出层，计算虚实概率
            nn.Sigmoid(),  # Sigmoid激活，将输出映射到[0,1]区间
        )

    def forward(self, x):
        """前向传播，计算输入序列的虚实概率
        Args:
            x: 输入的状态-动作序列
        Returns:
            虚实概率值，用于量化虚实差异
        """
        out = self.model(x)
        return out


def train_GAN(run_id, model_id, tuple_length=3, input_dim=30, num_epochs=500, batch_size=64, collect_sim_episode_num=20,
              discriminator_save_path="pkl/discriminator.pkl"):
    """训练生成对抗网络，构建虚实一致性损失并优化决策补偿模块
    Args:
        run_id: 运行标识符，用于区分不同实验
        model_id: 模型标识符，指定使用的模型类型
        tuple_length: 状态-动作序列的长度
        input_dim: 输入维度，即状态-动作序列的维度
        num_epochs: 训练的总轮数
        batch_size: 每次训练的批量大小
        collect_sim_episode_num: 每次采集的模拟轨迹数量
        discriminator_save_path: 判别器模型的保存路径
    Returns:
        best_param: 最优参数，用于补偿模块的误差调整
        best_score: 最高得分，表示虚实一致性程度
    """
    try:
        # 初始化最优参数和得分
        best_param = None
        best_score = None

        # 初始化生成器和判别器
        generator = Generator()  # 生成器用于生成模拟环境参数
        discriminator = Discriminator(input_dim)  # 判别器用于区分真实和模拟轨迹

        # 读取已有的参数和分数
        param_score_save_dict = dict()  # 创建字典存储参数和对应得分
        param_score_save_path = "pkl/param_score_%s.pkl" % run_id  # 构建参数得分文件路径
        if os.path.exists(param_score_save_path):  # 检查参数得分文件是否存在
            print("load param_score_save_dict")  # 打印加载提示信息
            with open(param_score_save_path, "rb") as f:  # 以二进制读取模式打开文件
                param_score_save_dict = pickle.load(f)  # 加载参数得分字典
        for key, value in param_score_save_dict.items():  # 遍历参数得分字典
            param = np.array(key)  # 将参数转换为numpy数组
            generator.update(param, value["score"])  # 使用历史数据更新生成器
            print("param", param)  # 打印当前参数信息

        # 加载预训练的判别器模型
        if os.path.exists(discriminator_save_path):
            discriminator.load_state_dict(torch.load(discriminator_save_path))
        with open("log.txt", "w", encoding="utf-8") as f:
            f.write("")

        # 开始训练循环
        for i in range(len(param_score_save_dict), num_epochs):
            # 生成一组新的模拟环境参数
            param_dist = generator.generate_param()

            # 加载真实轨迹数据
            trajectory_file_path = "trajectories/UGVRace-20-2-125_125_1.pkl"  # 定义真实轨迹文件路径
            with open(trajectory_file_path, "rb") as f:  # 以二进制读取模式打开文件
                realTrojectoties = pickle.load(f)  # 加载真实轨迹数据
            realTrajectoryDataset = TrajectoryDataset(realTrojectoties)  # 创建真实轨迹数据集
            realTrajectoryDataset.sasasass = realTrajectoryDataset.sasasass[
                                             : len(realTrajectoryDataset) // batch_size * batch_size
                                             ]  # 调整数据集大小使其能被batch_size整除
            realTrajectoryDataLoader = DataLoader(
                realTrajectoryDataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator(device="cuda:0")
            )  # 创建真实轨迹数据加载器

            # 更新虚拟环境参数
            config = YamlParser("configs/car1.yaml").get_config()  # 加载配置文件
            device = torch.device("cuda")  # 设置计算设备为GPU
            accelerate = True  # 启用加速模式
            steer_error, motor_error = param_dist[0], param_dist[1]  # 获取转向和电机误差参数
            action_delay_step = 1  # 设置动作延迟步数
            deducer = PPODeducer(
                config,
                model_id,
                device=device,
                accelerate=accelerate,
                steer_error=steer_error,
                motor_error=motor_error,
                action_delay_step=action_delay_step,
            )  # 创建PPO推理器实例

            # 采集模拟轨迹数据
            simTrajectoryDataset = TrajectoryDataset([])  # 创建空模拟轨迹数据集
            simTrajectoryTargetLength = len(realTrajectoryDataLoader.dataset)  # 获取目标数据集长度
            while len(simTrajectoryDataset) < simTrajectoryTargetLength:  # 循环采集直到达到目标长度
                reward_mean, rewards, _simTrojectoties = deducer.run_deduce(
                    episode_num=collect_sim_episode_num
                )  # 运行推理器采集模拟轨迹
                _simTrajectoryDataset = TrajectoryDataset(_simTrojectoties)  # 创建临时模拟轨迹数据集
                simTrajectoryDataset.sasasass.extend(_simTrajectoryDataset.sasasass)  # 扩展模拟轨迹数据集
                print("%d/%d" % (len(simTrajectoryDataset), simTrajectoryTargetLength))  # 打印采集进度
            simTrajectoryDataset.sasasass = simTrajectoryDataset.sasasass[
                                            :simTrajectoryTargetLength
                                            ]  # 截取模拟轨迹数据集到目标长度
            deducer.close()  # 关闭推理器

            # 创建模拟轨迹数据加载器
            simTrajectoryDataLoader = DataLoader(
                simTrajectoryDataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator(device="cuda:0")
            )

            # 定义损失函数和优化器
            criterion = nn.BCELoss()  # 二元交叉熵损失
            discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=0.001
            )

            # 训练判别器
            generator_total_rewards = []
            for sim_sas_batch, real_sas_batch in zip(
                    simTrajectoryDataLoader, realTrajectoryDataLoader
            ):
                # 合并真实和模拟数据
                sas_batch = torch.cat((sim_sas_batch, real_sas_batch), 0)
                label = torch.cat(
                    (
                        torch.zeros(sim_sas_batch.size()[0]),  # 模拟数据标签为0
                        torch.ones(real_sas_batch.size()[0]),  # 真实数据标签为1
                    ),
                    0,
                )
                # 随机采样
                rand_idx = torch.randperm(sas_batch.size()[0])[:batch_size]
                sas_batch = sas_batch[rand_idx]
                label = label[rand_idx]

                # 训练判别器
                discriminator_optimizer.zero_grad()
                sas_output = discriminator(sas_batch)
                sas_output = sas_output.squeeze()
                discriminator_loss = criterion(sas_output, label)
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # 记录生成器得分
                generator_rewards = sas_output.detach().cpu().numpy().tolist()
                generator_total_rewards.extend(generator_rewards)

                # 保存判别器模型
                torch.save(discriminator.state_dict(), discriminator_save_path)

            # 计算生成器平均得分
            generator_total_rewards_avg = np.mean(generator_total_rewards)
            print(
                "epoch %d, param_dist %s, score %f"
                % (i, param_dist, generator_total_rewards_avg)
            )

            # 更新生成器参数
            generator.update(param_dist, generator_total_rewards_avg)  # 使用当前参数和得分更新生成器
            key = tuple(param_dist)  # 将参数转换为元组作为字典键
            param_score_save_dict[key] = {
                "index": len(param_score_save_dict),  # 记录当前参数索引
                "score": generator_total_rewards_avg,  # 记录当前参数得分
            }  # 将参数和得分存入字典
            with open(param_score_save_path, "wb") as f:  # 以二进制写入模式打开文件
                pickle.dump(param_score_save_dict, f)  # 保存参数得分字典

            # 更新最优参数
            for param, value in param_score_save_dict.items():  # 遍历参数得分字典
                score = value["score"]  # 获取当前参数得分
                if best_score is None or score > best_score:  # 检查是否为当前最优得分
                    best_param = param  # 更新最优参数
                    best_score = score  # 更新最高得分
            print("best_param", best_param)  # 打印当前最优参数
            print("best_score", best_score)  # 打印当前最高得分
            print("-----------------------------------")  # 打印分隔线
    except:
        print(traceback.format_exc())
    # 保存最终判别器模型
    print("save discriminator model")
    torch.save(discriminator.state_dict(), discriminator_save_path)
    return best_param, best_score


def main():
    """主函数，用于启动GAN训练过程"""
    model_id = "UGVRace-20-2"  # 定义模型标识符
    run_id = "UGVRace-20-2-3-重复7次"  # 定义运行标识符
    tuple_length = 3  # 设置状态-动作序列长度
    input_dim = 6 * (tuple_length + 1) + 2 * tuple_length  # 计算输入维度
    num_epochs = 500  # 设置训练总轮数
    batch_size = 64  # 设置批量大小
    collect_sim_episode_num = 20  # 设置每次采集的模拟轨迹数量
    discriminator_save_path = "pkl/discriminator.pkl"  # 定义判别器模型保存路径
    train_GAN(run_id, model_id, tuple_length, input_dim, num_epochs, batch_size, collect_sim_episode_num,
              discriminator_save_path)  # 调用训练函数


if "__main__" == __name__:
    main()
