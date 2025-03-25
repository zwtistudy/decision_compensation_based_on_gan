# 面向自主搜索任务的智能体行为策略虚实迁移方法

## 容器训练

尝试将代码部署到楼下计算集群，基于镜像`harbor.n705.work/n705/zwt@sha256:84c5`

如果是第一次运行，需要修改执行权限

```bash
chmod -R 755 environments/BuildUGVRace-CollectingTrajectories/UGVRace.x86_64
```

创建完容器需要安装依赖包，利用clash代理网络

```bash
export http_proxy=http://58.199.163.81:7897
export https_proxy=https://58.199.163.81:7897
pip install opencv-python tblib ruamel.yaml psutil scikit-learn
```

设置虚拟显示

```bash
export DISPLAY=:0
```

注意检查连接mlagents时加入`--force-vulkan`参数。

### 安装依赖

安装虚拟环境：

```bash
conda create -n m python=3.9 -y
```

安装依赖包：

```bash
pip install -r requirements.txt
```

### Could not resolve host: gitee.com

修改hosts：`/etc/hosts`

### VPN使用环境下Failed to connect to gitee.com port 443 after 3 ms: Connection refuse

```bash
git config --global http.proxy 58.199.163.81:7897
git config --global https.proxy 58.199.163.81:7897
```

注意

在FinalShell上可以正常训练，在VSCode上不可以，会连不上mlagents环境。

## 其他工具命令

查看/终止mlagents进程：

```bash
ps aux | grep mlagents
ps aux | grep r4_IGAN.py
ps aux | grep mlagents | cut -c 9-15 | xargs kill -9
ps aux | grep main.py | cut -c 9-15 | xargs kill -9
ps aux | grep r4_IGAN.py | cut -c 9-15 | xargs kill -9
```

取消代理：

```bash
export http_proxy=""
export https_proxy=""
```

## 配置Gym Car Race环境

### 配置box2d

下载swig：https://www.swig.org/download.html

配置环境变量。

安装box2d

```bash
pip install box2d-py
pip install opencv-contrib-python
```

## 配置TORCS环境

下载torcs：https://sourceforge.net/projects/torcs/files/all-in-one/。直接下载，然后安装。

下载服务端平台：[https://sourceforge.net/projects/cig/files/SCR%20Championship/Server%20Windows/2.0/](https://sourceforge.net/projects/cig/files/SCR Championship/Server Windows/2.0/)，下载完解压缩并替换进torcs的安装目录。

wtorcs.exe即平台的入口地址。可将其创建快捷方式。

服务端平台说明文档：https://www.yumpu.com/en/document/view/48269886/simulated-car-racing-championship-competition-software-manual#

安装gym_torcs库

```bash
pip install gym_torcs
```

## 演示设置

### UGVRace

```bash
cd F:\code\rl\recurrent-ppo-truncated-bptt2; F:\condaenvs\mlagents\python.exe deduce.py
```

deduce.yaml:

```yaml
config_file: configs/car1.yaml
run_id: UGVRace_Ours
record_id: test
log_data: True
accelerate: False
episode_num: 200
cpu: False
continue_record: False
```

car1.yaml:

```yaml
file_name: environments/BuildUGVRace-OneObstacal/RLEnvironments.exe
```

### UGVSearch

```bash
cd F:\code\rl\recurrent-ppo-truncated-bptt2; F:\condaenvs\mlagents\python.exe main.py ugv/ugv_search_mg --name 20240713002100dddd --ckpt 150189 --run --port 17635
```

### GYMRace

```bash
cd F:\code\rl\recurrent-ppo-truncated-bptt2; F:\condaenvs\mlagents\python.exe deduce_gym.py
```

