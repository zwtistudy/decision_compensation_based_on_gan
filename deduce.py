import argparse
import os
import time
import traceback

import requests
import torch

from deducer import PPODeducer
from yaml_parser import YamlParser


dia_dict = {
    'UGVSearchMG_Search_Ours': [1.01, 0.99],
    'UGVSearchMG_Search_NNGAN': [1.10, 1.16],
    'UGVSearchMG_Search_Oracle': [0.83, 1.48],
    'UGVSearchMG_Search_GAN': [1.08, 2.46],
    'UGVSearchMG_Search_DR': [1.00, 1.00],
    'UGVSearchMG_Search_BO': [0.96, 0.94],

    'UGVRace_Ours': [1.073813, 0.703697],
    'UGVRace_NNGAN': [1.010828, 0.790905],
    'UGVRace_Oracle': [0.979361, 0.683994],
    'UGVRace_GAN': [1.335519, 0.509698],
    'UGVRace_DR': [1.00, 1.00],
    'UGVRace_BO': [1.031196, 0.632981],
}


def fmsgToQQ(user_id, title, desp):
    if user_id is None or user_id == "":
        return
    desp = desp.strip()
    text = title + "\n" + desp
    url = f"http://104.128.88.206:5701/send_msg"
    payload = {"user_id": user_id, "message": text}
    for i in range(3):
        try:
            response = requests.post(url=url, params=payload, timeout=1)
            # print(response.text)
            js = response.json()
            if js["data"] is not None:
                break
        except:
            print("消息推送失败，正在重试！")
            print(traceback.format_exc())
        time.sleep(0.01)


def read_argv():
    # Command line arguments
    argv = argparse.ArgumentParser()
    argv.add_argument('--config', type=str, default='./configs/cartpole.yaml', help='Path to the yaml config file')
    argv.add_argument('--cpu', action='store_true', help='Force training on CPU')
    argv.add_argument('--accelerate', action='store_true', help='Accelerate deduce by changing the time scale')
    argv.add_argument('--log_data', action='store_true', help='Log experimental data')
    argv.add_argument('--run_id', type=str, default='test', help='Specifies the tag for saving the torch model')
    argv.add_argument('--record_id', type=str, default='test',
                      help='Specifies the tag for saving the experimental record')
    argv.add_argument('--continue_record', action='store_true', help='Continuing with the previous record')
    argv.add_argument('--episode_num', type=int, default=5, help='Number of episodes to be collected')

    options = argv.parse_args()
    config = options.config
    cpu = options.cpu
    accelerate = options.accelerate
    log_data = options.log_data
    run_id = options.run_id
    record_id = options.record_id
    continue_record = options.continue_record
    episode_num = options.episode_num
    print('cpu: %s' % cpu)
    print('accelerate: %s' % accelerate)
    print('log_data: %s' % log_data)
    print('run_id: %s' % run_id)
    print('record_id: %s' % record_id)
    print('continue_record: %s' % continue_record)
    print('episode_num: %s' % episode_num)
    return config, cpu, accelerate, log_data, run_id, record_id, continue_record, episode_num


def main():
    if not os.path.exists('experimental_record'):
        os.mkdir('experimental_record')
    if not os.path.exists('experimental_record/experimental_record.csv'):
        with open('experimental_record/experimental_record.csv', 'w') as f:
            f.write('run id,mean reward\n')

    yaml_parser = YamlParser('deduce.yaml')
    config = yaml_parser.get_config()
    config_file = config['config_file']
    run_id = config['run_id']
    record_id = config['record_id']
    log_data = config['log_data']
    accelerate = config['accelerate']
    episode_num = config['episode_num']
    cpu = config['cpu']
    continue_record = config['continue_record']

    # Parse the yaml config file. The result is a dictionary, which is passed to the deducer.
    config_file = YamlParser(config_file).get_config()
    print(config_file)

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # log experimental data
    if log_data:
        if not os.path.exists(r'experimental_record/%s' % (record_id,)):
            os.mkdir(r'experimental_record/%s' % (record_id,))
    if log_data and not continue_record:
        experimental_record_titles = ['steer_error', 'motor_error', 'action_delay_step']
        for title in experimental_record_titles:
            record_experimental_data_file = 'experimental_record/%s/%s.csv' % (record_id, title)
            with open(record_experimental_data_file, 'w') as f:
                f.write('%s,episode,reward\n' % title)
        record_experimental_data_file = 'experimental_record/%s/%s.csv' % (record_id, 'mix_deviation')
        with open(record_experimental_data_file, 'w') as f:
            f.write('steer_error,motor_error,action_delay_step,episode,reward\n')

    # steer_errors = [i / 100 for i in range(100, 500, 5)]
    steer_errors = [2]
    steer_errors = []
    # motor_errors = [i / 100 for i in range(100, 1000, 5)]
    motor_errors = [2]
    motor_errors = []
    # action_delay_steps = range(0, 40)
    action_delay_steps = [2]
    action_delay_steps = []
    mix_deviation_combinations = [
        # [1.25, 1.25, 1],
        # [1.077, 2.464, 0],
        # [0.81, 1.45, 0],
        # [0.83, 1.48, 0],
        # [1.0, 0.90, 0],
        # [1.0, 0.85, 0],
        # [1.0, 0.80, 0],
        [0.98, 0.68, 0],
        # [1, 1, 0],
    ]
    if run_id in dia_dict:
        print('使用预设的配置', dia_dict[run_id])
        mix_deviation_combinations = [dia_dict[run_id] + [0]]

    # construct experimental combinations
    deviation_combinations = []
    for steer_error in steer_errors:
        deviation_combinations.append((steer_error, 1, 0, 'steer_error'))
    for action_delay_step in action_delay_steps:
        deviation_combinations.append((1, 1, action_delay_step, 'action_delay_step'))
    for motor_error in motor_errors:
        deviation_combinations.append((1, motor_error, 0, 'motor_error'))
    for steer_error, motor_error, action_delay_step in mix_deviation_combinations:
        deviation_combinations.append((steer_error, motor_error, action_delay_step, 'mix_deviation'))
    if len(deviation_combinations) == 0:
        deviation_combinations.append((1, 1, 0, 'no_deviation'))

    for i, (steer_error, motor_error, action_delay_step, title) in enumerate(deviation_combinations):
        print('\n[%d/%d] steer_error: %.2f, motor_error: %.2f, action_delay_step: %d' % (i, len(deviation_combinations),
                                                                                         steer_error, motor_error,
                                                                                         action_delay_step))
        deducer = PPODeducer(config_file, run_id, device=device, accelerate=accelerate, steer_error=steer_error,
                             motor_error=motor_error, action_delay_step=action_delay_step)

        reward_mean, rewards, trojectoties = deducer.run_deduce(episode_num=episode_num)
        if log_data:
            with open('experimental_record/experimental_record.csv', 'a') as f:
                f.write('%s,%.2f\n' % (record_id, reward_mean))

            record_experimental_data_file = 'experimental_record/%s/%s.csv' % (record_id, title)
            if title in ['steer_error', 'motor_error', 'action_delay_step']:
                with open(record_experimental_data_file, 'a') as f:
                    for episode, reward_sum in rewards:
                        if title == 'steer_error':
                            s = '%.2f,%02d,%.2f\n' % (
                                steer_error, episode, reward_sum)
                        elif title == 'motor_error':
                            s = '%.2f,%02d,%.2f\n' % (
                                motor_error, episode, reward_sum)
                        elif title == 'action_delay_step':
                            s = '%d,%02d,%.2f\n' % (
                                action_delay_step, episode, reward_sum)
                        f.write(s)

            record_experimental_data_file = 'experimental_record/%s/%s.csv' % (record_id, 'mix_deviation')
            if title == 'mix_deviation':
                with open(record_experimental_data_file, 'a') as f:
                    for episode, reward_sum in rewards:
                        s = '%.2f,%.2f,%d,%02d,%.2f\n' % (
                            steer_error, motor_error, action_delay_step, episode, reward_sum)
                        f.write(s)
        deducer.close()


if __name__ == "__main__":
    try:
        main()
        # fmsgToQQ('2778433408', 'UGVRace-20', 'UGVRace-20测试完成')
    except:
        print(traceback.format_exc())
        # fmsgToQQ('2778433408', 'UGVRace-20', 'UGVRace-20报错')
