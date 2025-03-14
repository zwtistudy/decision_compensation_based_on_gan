import os
import traceback

import torch

from deducer_gym import PPODeducer
from yaml_parser import YamlParser


dia_dict = {
    'GymCarRacing_run_Ours': [0.195, 0.213, 1.224],
    'GymCarRacing_run_NNGAN': [1.10, 1.16],
    'GymCarRacing_run_Oracle': [0.2, 0.2, 1.2],
    'GymCarRacing_run_GAN': [1.08, 2.46],
    'GymCarRacing_run_DR': [1.00, 1.00],
    'GymCarRacing_run_BO': [0.96, 0.94],
}


def main():
    if not os.path.exists('experimental_record'):
        os.mkdir('experimental_record')
    if not os.path.exists('experimental_record/experimental_record.csv'):
        with open('experimental_record/experimental_record.csv', 'w') as f:
            f.write('run id,mean reward\n')

    yaml_parser = YamlParser('deduce_gym.yaml')
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

    mix_deviation_combinations = [
        [0.2, 0.2, 0.6, 0],
    ]

    # construct experimental combinations
    deviation_combinations = []
    for steer_error, motor_error, brake_error, action_delay_step in mix_deviation_combinations:
        deviation_combinations.append((steer_error, motor_error, brake_error, action_delay_step, 'mix_deviation'))
    if len(deviation_combinations) == 0:
        deviation_combinations.append((1, 1, 1, 0, 'no_deviation'))

    for i, (steer_error, motor_error, brake_error, action_delay_step, title) in enumerate(deviation_combinations):
        print('\n[%d/%d] steer_error: %.2f, motor_error: %.2f, brake_error:%.2f, action_delay_step: %d' % (i, len(deviation_combinations),
                                                                                         steer_error, motor_error, brake_error,
                                                                                         action_delay_step))
        deducer = PPODeducer(config_file, run_id, device=device, accelerate=accelerate, steer_error=steer_error,
                             motor_error=motor_error, brake_error=brake_error, action_delay_step=action_delay_step)

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
    except:
        print(traceback.format_exc())
