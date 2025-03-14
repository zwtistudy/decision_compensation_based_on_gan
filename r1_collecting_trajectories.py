import os
import pickle
import traceback

import torch

from deducer import PPODeducer
from yaml_parser import YamlParser


def get_trajectory_save_path(model_id: str = None, trojectory_save_id: str = None):
    if not os.path.exists("trajectories"):
        os.mkdir("trajectories")
    if model_id is None:
        model_id = "test"
    trajectory_save_path = "trajectories/%s-%s" % (model_id, trojectory_save_id)
    return trajectory_save_path


def collect_trajectory(
    config_file,
    accelerate,
    model_id,
    collect_episode_num,
    trojectory_save_id,
    mix_deviation_combinations,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("device: %s" % device)

    deviation_combinations = []
    for steer_error, motor_error, action_delay_step in mix_deviation_combinations:
        deviation_combinations.append(
            (steer_error, motor_error, action_delay_step, "mix_deviation")
        )
    for i, (steer_error, motor_error, action_delay_step, title) in enumerate(
        deviation_combinations
    ):
        print(
            "\n[%d/%d] steer_error: %.2f, motor_error: %.2f, action_delay_step: %d"
            % (
                i,
                len(deviation_combinations),
                steer_error,
                motor_error,
                action_delay_step,
            )
        )
        config = YamlParser(config_file).get_config()
        deducer = PPODeducer(
            config,
            model_id,
            device=device,
            accelerate=accelerate,
            steer_error=steer_error,
            motor_error=motor_error,
            action_delay_step=action_delay_step,
        )
        reward_mean, rewards, trojectoties = deducer.run_deduce(
            episode_num=collect_episode_num
        )
        deducer.close()

    trojectory_save_path = get_trajectory_save_path(model_id, trojectory_save_id)
    with open("%s.pkl" % (trojectory_save_path,), "wb") as f:
        pickle.dump(trojectoties, f)


def main():
    yaml_parser = YamlParser('r1.yaml')
    config = yaml_parser.get_config()
    config_file = config['config_file']
    model_id = config['model_id']
    accelerate = config['accelerate']
    collect_episode_num = config['collect_episode_num']
    trojectory_save_id = config['trojectory_save_id']
    mix_deviation_combinations = [[1.25, 1.25, 1]]
    collect_trajectory(
        config_file,
        accelerate,
        model_id,
        collect_episode_num,
        trojectory_save_id,
        mix_deviation_combinations,
    )


if __name__ == "__main__":
    try:
        main()
        # fmsgToQQ('2778433408', 'UGVRace-20测试完成')
    except:
        print(traceback.format_exc())
        # fmsgToQQ('2778433408', 'UGVRace-20报错')
