import os
from trainer import PPOTrainer
from yaml_parser import YamlParser
from r1_collecting_trajectories import collect_trajectory
from r2_研究内容二核心代码 import train_GAN


def save_param_score(best_param, best_score, run_id, train_epoch):
    save_file_path = f"results/param_score_{run_id}.csv"
    if os.path.exists(save_file_path):
        with open(save_file_path, "a", encoding="utf-8") as f:
            f.write(f"{train_epoch},{best_param[0]},{best_param[1]},{best_score}\n")
    else:
        with open(save_file_path, "w", encoding="utf-8") as f:
            f.write("train_epoch,steer_error,motor_error,score\n")
            f.write(f"{train_epoch},{best_param[0]},{best_param[1]},{best_score}\n")


def main():
    # train
    config_file = 'configs/car1.yaml'
    config = YamlParser(config_file).get_config()
    run_id = 'IGAN_Docker_3'
    resume = False
    train_loop_num = 10
    motor_error = 1
    steer_error = 1

    # collect_trajectory
    accelerate = True
    collect_episode_num = 20
    trojectory_save_id = 125_125_1
    mix_deviation_combinations = [[3, 3, 1]]

    # GAN
    tuple_length = 3
    input_dim = 6 * (tuple_length + 1) + 2 * tuple_length
    num_epochs = 200
    batch_size = 64
    collect_sim_episode_num = 20
    discriminator_save_path = "pkl/discriminator.pkl"

    n_workers = config["n_workers"]
    for train_epoch in range(train_loop_num):
        config["n_workers"] = n_workers

        # train
        trainer = PPOTrainer(config, run_id=run_id, resume=resume, motor_error=motor_error, steer_error=steer_error)
        trainer.run_training()
        trainer.close()

        # collect_trajectory
        collect_trajectory(
            config,
            accelerate,
            run_id,
            collect_episode_num,
            trojectory_save_id,
            mix_deviation_combinations,
        )

        # GAN
        GAN_run_id = f"{run_id}-{train_epoch}"
        best_param, best_score = train_GAN(GAN_run_id, run_id, tuple_length, input_dim, num_epochs, batch_size,
                                           collect_sim_episode_num, discriminator_save_path)
        save_param_score(best_param, best_score, run_id, train_epoch)

        steer_error, motor_error = best_param


if '__main__' == __name__:
    main()
