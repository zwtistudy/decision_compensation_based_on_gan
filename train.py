import torch
from trainer import PPOTrainer
from yaml_parser import YamlParser


def main():
    config_parser = YamlParser('train.yaml')
    config = config_parser.get_config()

    config_file = config['config_file']
    run_id = config['run_id']
    resume = config['resume']
    config_config = YamlParser(config_file).get_config()

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config_config, run_id=run_id, resume=resume)
    trainer.run_training()
    trainer.close()


if __name__ == "__main__":
    main()
