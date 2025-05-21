import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer
import os

# Import necessary modules from the installed 'yolo' package
from yolo.tools.solver import TrainModel
from yolo.utils.logging_utils import setup

@hydra.main(config_path="../configs", config_name="train_carparts_bb.yaml", version_base=None)
def run_training(cfg: DictConfig):
    print("--- Effective Training Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------------")

    # Resolve dataset path if it's relative and needs to be absolute
    # Hydra's original_cwd can be useful if paths are relative to where the command was launched
    if not os.path.isabs(cfg.dataset.path):
        cfg.dataset.path = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.path)
        print(f"Resolved dataset.path to: {cfg.dataset.path}")

    callbacks, loggers, save_path = setup(cfg)
    
    print(f"Training run outputs (checkpoints, logs) will be saved to: {save_path}")
    # save_path is derived from hydra.run.dir, which we set in train_carparts_job.yaml

    trainer = Trainer(
        accelerator="auto",
        devices=1 if isinstance(cfg.device, (str, int)) and cfg.device != "cpu" else "auto",
        max_epochs=cfg.task.epoch,
        precision=cfg.task.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.task.get("log_every_n_steps", 1),
        gradient_clip_val=cfg.task.get("gradient_clip_val", 10.0),
        deterministic=cfg.task.get("deterministic", True),
        default_root_dir=save_path # Lightning saves checkpoints here
    )

    model = TrainModel(cfg) # Pass the composed DictConfig
    trainer.fit(model)

    print(f"--- Training complete for run: {cfg.name} ---")
    print(f"Checkpoints and logs are in {save_path}")

if __name__ == "__main__":
    run_training()
