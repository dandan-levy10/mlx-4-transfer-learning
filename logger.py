import os
import torch
import wandb
from typing import Optional, Dict, Any


class TrainingLogger:
    def __init__(self, use_wandb: bool = False, project_name: str = "image-captioning"):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        if self.use_wandb:
            wandb.config.update(config)

    def log_batch(
        self,
        loss: float,
        total_loss: float,
        batch_idx: int,
        epoch: int,
        split: str = "train",
    ) -> dict:
        """Log metrics for a single batch."""
        running_avg_loss = total_loss / batch_idx

        if self.use_wandb:
            wandb.log(
                {
                    f"{split}/step_loss": loss,
                    f"{split}/running_avg_loss": running_avg_loss,
                    "epoch": epoch,
                    "step": batch_idx,
                }
            )

        return {"avg_loss": f"{running_avg_loss:.4f}"}

    def log_epoch(self, train_loss: float, val_loss: float, epoch: int) -> None:
        """Log metrics for entire epoch."""
        if self.use_wandb:
            wandb.log(
                {
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "epoch": epoch,
                }
            )
        print(f"Epoch {epoch+1} average train loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1} average val loss: {val_loss:.4f}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        train_loss: float,
        val_loss: float,
        config: dict = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Save model checkpoint locally and to wandb."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if config is not None:
            checkpoint["config"] = config
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch}",
                type="model",
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Clean up logging."""
        if self.use_wandb:
            wandb.finish()