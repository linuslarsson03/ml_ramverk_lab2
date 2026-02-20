from src.train import fit
from src.dataset import make_dataloaders
from src.model import MyModel
from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int
    learning_rate: float
    epochs: int
    validation_split: float
    exp_name: str

def run_test(cfg):
    print(f"starting experiment: {cfg.exp_name}")
    loaders = make_dataloaders(cfg)
    model = MyModel(num_classes=10)
    fit(model, loaders, cfg)

def main():
    experiments = [
        Config(batch_size=64, learning_rate=0.001, epochs=5, validation_split=0.2, exp_name="exp1"),
        Config(batch_size=64, learning_rate=0.0002, epochs=5, validation_split=0.2, exp_name="exp2"),
        Config(batch_size=128, learning_rate=0.003, epochs=5, validation_split=0.2, exp_name="exp3"),
    ]

    for cfg in experiments:
        run_test(cfg)

if __name__ == "__main__":
    main()