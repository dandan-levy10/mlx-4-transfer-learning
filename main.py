from train import run_training

def main():
    # Call run_training with any desired arguments.
    run_training(num_epochs=1, batch_size=32, lr=0.001, num_layers=2, embedding_dim=128,
                 num_heads=4, ff_dim=256, step_size=5, gamma=0.5)

if __name__ == "__main__":
    main()
