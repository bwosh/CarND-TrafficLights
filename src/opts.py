import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    parser.add_argument("--lr", type=float, default=5e-04)

    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr_epochs", type=str, default="90,120") # TODO implement
    parser.add_argument("--lr_gamma", type=float, default=0.1) # TODO implement
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--output_size", type=float, default=64)

    args = parser.parse_args()
    return args