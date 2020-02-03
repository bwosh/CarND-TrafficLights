import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--class_name", type=str, default="traffic light")
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--val", action="store_true")
    #parser.add_argument("--quantize", action="store_true")

    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--wh_weight", type=float, default=0.1)
    parser.add_argument("--hm_weight", type=float, default=1)

    parser.add_argument("--lr", type=float, default=5e-04)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_epochs", type=str, default="300,600")
    parser.add_argument("--lr_gammas", type=str, default="0.1,0.1")

    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--vis_thresh", type=float, default=0.2)

    args = parser.parse_args()
    return args