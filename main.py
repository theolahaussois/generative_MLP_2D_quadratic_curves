import argparse
from MLP_quadratique import main_inference

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict 2D quadratic curve images from parameters a, b, c"
    )
    parser.add_argument(
        "--coeffs", type=float, nargs=3, required=True,
        help="Three coefficients a b c for the quadratic curve"
    )
    parser.add_argument(
        "--out", type=str, default="predictions",
        help="Output folder for predicted images"
    )

    parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to model checkpoint (.pth file)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    a, b, c = args.coeffs
    main_inference(a, b, c, checkpoint=args.checkpoint, output_folder=args.out)
