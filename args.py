import argparse

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--eps", type=float, default=3.0)
    return parser