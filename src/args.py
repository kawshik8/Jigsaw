# This file defines all the configuations of the program
import argparse

parser = argparse.ArgumentParser()

# General settings
# exp_name
parser.add_argument("--exp-name", type=str, default="debug", help="experiment name")
# load_args
parser.add_argument("--")
# device
parser.add_argument("--device", type=str, default="cuda", help="which device to run on")
# results_dir
parser.add_argument(
    "--results-dir",
    type=str,
    default="/scratch/hl3236/cv_results/",
    help="directory to save results and files",
)
# load_ckpt
parser.add_argument(
    "--load-ckpt",
    type=str,
    default="",
    help="load parameters from a checkpoint, choose auto to resume interrupted experiment",
)

# Data settings
# data_dir
parser.add_argument(
    "--data-dir", type=str, default="/scratch/hl3236/data/", help="directory of the data files",
)
# dataset
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "stl10", "mnist", "imagenet"]
)
# num_workers
parser.add_argument("--num_workers", type=int, default=4, help="number of workers in iterator")
# batch_size
parser.add_argument(
    "--batch-size", type=int, default=64, help="# images per minibatch",
)
# cache_pos
parser.add_argument(
    "--cache-pos", type=int, default=0, help="# cached positive images per image in minibatch",
)
# cache_neg
parser.add_argument(
    "--cache-neg", type=int, default=0, help="# cached negative images per image in minibatch"
)

# Model settings
# backbone
parser.add_argument("--backbone", type=str, default="resnet50", options=["resnet50"])
# TODO: some settings about init

# TODO: some settings about model extensions


# Training settings
# clip
parser.add_argument("--clip", type=float, default=0.5, help="gradient clip")
# learning_rate
parser.add_argument("--learning-rate", type=float, default=1e-3)
# weight_decay
parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
