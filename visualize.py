# visualize noisy images
import argparse
import torch
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage

rawdata_path='./benchmark/cifar100/data'

def get_dataset(dataset: str, split: str):
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return 
    elif dataset == "cifar100":
        if split =="train": return datasets.CIFAR100(rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))
        else: return datasets.CIFAR100(rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
DATASETS = ["imagenet", "cifar100"]
parser = argparse.ArgumentParser(description='visualize noisy images')
parser.add_argument("--dataset", type=str, choices=DATASETS)
parser.add_argument("--outdir", type=str, help="output directory")
parser.add_argument("--noise_sds",type=float)
parser.add_argument("--idx", type=int, nargs="+")
parser.add_argument("--split", choices=["train", "test"], default="test")
args = parser.parse_args()

toPilImage = ToPILImage()
dataset = get_dataset(args.dataset, args.split)
import os
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
for idx in args.idx:
    image, _ = dataset[idx]
    noise = torch.randn_like(image)
    noisy_image = torch.clamp(image + noise * args.noise_sds, min=0, max=1)
    pil = toPilImage(noisy_image)
    pil.save("{}/{}_{}.png".format(args.outdir, idx, int(args.noise_sds * 100)))
    toPilImage(image).save("{}/{}_{}_origin.png".format(args.outdir, idx, int(args.noise_sds * 100)))