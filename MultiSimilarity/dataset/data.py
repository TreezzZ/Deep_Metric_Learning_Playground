import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from .spc_sampler import SPC_Sampler


def get_dataloader(data_dir, img_per_class, batch_size, num_workers, args):
    # get dataset
    train_dataset = MetricLearningDataset(
        osp.join(data_dir, "train.txt"),
        get_transforms(args.arch, is_train=True),
    )
    test_dataset = MetricLearningDataset(
        osp.join(data_dir, "test.txt"),
        get_transforms(args.arch, is_train=False),
    )

    # get dataloader
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=SPC_Sampler(train_dataset.labels, batch_size, img_per_class)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


class MetricLearningDataset(Dataset):
    def __init__(self, txt_pth, transform=None):
        super(MetricLearningDataset, self).__init__()
        self.transform=transform
        
        # Load data from txt file
        img_dir = "/".join(txt_pth.split("/")[:-1])
        self.img_pths = []
        self.labels = []
        with open(txt_pth, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                self.img_pths.append(osp.join(img_dir, line[0]))
                self.labels.append(int(line[1]))

        # Creat index
        self.classes = list(set(self.labels))
        self.data_dict = {}
        for img, label in zip(self.img_pths, self.labels):
            if label not in self.data_dict.keys():
                self.data_dict[label] = []
            self.data_dict[label].append(img)


    def __getitem__(self, index):
        img = Image.open(self.img_pths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"data": img, "label": self.labels[index]}
    
    def __len__(self):
        return len(self.img_pths)


def get_transforms(arch="resnet50", is_train=True):
    transforms = []
    if is_train:
        transforms.extend([
            T.RandomResizedCrop(size=224),
            T.RandomHorizontalFlip(0.5),
        ])
    else:
        transforms.extend([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    transforms.append(T.ToTensor())
    if arch == "bninception":
        transforms.append(T.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039]))
    else:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

