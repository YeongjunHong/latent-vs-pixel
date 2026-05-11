import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class IntelligenceDataLoader:
    """
    다양한 데이터셋을 공통 인터페이스로 로드하는 Stage.
    """
    def __init__(self, dataset_name="cifar10", batch_size=64):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.transform = self._get_transforms()

    def _get_transforms(self):
        # 모델별(JEPA vs CNN)로 다를 수 있지만 일단 공통 적용
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_loader(self, train=True):
        if self.dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)
        elif self.dataset_name == "mnist":
            dataset = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)
        elif self.dataset_name == "stl10":
            # STL-10은 'train', 'test', 'unlabeled' 옵션이 있음
            split = 'train' if train else 'test'
            dataset = datasets.STL10(root='./data', split=split, download=True, transform=self.transform)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)