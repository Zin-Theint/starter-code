import torchvision.transforms as transforms

def get_train_transforms():
    """
    Return transform operations for training dataset.
    Includes random horizontal flip, random crop, normalization.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_test_transforms():
    """
    Return transform operations for test dataset.
    Only includes tensor conversion and normalization.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
