import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            # A.RandomBrightnessContrast(brightness_limit=32, contrast_limit=(0.5, 1.5)),
            # A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=(1, 2)),
            # A.CoarseDropout(20),
            A.Rotate(10),

            A.Resize(image_size, image_size),
            # A.RandomCrop(image_size, image_size, p=0.5),

            A.LongestMaxSize(image_size),
            A.Normalize(mean=mean, std=std),
            A.HorizontalFlip(),
            A.PadIfNeeded(image_size, image_size, 0),
            # A.Transpose(),
            ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.LongestMaxSize(image_size),
            A.Normalize(mean=mean, std=std),
            A.PadIfNeeded(image_size, image_size, 0),
            ToTensor(),
        ]
    )
