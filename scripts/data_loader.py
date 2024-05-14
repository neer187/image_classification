import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_processing.data_loader_with_custom_dataset import CustomDataset

def load_data(data_path, image_size, batch_size = 16):
    train_dataset_path = data_path + "/" + "train"
    test_dataset_path = data_path + "/" + "test"

    transforms_train = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(image_size),
                                           transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=30),
                                           transforms.ToTensor()
                                           ])

    transforms_test = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(image_size),
                                          transforms.ToTensor()
                                          ])

    train_dataset = CustomDataset(train_dataset_path, transforms=transforms_train)
    test_dataset = CustomDataset(test_dataset_path, transforms=transforms_test)

    print("no of samples in train dataset", len(train_dataset))
    print("no of samples in test dataset", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


# if __name__ == "__main__":
#     root_path = "../custom_data/data_set_1/smaller_dataset"
#     train_data, test_data = load_data(root_path, (224,224), 32)
#
#     for data in train_data:
#         images, labels = data
