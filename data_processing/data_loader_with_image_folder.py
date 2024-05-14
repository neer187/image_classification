from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dataset_path = "../custom_data/data_set_1/smaller_dataset/train"
test_dataset_path = "../custom_data/data_set_1/smaller_dataset/test"

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = datasets.ImageFolder(train_dataset_path, train_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for data in train_loader:
    images, labels = data
    print("Batch of image shape: ", images.shape)
    print("Labels: ", labels)
    break
