import unittest
import torch
from models.lenet import LeNet
from scripts.data_loader import load_data



# if __name__ == "__main__":
#     root_path = "data/smaller_dataset"
#     train_data, test_data = load_data(root_path, (224,224), 32)
#
#     for data in train_data:
#         images, labels = data

class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        root_path = "../custom_data/data_set_1/smaller_dataset"
        train_data, test_data = load_data(root_path, (224, 224), 32)


class TestLeNetModel(unittest.TestCase):
    def test_output_shape(self):
        """Test whether the output shape of the LeNet model is correct."""
        in_channel = 3
        num_classes = 7
        net = LeNet(in_channel, num_classes)
        input_tensor = torch.rand(1, in_channel, 32, 32)
        output = net(input_tensor)
        self.assertEqual(output.shape, (1, num_classes))  # Validate the output size

if __name__ == '__main__':
    unittest.main()