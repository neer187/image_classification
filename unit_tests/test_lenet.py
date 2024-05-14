import unittest
import torch
from models.lenet import LeNet

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