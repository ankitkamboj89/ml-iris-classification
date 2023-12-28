import unittest
import numpy as np
from ml_model import validate_data

class TestDataValidation(unittest.TestCase):
    def test_validate_data(self):
        # Correct case
        data = np.array([[1, 2], [3, 4]])
        target = np.array([0, 1])
        # Should not raise an exception
        validate_data(data, target)

        # Error case: empty data
        with self.assertRaises(ValueError):
            validate_data(np.array([]), target)

        # Error case: mismatched data and target
        with self.assertRaises(ValueError):
            validate_data(data, np.array([0]))

if __name__ == '__main__':
    unittest.main()
