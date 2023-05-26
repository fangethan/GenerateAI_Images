from __future__ import annotations
import unittest
from generative import flatten_image, unflatten_image, check_adjacent_for_one, pixel_flip, write_image, generate_new_images
from ai import read_image


class TestGenerative(unittest.TestCase):
    """Unit tests for the module generative.py"""

    def test_flatten_image(self) -> None:
        """
        Verify output of flatten_image for at least three different sizes of images.
        """
        # Test 1
        image = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], "Image was not flatten"
        
        # Test 2
        image = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 1, 0, 1, 0, 0, 1, 0], "Image was not flatten"
    
        # Test 3
        image = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], "Image was not flatten"

    def test_unflatten_image(self) -> None:
        """
        Verify output of unflatten_image for at least three different sizes of flattened images.
        """
        # Test 1
        flat_image = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ], "Image was not unflatten"

        # Test 2
        flat_image = [1, 0, 1, 0, 1, 0, 0, 1, 0]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ], "Image was not unflatten"

        # Test 3
        flat_image = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        ], "Image was not unflatten"

    def test_check_adjacent_for_one(self) -> None:
        """
        Verify output of check_adjacent_for_one for three different pixel indexes of an image representing different scenarios.
        """
        # Test 1
        image = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 2) == False, "It should be not adjacent"
        
        # Test 2
        image = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 3) == True, "It should be adjacent"

        # Test 3
        image = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 2) == False, "It should be not adjacent"

        # Test 4
        image = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 8) == True, "It should be adjacent"

    def test_pixel_flip(self) -> None:
        """
        Verify output of pixel_flip for a 5x5 image with a budget of 2.
        """
        
        image = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
        ]
        
        flat_image = flatten_image(image)
        flipped_possibilities = []
        pixel_flip(flat_image, flat_image, 2, flipped_possibilities)
        
        # check flipped possibilities is greater than equal to 8
        assert len(flipped_possibilities) >= 8, "flipped_possibilities was less than 8"

        # all the expected flipped possibilities from image
        expected_flipped_possibilites = [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], 
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        ]

        # check if flipped_possibilities matches all the expected flipped possibilities
        for possibility in flipped_possibilities:
            assert possibility in expected_flipped_possibilites, "Possibility was not found in expected_flipped_possibilites"

        
    def test_generate_new_images(self) -> None:
        """
        Verify generate_new_images with image.txt and for each image of the generated images verify that:
        - image is of size 28x28,
        - all values in the generated image are either 1s or 0s,
        - the number of pixels flipped from original image are within budget,
        - all pixels flipped from the original image had an adjacent value of 1.
        """
        image = read_image("image.txt")
        original_flat = flatten_image(image)
        new_images = generate_new_images(image, 2)

        # check all images have 28 rows 
        for new_image in new_images:
            assert len(new_image) == 28, "Not all images have 28 rows"
        
        # check each row of all images, has 28 columns
        for new_image in new_images:
            for row in new_image:
                assert len(row) == 28, "Not all images have 28 columns"

        # test all values in the generated image are either 1s or 0s
        for new_image in new_images:
            flat_image = flatten_image(new_image)
            assert all(elem in [0, 1] for elem in flat_image), "Not all numbers are 0 and 1"   
        
        # test the number of pixels flipped from original image are within budget
        num_flipped_zeros = 0
        for new_image in new_images:
            flat_image = flatten_image(new_image)
            for idx in range(len(flat_image)):
                if original_flat[idx] != flat_image[idx]:
                        num_flipped_zeros += 1
            assert num_flipped_zeros <= 2, "Numbers flipped from 0 to 1 is over budget"
            num_flipped_zeros = 0

        # test all pixels flipped from the original image had an adjacent value of 1
        for new_image in new_images: 
            flat_image = flatten_image(new_image)
            for idx in range(len(original_flat)):
                if original_flat[idx] != flat_image[idx]:  
                    assert check_adjacent_for_one(original_flat, idx) == True, "Not all pixels flipped are from the original image that had an adjacent value of 1"   

if __name__ == "__main__":
    unittest.main()