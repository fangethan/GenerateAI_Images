from __future__ import annotations
import unittest
from generative import flatten_image, unflatten_image, check_adjacent_for_one, pixel_flip, write_image, generate_new_images
from ai import read_image


class TestGenerative(unittest.TestCase):
    """Unit tests for the module generative.py"""

    def test_flatten_image_5x5(self) -> None:
        """
        Verify flatten_image of a 5x5 table.
        """
        image = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], "Image was not flatten"
    
    def test_flatten_image_3x3(self) -> None:
        """
        Verify flatten_image of a 3x3 table.
        """
        image = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 1, 0, 1, 0, 0, 1, 0], "Image was not flatten"
    
    def test_flatten_image_4x4(self) -> None:
        """
        Verify flatten_image of a 4x4 table.
        """
        image = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        ]
        flat_image = flatten_image(image)
        assert flat_image == [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1], "Image was not flatten"

    def test_unflatten_image_5x5(self) -> None:
        """
        Verify output of unflatten_image for a 5x5 image.
        """
        flat_image = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ], "Image was not unflatten"

    def test_unflatten_image_3x3(self) -> None:
        """
        Verify output of unflatten_image for a 3x3 image.
        """
        flat_image = [1, 0, 1, 0, 1, 0, 0, 1, 0]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        ], "Image was not unflatten"

    def test_unflatten_image_4x4(self) -> None:
        """
        Verify output of unflatten_image for a 4x4 image.
        """
        flat_image = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
        unflat_image = unflatten_image(flat_image)
        assert unflat_image == [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        ], "Image was not unflatten"

    def test_check_adjacent_for_one_no_adjacent_middle(self) -> None:
        """
        Verify output of check_adjacent_for_one when there's no adjacent in the middle.
        """
        image = [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 2) == False, "It should be not adjacent"
    
    def test_check_adjacent_for_one_on_the_left(self) -> None:
        """
        Verify output of check_adjacent_for_one when there's an adjacent to the left.
        """
        image = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 1) == True, "It should be adjacent"

    def test_check_adjacent_for_one_on_the_right(self) -> None:
        """
        Verify output of check_adjacent_for_one when there's an adjacent to the right.
        """
        image = [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 1) == True, "It should be adjacent"
    
    def test_check_adjacent_for_one_on_the_up(self) -> None:
        """
        Verify output of check_adjacent_for_one when there's an adjacent its up.
        """
        image = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 4) == True, "It should be adjacent"
    
    def test_check_adjacent_for_one_on_the_down(self) -> None:
        """
        Verify output of check_adjacent_for_one when there's an adjacent its down.
        """
        image = [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 3) == True, "It should be adjacent"
    
    def test_check_adjacent_for_one_when_position_is_one(self) -> None:
        """
        Verify output of check_adjacent_for_one when the position lands on a 1.
        """
        image = [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 2) == False, "It should not be adjacent"
    
    def test_check_adjacent_for_one_when_position_is_on_corner(self) -> None:
        """
        Verify output of check_adjacent_for_one when the position lands on a 1.
        """
        image = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        ]
        flat_image = flatten_image(image)
        assert check_adjacent_for_one(flat_image, 8) == False, "It should not be adjacent"

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

        
    def test_generate_new_images_size(self) -> None:
        """
        Verify generate_new_images with image.txt and for each image of the generated images verify that image is of size 28x28,
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
    
    def test_generate_new_images_size_check_values(self) -> None:
        """
        Verify generate_new_images with image.txt and for each image of the generated images verify that all values in the generated image are either 1s or 0s,
        """
        image = read_image("image.txt")
        original_flat = flatten_image(image)
        new_images = generate_new_images(image, 2)

        # test all values in the generated image are either 1s or 0s
        for new_image in new_images:
            flat_image = flatten_image(new_image)
            assert all(elem in [0, 1] for elem in flat_image), "Not all numbers are 0 and 1"   
    
    def test_generate_new_images_size_within_budget(self) -> None:
        """
        Verify generate_new_images with image.txt and for each image of the generated images verify that the number of pixels flipped from original image are within budget,
        """
        image = read_image("image.txt")
        original_flat = flatten_image(image)
        new_images = generate_new_images(image, 2)

        # test the number of pixels flipped from original image are within budget
        num_flipped_zeros = 0
        for new_image in new_images:
            flat_image = flatten_image(new_image)
            for idx in range(len(flat_image)):
                if original_flat[idx] != flat_image[idx]:
                        num_flipped_zeros += 1
            assert num_flipped_zeros <= 2, "Numbers flipped from 0 to 1 is over budget"
            num_flipped_zeros = 0

    def test_generate_new_images_size_adjacent_to_one(self) -> None:
        """
        Verify generate_new_images with image.txt and for each image of the generated images verify that all pixels flipped from the original image had an adjacent value of 1.
        """
        image = read_image("image.txt")
        original_flat = flatten_image(image)
        new_images = generate_new_images(image, 2)
        # test all pixels flipped from the original image had an adjacent value of 1
        for new_image in new_images: 
            flat_image = flatten_image(new_image)
            for idx in range(len(original_flat)):
                if original_flat[idx] != flat_image[idx]:  
                    assert check_adjacent_for_one(original_flat, idx) == True, "Not all pixels flipped are from the original image that had an adjacent value of 1"   

if __name__ == "__main__":
    unittest.main()