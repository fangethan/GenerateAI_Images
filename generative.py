from __future__ import annotations
from ai import predict_number, read_image


def flatten_image(image: list[list[int]]) -> list[int]:
    """
    Flattens a 2D list into a 1D list.

    :param image: 2D list of integers representing an image.
    :return: 1D list of integers representing a flattened image.
    """
    return [pixel for row in image for pixel in row]


def unflatten_image(flat_image: list[int]) -> list[list[int]]:
    """
    Unflattens a 1D list into a 2D list.

    :param flat_image: 1D list of integers representing a flattened image.
    :return: 2D list of integers.
    """

    # find the dimensions of the 2d list, since the 2d list always has same amount of rows and columns
    dimensions = int(len(flat_image) ** 0.5)
    # hold values of each row
    row = []
    # hold all the row arrays into one array
    unflatten_list = []

    # add row to unflatten_list
    # idx is incremented by dimensions, as that shows the limit of how many items can be in each row
    for idx in range(0, len(flat_image), dimensions):
        row = flat_image[idx : idx + dimensions]
        unflatten_list.append(row)

    return unflatten_list


def check_adjacent_for_one(flat_image: list[int], flat_pixel: int) -> bool:
    """
    Checks if a pixel has an adjacent pixel with the value of 1.

    :param flat_image: 1D list of integers representing a flattened image.
    :param flat_pixel: Integer representing the index of the pixel in question.
    :return: Boolean.
    """

    # make the image unflat so we can see the adjacent values
    image = unflatten_image(flat_image)
    DIMENSIONS = len(image[0])

    unflatten_row_index = int(flat_pixel / DIMENSIONS)
    unflatten_col_index = flat_pixel % DIMENSIONS
    is_adjacent = False

    # make sure whichever row and column index we are looking, it is valid by being less than the dimensions of unflatten image
    if unflatten_row_index < DIMENSIONS and unflatten_col_index < DIMENSIONS:
        # check if value is adjacent left, right or vertical
        if (
            check_adjacent_left(image, unflatten_row_index, unflatten_col_index)
            or check_adjacent_right(image, unflatten_row_index, unflatten_col_index)
            or check_adjacent_vertical(image, unflatten_row_index, unflatten_col_index)
        ):
            is_adjacent = True

    return is_adjacent


def check_adjacent_left(image, row_index, col_index):
    """
    Checks if a pixel has an adjacent pixel with the value of 1 to the left.

    :param image: 2D list of an unflattened image.
    :param row_index: Integer representing the row index of image.
    :param col_index: Integer representing the column index of image.
    :return: Boolean.
    """
    # make sure col_index isn't 0 (i.e., the leftmost column), since there is no pixel to the left of it
    if col_index != 0:
        if image[row_index][col_index] == 0 and image[row_index][col_index - 1] == 1:
            return True
    return False


def check_adjacent_right(image, row_index, col_index):
    """
    Checks if a pixel has an adjacent pixel with the value of 1 to the right.

    :param image: 2D list of an unflattened image.
    :param row_index: Integer representing the row index of image.
    :param col_index: Integer representing the column index of image.
    :return: Boolean.
    """
    # make sure col_index isn't the last index of the `row, since that would cause the index to be out of the array's range
    if col_index != len(image[0]) - 1:
        if image[row_index][col_index] == 0 and image[row_index][col_index + 1] == 1:
            return True
    return False


def check_adjacent_vertical(image, row_index, col_index):
    """
    Checks if a pixel has an adjacent pixel with the value of 1 that is above or below.

    :param image: 2D list of an unflattened image.
    :param row_index: Integer representing the row index of image.
    :param col_index: Integer representing the column index of image.
    :return: Boolean.
    """
    # when checking down, make sure we don't arrive to the last row
    if row_index < len(image) - 1:
        if image[row_index][col_index] == 0 and image[row_index + 1][col_index] == 1:
            return True
    # when checking up, make sure we don't arrive to the top row
    if row_index > 0:
        if image[row_index][col_index] == 0 and image[row_index - 1][col_index] == 1:
            return True

    return False


def pixel_flip(
    lst: list[int], orig_lst: list[int], budget: int, results: list, i: int = 0
) -> None:
    """
    Uses recursion to generate all possible combinations of flipped arrays where
    a pixel was a 0 and there was an adjacent pixel with the value of 1.

    :param lst: 1D list of integers representing a flattened image.
    :param orig_lst: 1D list of integers representing the original flattened image.
    :param budget: Integer representing the number of pixels that can be flipped.
    :param results: List of 1D lists of integers representing all possible combinations of flipped arrays, initially empty.
    :param i: Integer representing the index of the pixel in question.
    :return: None.
    """
    # copy() is used otherwise original list and changed list will be referring the same object
    # copy() is also used to ensure any modifications made inside the func, would stay in the func
    orignal_list = orig_lst.copy()
    changed_list = lst.copy()

    # the base case has two conditions that need to be met, or the recursion will be stopped
    # 1) checks if budget is not 0
    # 2) checks for the possible flips that can be done to the image
    while i < len(orignal_list) and budget > 0:
        # checks if the value is adjacent or not from the original_list
        if check_adjacent_for_one(orignal_list, i):
            # create a copy() again because we're trying to find all possibilities
            # keeping it as changed_list will limit the possibilities found, as it creates a reference to the same object in memory (any modifications made in new_changed_list would also modify changed_list)
            # this is our changed state (new_changed_list, results)
            new_changed_list = changed_list.copy()
            new_changed_list[i] = 1
            results.append(new_changed_list)
            # recursion is inside the while loop and not outside because we want to generate all possibilities
            # keeping it outside, meant we generate flipped arrays for the first pixel that met the condition
            pixel_flip(new_changed_list, orignal_list, budget - 1, results, i + 1)
        i += 1


def write_image(
    orig_image: list[list[int]], new_image: list[list[int]], file_name: str
) -> None:
    """
    Writes a newly generated image into a file where the modified pixels are marked as 'X'.

    :param orig_image: 2D list of integers representing the original image.
    :param new_image: 2D list of integers representing a newly generated image.
    :param file_name: String representing the name of the file.
    :return: None.
    """
    # new_row is responsible to hold the values of each row
    new_row = ""

    # use w, since we are writing
    with open(file_name, "w") as f:
        for row_index in range(len(orig_image)):
            for col_index in range(len(orig_image[0])):
                # check if we shoud add an X or the original value of the cell
                # The condition is if the original image and new image are identical or not when looking at the same position of the row and column for each image
                if orig_image[row_index][col_index] != new_image[row_index][col_index]:
                    new_row += "X"
                else:
                    new_row += str(orig_image[row_index][col_index])
                # check when the length of the new_row is the same as length of a row in an image such as original image
                # if it is, we want to add a new line, and write it, and reset row back to an empty string
                if len(new_row) == len(orig_image[0]):
                    f.write(new_row + "\n")
                    new_row = ""
        # a final write for the last new_row since the for loop will end before the last new_row can be written
        f.write(new_row)


def generate_new_images(image: list[list[int]], budget: int) -> list[list[list[int]]]:
    """
    Generates all possible new images that can be generated within the budget.

    :param image: 2D list of integers representing an image.
    :param budget: Integer representing the number of pixels that can be flipped.
    :return: List of 2D lists of integers representing all possible new images.
    """
    # grab all the flipped possibilities
    flipped_possibilities = []
    flat_image = flatten_image(image)
    pixel_flip(flat_image, flat_image, budget, flipped_possibilities)

    # new list as we need to return a 3D array
    all_new_possible_images_list = []

    # find the original image number from predict number
    original_image_number = predict_number(image)

    # form the flipped_possibilities into unflatten image (2D array) and add it to all_new_possible_images_list
    for possibility in flipped_possibilities:
        new_image = unflatten_image(possibility)
        # find the new image number from predict number
        new_image_number = predict_number(new_image)
        # compare if the image numbers are the same, as we only want to add possibilities with the same predicted number
        if new_image_number == original_image_number:
            all_new_possible_images_list.append(new_image)

    return all_new_possible_images_list


if __name__ == "__main__":
    image = read_image("image.txt")
    new_images = generate_new_images(image, 2)
    print(f"Number of new images generated: {len(new_images)}")
    # Write first image to test generation
    write_image(image, new_images[0], "new_image_1.txt")
