from ImageData import ImageData
from Matrix import Matrix

image_data = ImageData("images\\grito-edvard-munch.jpg")

mean_mask = Matrix().get_matrix_from_file("mask\\mean 5x5.txt")

print(mean_mask)