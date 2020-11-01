from ImageData import ImageData
from Matrix import Matrix
from Filter import Filter

image_data = ImageData("images\\triangulo-cores-maxwell.jpg")

#image_data.green = image_data.get_green_by_matrix(image_data.get_matrix_red())

image_data.save_images_per_channel()

mean_mask = Matrix().get_matrix_from_file("mask\\mean 5x5.txt")
