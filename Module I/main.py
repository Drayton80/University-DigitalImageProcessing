import re
import time
import argparse
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.preprocessing import normalize

from ImageData import ImageData
from Converter import Converter
from Matrix import Matrix
from PointFilter import PointFilter
from LocalFilter import LocalFilter


def functionality1(image: str, plot):
    image_data = ImageData("images\\" + image)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())
    r, g, b = Converter().yiq_to_rgb(y, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(rgb-yiq-rgb)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(new_image_path) 
        plt.imshow(image)
        plt.show()


def functionality2(image: str, plot):
    # PARTE 1 - Negativo em RGB Banda a Banda
    image_data = ImageData("images\\" + image)

    r_negative = PointFilter().apply_negative(image_data.get_matrix_red())
    g_negative = PointFilter().apply_negative(image_data.get_matrix_green())
    b_negative = PointFilter().apply_negative(image_data.get_matrix_blue())

    image_data.set_rgb_from_matrices(r_negative, g_negative, b_negative)
    new_image_path = image_data.save_image(new_file_name_suffix='(negative-rgb)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(new_image_path) 
        plt.imshow(image)
        plt.show()

    # PARTE 2 - Negativo em Y
    image_data = ImageData("images\\" + image)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())

    max_value_y = 0.299*255 + 0.587*255 + 0.114*255
    y_negative = PointFilter().apply_negative(y, max_component_value=max_value_y)

    r, g, b = Converter().yiq_to_rgb(y_negative, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(negative-y)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(new_image_path) 
        plt.imshow(image)
        plt.show()
    

def functionality3(image: str, plot):
    # PARTE 1 - Filtro Média
    image_data = ImageData("images\\" + image)
    
    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)
    image_filtered_mean_path = image_data.save_image(new_file_name_suffix='(media)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_mean_path) 
        plt.imshow(image)
        plt.show()

    # PARTE 2 - Filtro Sobel
    sobel_horizontal_mask = Matrix().get_matrix_from_file('mask\\sobel horizontal.txt')
    red_sobel_horizontal   = LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , sobel_horizontal_mask)
    green_sobel_horizontal = LocalFilter().apply_generic_filter(image_data.get_matrix_green(), sobel_horizontal_mask)
    blue_sobel_horizontal  = LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , sobel_horizontal_mask)

    image_data.set_rgb_from_matrices(red_sobel_horizontal, green_sobel_horizontal, blue_sobel_horizontal)
    image_filtered_sobel_horizontal_path = image_data.save_image(new_file_name_suffix='(sobel horizontal)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_sobel_horizontal_path) 
        plt.imshow(image)
        plt.show()

    sobel_vertical_mask = Matrix().get_matrix_from_file('mask\\sobel vertical.txt')
    red_sobel_vertical   = LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , sobel_vertical_mask)
    green_sobel_vertical = LocalFilter().apply_generic_filter(image_data.get_matrix_green(), sobel_vertical_mask)
    blue_sobel_vertical  = LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , sobel_vertical_mask)

    image_data.set_rgb_from_matrices(red_sobel_vertical, green_sobel_vertical, blue_sobel_vertical)
    image_filtered_sobel_vertical_path = image_data.save_image(new_file_name_suffix='(sobel vertical)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_sobel_vertical_path) 
        plt.imshow(image)
        plt.show()


def functionality4(image: str, plot):
    # PARTE 1 - Aplicando Matriz 25x25
    image_data = ImageData("images\\" + image)

    start = time.time()
    
    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(25,25))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(25,25))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(25,25))
    
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)
    image_filtered_mean_path = image_data.save_image(new_file_name_suffix='(media 25x25)')

    end = time.time()
    print(end - start)

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_mean_path) 
        plt.imshow(image)
        plt.show()

    image_data = ImageData("images\\" + image)

    # PARTE 2 - Aplicando Matriz 25x1 e depois 1x25
    start = time.time()
    
    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(25,1))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(25,1))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(25,1))
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)

    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(1,25))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(1,25))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(1,25))
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)

    image_filtered_mean_path = image_data.save_image(new_file_name_suffix='(media 25x1 e 1x25)')

    end = time.time()
    print(end - start)

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_mean_path) 
        plt.imshow(image)
        plt.show()


def functionality5(image: str, plot):
    image_data = ImageData("images\\" + image)
    
    red_median   = LocalFilter().apply_median_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_median = LocalFilter().apply_median_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_median  = LocalFilter().apply_median_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_median, green_median, blue_median)
    image_filtered_median_path = image_data.save_image(new_file_name_suffix='(mediana)')
    
    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_median_path) 
        plt.imshow(image)
        plt.show()

def functionality6(plot):
    image_data = ImageData("images\\grito-edvard-munch.jpg")
    image_red_normalized   = normalize( np.asmatrix(np.asarray(image_data.get_matrix_red()))   , norm='l1')
    image_green_normalized = normalize( np.asmatrix(np.asarray(image_data.get_matrix_green())) , norm='l1')
    image_blue_normalized  = normalize( np.asmatrix(np.asarray(image_data.get_matrix_blue()))  , norm='l1')

    pattern_data = ImageData("images\\grito-edvard-munch-cabeca.jpg")
    pattern_red_normalized   = normalize( np.asmatrix(np.asarray(pattern_data.get_matrix_red()))   , norm='l1')
    pattern_green_normalized = normalize( np.asmatrix(np.asarray(pattern_data.get_matrix_green())) , norm='l1')
    pattern_blue_normalized  = normalize( np.asmatrix(np.asarray(pattern_data.get_matrix_blue()))  , norm='l1')

    red_correlation   = signal.correlate2d(image_red_normalized  , pattern_red_normalized  , boundary='fill', mode='same')
    green_correlation = signal.correlate2d(image_green_normalized, pattern_green_normalized, boundary='fill', mode='same')
    blue_correlation  = signal.correlate2d(image_blue_normalized , pattern_blue_normalized , boundary='fill', mode='same')

    mean_correlation = (red_correlation + green_correlation + blue_correlation) / 3

    biggest_correlation_positions = np.where(mean_correlation == mean_correlation.max())
    rows_positions, cols_positions = biggest_correlation_positions

    image = mpimg.imread("images\\baboon.png") 
    plt.imshow(mean_correlation, cmap='gray')
    plt.scatter(x=cols_positions, y=rows_positions, c='g', s=40)
    plt.show()
    #y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

def functionality7(plot):
    image_data = ImageData("images\\baboon.png")
    pattern_data = ImageData("images\\babooneye.png")

    pattern_red_channel = pattern_data.get_matrix_red()
    pattern_green_channel = pattern_data.get_matrix_green()
    pattern_blue_channel = pattern_data.get_matrix_blue()

    red_correlation   = LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , pattern_red_channel  )
    green_correlation = LocalFilter().apply_generic_filter(image_data.get_matrix_green(), pattern_green_channel)
    blue_correlation  = LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , pattern_blue_channel )

    image_data.set_rgb_from_matrices(red_correlation, green_correlation, blue_correlation)
    image_filtered_path = image_data.save_image(new_file_name_suffix='(correlação)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_path) 
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--funcionalidade", required=True, help="quais funcionalidades serão exibidas")
    ap.add_argument("-i", "--imagem", required=False, help="qual imagem será operada", nargs='?', const="Detran_Minas-Gerais.jpg")
    ap.add_argument("-pl", "--plot", required=False, help="se a imagem será plotada", nargs='?', const=True)

    args = vars(ap.parse_args())

    # Inicializa o servido da aplicação:
    if args["funcionalidade"] == "all":
        pass
    elif args["funcionalidade"] == "1":
        functionality1(args["imagem"], args["plot"])
    elif args["funcionalidade"] == "2":
        functionality2(args["imagem"], args["plot"])
    elif args["funcionalidade"] == "3":
        functionality3(args["imagem"], args["plot"])
    elif args["funcionalidade"] == "4":
        functionality4(args["imagem"], args["plot"])
    elif args["funcionalidade"] == "5":
        functionality5(args["imagem"], args["plot"])
    elif args["funcionalidade"] == "6":
        functionality6(args["plot"])
    elif args["funcionalidade"] == "7":
        functionality7(args["plot"])
