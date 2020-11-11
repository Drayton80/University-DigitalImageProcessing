import re
import time
import argparse
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from progress.bar import Bar
from scipy import signal
from sklearn.preprocessing import normalize

from ImageData import ImageData
from Converter import Converter
from Matrix import Matrix
from PointFilter import PointFilter
from LocalFilter import LocalFilter


def functionality1(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())
    r, g, b = Converter().yiq_to_rgb(y, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(rgb-yiq-rgb)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(new_image_path) 
        plt.imshow(image)
        plt.show()


def functionality2(image_name: str, plot):
    # PARTE 1 - Negativo em RGB Banda a Banda
    image_data = ImageData("images\\" + image_name)

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
    image_data = ImageData("images\\" + image_name)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())

    y_negative = PointFilter().apply_negative(y)

    r, g, b = Converter().yiq_to_rgb(y_negative, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(negative-y)')

    if not plot in ["False", "false", False]:
        image = mpimg.imread(new_image_path) 
        plt.imshow(image)
        plt.show()
    

def functionality3(image_name: str, plot):
    # PARTE 1 - Filtro Média
    image_data = ImageData("images\\" + image_name)
    
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


def functionality4(image_name: str, plot):
    # PARTE 1 - Aplicando Matriz 25x25
    image_data = ImageData("images\\" + image_name)

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

    # PARTE 2 - Aplicando Matriz 25x1 e depois 1x25
    image_data = ImageData("images\\" + image_name)
    
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


def functionality5(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)
    
    red_median   = LocalFilter().apply_median_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_median = LocalFilter().apply_median_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_median  = LocalFilter().apply_median_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_median, green_median, blue_median)
    image_filtered_median_path = image_data.save_image(new_file_name_suffix='(mediana)')
    
    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_filtered_median_path) 
        plt.imshow(image)
        plt.show()

def functionality6(image_name, pattern_name, plot):    
    pattern_data  = ImageData("images\\" + pattern_name)
    pattern_red_normalized   = normalize( np.asmatrix(pattern_data.get_matrix_red())   , norm='l2')  
    pattern_green_normalized = normalize( np.asmatrix(pattern_data.get_matrix_green()) , norm='l2')  
    pattern_blue_normalized  = normalize( np.asmatrix(pattern_data.get_matrix_blue())  , norm='l2')  

    image_data  = ImageData("images\\" + image_name)
    image_red   = LocalFilter().zero_padding(image_data.get_matrix_red()  , pattern_red_normalized.shape  )
    image_green = LocalFilter().zero_padding(image_data.get_matrix_green(), pattern_green_normalized.shape)
    image_blue  = LocalFilter().zero_padding(image_data.get_matrix_blue() , pattern_blue_normalized.shape )

    mean_cross_correlation = []
    for i in range(image_data.number_rows):
        if i + pattern_data.number_rows > image_data.number_rows:
            break
        mean_cross_correlation_row = []
        for j in range(image_data.number_columns):
            if j + pattern_data.number_columns > image_data.number_columns:
                break
            # Gera a matriz local de cada canal:
            red_local_matrix   = []
            green_local_matrix = []
            blue_local_matrix  = []
            for local_i in range(pattern_data.number_rows):
                red_local_matrix_row   = []
                green_local_matrix_row = []
                blue_local_matrix_row  = []

                for local_j in range(pattern_data.number_columns):
                    red_local_matrix_row.append(image_red[i+local_i][j+local_j])
                    green_local_matrix_row.append(image_green[i+local_i][j+local_j])
                    blue_local_matrix_row.append(image_blue[i+local_i][j+local_j])

                red_local_matrix.append(red_local_matrix_row)
                green_local_matrix.append(green_local_matrix_row)
                blue_local_matrix.append(blue_local_matrix_row)

            image_red_local_normalized   = normalize(np.asmatrix(red_local_matrix)  , norm='l2')
            image_green_local_normalized = normalize(np.asmatrix(green_local_matrix), norm='l2')
            image_blue_local_normalized  = normalize(np.asmatrix(blue_local_matrix) , norm='l2')

            red_correlation   = signal.correlate2d(image_red_local_normalized  , pattern_red_normalized  , boundary='symm', mode='valid')[0][0]
            green_correlation = signal.correlate2d(image_green_local_normalized, pattern_green_normalized, boundary='symm', mode='valid')[0][0]
            blue_correlation  = signal.correlate2d(image_blue_local_normalized , pattern_blue_normalized , boundary='symm', mode='valid')[0][0]

            mean_cross_correlation_row.append((red_correlation + green_correlation + blue_correlation)/3)
        mean_cross_correlation.append(mean_cross_correlation_row)
    mean_cross_correlation = np.asmatrix(mean_cross_correlation)

    biggest_mean_correlation_positions = np.where(mean_cross_correlation == mean_cross_correlation.max())
    mean_row_center = biggest_mean_correlation_positions[0][0]
    mean_col_center = biggest_mean_correlation_positions[1][0]

    plt.imshow(mean_cross_correlation, cmap='gray')
    plt.show()

    image = mpimg.imread("images\\" + image_name) 
    plt.imshow(image)
    dot_rectangle_plot((mean_col_center, mean_row_center)  , (pattern_data.number_columns, pattern_data.number_rows), color='white', size=25)
    plt.show()
        

def functionality7(image_name, pattern_name, plot):
    image_data = ImageData("images\\" + image_name)
    pattern_data = ImageData("images\\" + pattern_name)

    pattern_red_channel = pattern_data.get_matrix_red()
    pattern_green_channel = pattern_data.get_matrix_green()
    pattern_blue_channel = pattern_data.get_matrix_blue()

    red_correlation   = np.asmatrix(LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , pattern_red_channel  ))
    green_correlation = np.asmatrix(LocalFilter().apply_generic_filter(image_data.get_matrix_green(), pattern_green_channel))
    blue_correlation  = np.asmatrix(LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , pattern_blue_channel ))

    mean_correlation_matrix = (red_correlation + green_correlation + blue_correlation) / 3
    biggest_correlation_positions = np.where(mean_correlation_matrix == mean_correlation_matrix.max())

    rows_positions, cols_positions = biggest_correlation_positions

    if not plot in ["False", "false", False]:
        plt.imshow(mpimg.imread("images\\" + image_name))
        plt.scatter(x=cols_positions, y=rows_positions, c='g', s=40)
        plt.show()

def dot_rectangle_plot(center_position: tuple, rectangle_size: tuple, color='black', size=5):
    half_width = int(rectangle_size[0] / 2)
    half_height = int(rectangle_size[1] / 2)

    x0 = center_position[0] - half_width
    x1 = center_position[0] - int(half_width/2)
    x2 = center_position[0]
    x3 = center_position[0] + int(half_width/2)
    x4 = center_position[0] + half_width

    y0 = center_position[1] - half_height
    y1 = center_position[1] - int(half_height/2)
    y2 = center_position[1]
    y3 = center_position[1] + int(half_height/2)
    y4 = center_position[1] + half_width

    rectangle_xs = [x0, x1, x2, x3, x4, x0, x4, x0, x4, x0, x4, x0, x1, x2, x3, x4]
    rectangle_ys = [y0, y0, y0, y0, y0, y1, y1, y2, y2, y3, y3, y4, y4, y4, y4, y4]

    plt.scatter(x=rectangle_xs, y=rectangle_ys, c=color, s=size)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--funcionalidade", required=True, help="quais funcionalidades serão exibidas")
    ap.add_argument("-i", "--imagem", required=False, help="qual imagem será operada", nargs='?', const="baboon.png")
    ap.add_argument("-pt", "--pattern", required=False, help="qual pattern que será pesquisado na imagem", nargs='?', const="babooneye.png")
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
        functionality6(args["imagem"], args["pattern"], args["plot"])
    elif args["funcionalidade"] == "7":
        functionality7(args["imagem"], args["pattern"], args["plot"])
