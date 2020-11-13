import re
import time
import argparse
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import normalize

from ImageData import ImageData
from Converter import Converter
from Matrix import Matrix
from PointFilter import PointFilter
from LocalFilter import LocalFilter


def show_image(image_path: str, plot: bool) -> None:
    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_path)
        _, axes = plt.subplots(1)
        axes.imshow(image)
        axes.set_axis_off()
        plt.show()

def show_image_with_dot_rectangle(image_path: str, plot: bool, center_position: tuple, rectangle_size: tuple, color='white', size=15):
    if not plot in ["False", "false", False]:
        image = mpimg.imread(image_path) 
        _, axes = plt.subplots(1)
        axes.imshow(image)
        dot_rectangle_plot(center_position, rectangle_size, color=color, size=size)
        axes.set_axis_off()
        plt.show()

def show_gray_map(matrix) -> None:
    _, axes = plt.subplots(1)
    plt.imshow(matrix, cmap='gray')
    axes.set_axis_off()
    plt.show()

def dot_rectangle_plot(center_position: tuple, rectangle_size: tuple, color='white', size=15):
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
    y4 = center_position[1] + half_height

    rectangle_xs = [x0, x1, x2, x3, x4, x0, x4, x0, x4, x0, x4, x0, x1, x2, x3, x4]
    rectangle_ys = [y0, y0, y0, y0, y0, y1, y1, y2, y2, y3, y3, y4, y4, y4, y4, y4]

    plt.scatter(x=rectangle_xs, y=rectangle_ys, c=color, s=size)


def functionality1(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())
    r, g, b = Converter().yiq_to_rgb(y, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(rgb-yiq-rgb)')
    show_image(new_image_path, plot)


def functionality2(image_name: str, plot):
    # PARTE 1 - Negativo em RGB Banda a Banda
    image_data = ImageData("images\\" + image_name)

    r_negative = PointFilter().apply_negative(image_data.get_matrix_red())
    g_negative = PointFilter().apply_negative(image_data.get_matrix_green())
    b_negative = PointFilter().apply_negative(image_data.get_matrix_blue())

    image_data.set_rgb_from_matrices(r_negative, g_negative, b_negative)
    new_image_path = image_data.save_image(new_file_name_suffix='(negative-rgb)')
    show_image(new_image_path, plot)

    # PARTE 2 - Negativo em Y
    image_data = ImageData("images\\" + image_name)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())

    y_negative = PointFilter().apply_negative(y)

    r, g, b = Converter().yiq_to_rgb(y_negative, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(negative-y)')
    show_image(new_image_path, plot)
    

def functionality3(image_name: str, plot):
    # PARTE 1 - Filtro Média
    image_data = ImageData("images\\" + image_name)
    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)
    image_filtered_mean_path = image_data.save_image(new_file_name_suffix='(media)')

    show_image(image_filtered_mean_path, plot)

    # PARTE 2 - Filtros de Sobel
    image_data = ImageData("images\\" + image_name)

    sobel_horizontal_mask = Matrix().get_matrix_from_file('mask\\sobel horizontal.txt')
    red_sobel_horizontal   = LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , sobel_horizontal_mask)
    green_sobel_horizontal = LocalFilter().apply_generic_filter(image_data.get_matrix_green(), sobel_horizontal_mask)
    blue_sobel_horizontal  = LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , sobel_horizontal_mask)

    image_data.set_rgb_from_matrices(red_sobel_horizontal, green_sobel_horizontal, blue_sobel_horizontal)
    image_filtered_sobel_horizontal_path = image_data.save_image(new_file_name_suffix='(sobel horizontal)')
    show_image(image_filtered_sobel_horizontal_path, plot)

    image_data = ImageData("images\\" + image_name)

    sobel_vertical_mask = Matrix().get_matrix_from_file('mask\\sobel vertical.txt')
    red_sobel_vertical   = LocalFilter().apply_generic_filter(image_data.get_matrix_red()  , sobel_vertical_mask)
    green_sobel_vertical = LocalFilter().apply_generic_filter(image_data.get_matrix_green(), sobel_vertical_mask)
    blue_sobel_vertical  = LocalFilter().apply_generic_filter(image_data.get_matrix_blue() , sobel_vertical_mask)

    image_data.set_rgb_from_matrices(red_sobel_vertical, green_sobel_vertical, blue_sobel_vertical)
    image_filtered_sobel_vertical_path = image_data.save_image(new_file_name_suffix='(sobel vertical)')
    show_image(image_filtered_sobel_vertical_path, plot)


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

    show_image(image_filtered_mean_path, plot)

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

    show_image(image_filtered_mean_path, plot)


def functionality5(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)
    
    red_median   = LocalFilter().apply_median_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_median = LocalFilter().apply_median_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_median  = LocalFilter().apply_median_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_median, green_median, blue_median)
    image_filtered_median_path = image_data.save_image(new_file_name_suffix='(mediana)')
    show_image(image_filtered_median_path, plot)


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
    # Itera até menos o pattern para não ultrapassar os limites da imagem com o local i e j:
    for i in tqdm(range(image_data.number_rows - pattern_data.number_rows)):
        mean_cross_correlation_row = []
        for j in range(image_data.number_columns - pattern_data.number_columns):
            red_local_matrix   = []
            green_local_matrix = []
            blue_local_matrix  = []
            # Geração da matriz local de cada canal:
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
            # Normalização das matrizes locais:
            image_red_local_normalized   = normalize(np.asmatrix(red_local_matrix)  , norm='l2')
            image_green_local_normalized = normalize(np.asmatrix(green_local_matrix), norm='l2')
            image_blue_local_normalized  = normalize(np.asmatrix(blue_local_matrix) , norm='l2')
            # Correlação:
            red_correlation   = signal.correlate2d(image_red_local_normalized  , pattern_red_normalized  , boundary='symm', mode='valid')[0][0]
            green_correlation = signal.correlate2d(image_green_local_normalized, pattern_green_normalized, boundary='symm', mode='valid')[0][0]
            blue_correlation  = signal.correlate2d(image_blue_local_normalized , pattern_blue_normalized , boundary='symm', mode='valid')[0][0]

            mean_cross_correlation_row.append((red_correlation + green_correlation + blue_correlation)/3)

        mean_cross_correlation.append(mean_cross_correlation_row)

    mean_cross_correlation = np.asmatrix(mean_cross_correlation)

    biggest_mean_correlation_positions = np.where(mean_cross_correlation == mean_cross_correlation.max())
    mean_row_center = biggest_mean_correlation_positions[0][0]
    mean_col_center = biggest_mean_correlation_positions[1][0]

    if not plot in ["False", "false", False]:
        # Correlação mapeada e exibida em tons de cinza:
        show_gray_map(mean_cross_correlation)
        # Exibição da imagem com a região de maior correlação destacada:
        show_image_with_dot_rectangle("images\\" + image_name, plot, (mean_col_center, mean_row_center), (pattern_data.number_columns, pattern_data.number_rows))
        

def functionality7(image_name, pattern_name, plot):
    pattern_data  = ImageData("images\\" + pattern_name)
    pattern_red   = pattern_data.get_matrix_red()   
    pattern_green = pattern_data.get_matrix_green()
    pattern_blue  = pattern_data.get_matrix_blue() 

    image_data  = ImageData("images\\" + image_name)
    image_red   = LocalFilter().zero_padding(image_data.get_matrix_red()  , (pattern_data.number_rows, pattern_data.number_columns))
    image_green = LocalFilter().zero_padding(image_data.get_matrix_green(), (pattern_data.number_rows, pattern_data.number_columns))
    image_blue  = LocalFilter().zero_padding(image_data.get_matrix_blue() , (pattern_data.number_rows, pattern_data.number_columns))

    mean_cross_correlation = []
    # Itera até menos o pattern para não ultrapassar os limites da imagem com o local i e j:
    for i in tqdm(range(image_data.number_rows - pattern_data.number_rows)):
        mean_cross_correlation_row = []
        for j in range(image_data.number_columns - pattern_data.number_columns):
            red_local_matrix   = []
            green_local_matrix = []
            blue_local_matrix  = []
            # Geração da matriz local de cada canal:
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
           
            red_correlation   = LocalFilter().correlation(red_local_matrix  , pattern_red  )
            green_correlation = LocalFilter().correlation(green_local_matrix, pattern_green)
            blue_correlation  = LocalFilter().correlation(blue_local_matrix , pattern_blue )

            mean_cross_correlation_row.append((red_correlation + green_correlation + blue_correlation)/3)

        mean_cross_correlation.append(mean_cross_correlation_row)

    mean_cross_correlation = np.asmatrix(mean_cross_correlation)

    biggest_mean_correlation_positions = np.where(mean_cross_correlation == mean_cross_correlation.max())
    mean_row_center = biggest_mean_correlation_positions[0][0]
    mean_col_center = biggest_mean_correlation_positions[1][0]

    if not plot in ["False", "false", False]:
        # Correlação mapeada e exibida em tons de cinza:
        show_gray_map(mean_cross_correlation)
        # Exibição da imagem com a região de maior correlação destacada:
        show_image_with_dot_rectangle("images\\" + image_name, plot, (mean_col_center, mean_row_center), (pattern_data.number_columns, pattern_data.number_rows))


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
