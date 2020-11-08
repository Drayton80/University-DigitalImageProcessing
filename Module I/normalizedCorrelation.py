import numpy as np

from scipy import signal
from sklearn.preprocessing import normalize


m1 = np.asmatrix(np.array([[0, 2, 1, 2],
                           [1, 4, 2, 0],
                           [3, 3, 0, 1]]))

m2 = np.asmatrix(np.array([[2, 1, 2],
                           [4, 2, 0],
                           [3, 0, 1]]))

#array([[  0.,   3.,   6.],
#   [  9.,  12.,  15.],
#   [ 18.,  21.,  24.]])

m1_normalized = normalize(m1, norm='l2')
m2_normalized = normalize(m2, norm='l2')

#[[ 0.          0.33333333  0.66666667]
#[ 0.25        0.33333333  0.41666667]
#[ 0.28571429  0.33333333  0.38095238]]


corr = signal.correlate2d(m1_normalized, m2_normalized, boundary='fill', mode='same')

print(corr)

biggest_correlation = np.where(corr == corr.max())

print(biggest_correlation)
 # Test:
image_data = ImageData("images\\" + image_name)
image_greyscale, _ , _ = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())
image_greyscale_normalized = normalize( np.asmatrix(image_greyscale), norm='l2')
print(image_greyscale_normalized.shape)
print(image_greyscale)

pattern_data = ImageData("images\\" + pattern_name)
pattern_greyscale, _ , _ = Converter().rgb_to_yiq(pattern_data.get_matrix_red(), pattern_data.get_matrix_green(), pattern_data.get_matrix_blue())
pattern_greyscale_normalized = normalize( np.asmatrix(pattern_greyscale), norm='l2')
print(pattern_greyscale_normalized.shape)

cross_correlation = signal.correlate2d(image_greyscale_normalized, pattern_greyscale_normalized, boundary='fill', mode='same')

print(cross_correlation.shape)

biggest_correlation_positions = np.where(cross_correlation == cross_correlation.max())

rows_positions, cols_positions = biggest_correlation_positions

plt.imshow(mpimg.imread("images\\" + image_name))
plt.scatter(x=cols_positions, y=rows_positions, c='g', s=6)
plt.show()
   
   
'''
max_possible_pixel_value = 255

    pattern_data  = ImageData("images\\" + pattern_name)
    pattern_red_normalized   = np.asmatrix(pattern_data.get_matrix_red())   / max_possible_pixel_value  
    pattern_green_normalized = np.asmatrix(pattern_data.get_matrix_green()) / max_possible_pixel_value  
    pattern_blue_normalized  = np.asmatrix(pattern_data.get_matrix_blue())  / max_possible_pixel_value  
    
    image_data  = ImageData("images\\" + image_name)
    image_red   = LocalFilter().zero_padding(image_data.get_matrix_red()  , (pattern_data.number_rows, pattern_data.number_columns))   
    image_green = LocalFilter().zero_padding(image_data.get_matrix_green(), (pattern_data.number_rows, pattern_data.number_columns))
    image_blue  = LocalFilter().zero_padding(image_data.get_matrix_blue() , (pattern_data.number_rows, pattern_data.number_columns))

    image_red_local_matrices   = LocalFilter().get_matrix_of_all_local_matrices(image_red  , (pattern_data.number_rows, pattern_data.number_columns))
    image_green_local_matrices = LocalFilter().get_matrix_of_all_local_matrices(image_green, (pattern_data.number_rows, pattern_data.number_columns))
    image_blue_local_matrices  = LocalFilter().get_matrix_of_all_local_matrices(image_blue , (pattern_data.number_rows, pattern_data.number_columns))
    
    mean_correlation_matrix = []
    # Iterando sobre todas as matrizes locais:
    with Bar('Processing Rows', max=len(image_red_local_matrices)) as bar:
        for row_index in range(len(image_red_local_matrices)):
            mean_correlation_matrix_row = []
            for col_index in range(len(image_red_local_matrices[row_index])):
                image_red_normalized   = np.asmatrix(image_red_local_matrices[row_index][col_index])   / max_possible_pixel_value
                image_green_normalized = np.asmatrix(image_green_local_matrices[row_index][col_index]) / max_possible_pixel_value
                image_blue_normalized  = np.asmatrix(image_blue_local_matrices[row_index][col_index])  / max_possible_pixel_value
                # Obtendo-se o módulo das matrizes de cada correlação entre os canais:
                red_correlation   = np.linalg.norm(signal.correlate2d(image_red_normalized  , pattern_red_normalized  , boundary='fill', mode='same'))
                green_correlation = np.linalg.norm(signal.correlate2d(image_green_normalized, pattern_green_normalized, boundary='fill', mode='same'))
                blue_correlation  = np.linalg.norm(signal.correlate2d(image_blue_normalized , pattern_blue_normalized , boundary='fill', mode='same'))
                # Tirando a média entre as correlações
                mean_correlation_matrix_row.append((red_correlation + green_correlation + blue_correlation) / 3)

            mean_correlation_matrix.append(mean_correlation_matrix_row)
            bar.next()

    mean_correlation_matrix = np.asmatrix(mean_correlation_matrix)

    biggest_correlation_positions = np.where(mean_correlation_matrix == mean_correlation_matrix.max())
    rows_positions, cols_positions = biggest_correlation_positions
'''

'''
image_data  = ImageData("images\\" + image_name)
    image_red   = np.asmatrix(image_data.get_matrix_red())  
    image_green = np.asmatrix(image_data.get_matrix_green())
    image_blue  = np.asmatrix(image_data.get_matrix_blue()) 

    pattern_data  = ImageData("images\\" + pattern_name)
    pattern_red   = np.asmatrix(pattern_data.get_matrix_red())    
    pattern_green = np.asmatrix(pattern_data.get_matrix_green())  
    pattern_blue  = np.asmatrix(pattern_data.get_matrix_blue())   
    
    biggest_values = []
    biggest_values.append(image_red.max())
    biggest_values.append(image_green.max())
    biggest_values.append(image_blue.max())
    biggest_values.append(pattern_red.max())
    biggest_values.append(pattern_green.max())
    biggest_values.append(pattern_blue.max())

    biggest_value_possible = max(biggest_values)

    image_red_normalized   = image_red   / biggest_value_possible
    image_green_normalized = image_green / biggest_value_possible
    image_blue_normalized  = image_blue  / biggest_value_possible

    pattern_red_normalized   = pattern_red   / biggest_value_possible
    pattern_green_normalized = pattern_green / biggest_value_possible
    pattern_blue_normalized  = pattern_blue  / biggest_value_possible
'''

'''
# Iterando sobre a transposta da matriz obtem-se linha a linha dela:
m1_columns = m1.T
m2_columns = m2.T

for col_index in len(m1_columns):
    m1_minus_u1 = np.subtract(m1, u1)
    # linalg.norm utiliza o módulo de Frobenius
    r1 = m1_minus_u1 / np.linalg.norm(m1_minus_u1)

    u2 = np.full(m1.shape, m1.mean())
    m2_minus_u2 = np.subtract(m2, u2)
    # linalg.norm utiliza o módulo de Frobenius
    r2 = m2_minus_u2 / np.linalg.norm(m2_minus_u2)
    # Inner Product entre r1 e r2
    r = np.inner(r1, r2)
u1 = np.full(m1.shape, m1.mean())
m1_minus_u1 = np.subtract(m1, u1)
# linalg.norm utiliza o módulo de Frobenius
r1 = m1_minus_u1 / np.linalg.norm(m1_minus_u1)

u2 = np.full(m1.shape, m1.mean())
m2_minus_u2 = np.subtract(m2, u2)
# linalg.norm utiliza o módulo de Frobenius
r2 = m2_minus_u2 / np.linalg.norm(m2_minus_u2)
# Inner Product entre r1 e r2
r = np.inner(r1, r2)'''
