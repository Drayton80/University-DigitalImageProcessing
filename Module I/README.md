# Módulo I do Trabalho Prático

## Introdução

O projeto aqui descrito consiste do primeiro módulo do trabalho prático relativo à disciplina de Introdução ao Processamento Digital de Imagens minstrada pelo professor Leonardo Vidal Batista e pertencente a grade curricular do curso de Ciência da Computação da Universidade Federal da Paraíba (UFPB).

Tal módulo consiste em aplicar de forma prática os conhecimentos teóricos ofertados na disciplina relativos a sistemas de cores e domínio do espaço. Sendo isso feito através de um programa que possui funcionalidades acerca de converção entre os sistemas de cores RGB e YIQ, aplicações de filtros pontuais e locais e utilização de dois tipos de correlações para tentar buscar por um padrão em determinada região de uma imagem.

## Materiais e Métodos
### Ferramentas Utilizadas
Para o desenvolvimento das funcionalidades em si foi utilizada a linguagem de programação Python em conjunto com as suas seguintes bibliotecas: 
- **PIL:** para a extração dos dados dos pixels de imagens
- **time:** para a medição do tempo requisitado na especificação do projeto
- **numpy:** para a vetorização e utilização de certas operações em matrizes
- **matplotlib:** para a exibição das imagens após aplicadas as funcionalidades
- **scipy:** para o uso da correlação cruzada em duas dimensões
- **sklearn:** para a normalização de dados necessária em uma das funcionalidades

Além dessas, foram utilizadas algumas bibliotecas à mais para maior facilidade no uso e desenvolvimento da aplicação, como re, argparse e tqdm.

### Conversão entre RGB e YIQ
Ao converter uma imagem do sistema de cores RGB para o YIQ é necessário aplicar três fórmulas que extraem a luminância das cores denotada por Y e as crominâncias da imagem que são armazenadas em I e Q.
``` 
Y = 0.299*R + 0.587*G + 0.114*B
I = 0.596*R – 0.274*G – 0.322*B
Q = 0.211*R – 0.523*G + 0.312*B
```

Já para a conversão contrária de YIQ e RGB é necessário aplicar outras três fórmulas para fazer a transcrição de volta da informação não-visual que o sistema YIQ representa para o RGB.

```
R = 1.000*Y + 0.956*I + 0.621*Q
G = 1.000*Y – 0.272*I – 0.647*Q
B = 1.000*Y – 1.106*I + 1.703*Q
```

Para o desenvolvimento dessa funcionalidade foi criada a classe `Converter` que possui um método para conversão de RGB para YIQ e outro para o procedimento reverso:
``` python
class Converter:
    def rgb_to_yiq(self, r_channel: list, g_channel: list, b_channel: list):
        y_channel = []
        i_channel = []
        q_channel = []

        for i in range(len(r_channel)):
            y_channel_row = []
            i_channel_row = []
            q_channel_row = []

            for j in range(len(r_channel[0])):
                y_value = round(0.299*r_channel[i][j] + 0.587*g_channel[i][j] + 0.114*b_channel[i][j])
                i_value = round(0.596*r_channel[i][j] - 0.274*g_channel[i][j] - 0.322*b_channel[i][j])
                q_value = round(0.211*r_channel[i][j] - 0.523*g_channel[i][j] + 0.312*b_channel[i][j])

                y_channel_row.append(y_value)
                i_channel_row.append(i_value)
                q_channel_row.append(q_value)
                
            y_channel.append(y_channel_row)
            i_channel.append(i_channel_row)
            q_channel.append(q_channel_row)

        return y_channel, i_channel, q_channel

    def _truncate_values_outside_limits(self, value, min_value=0, max_value=255):
        if value < min_value:
            return min_value
        elif max_value < value:
            return max_value
        else:
            return value

    def yiq_to_rgb(self, y_channel: list, i_channel: list, q_channel: list) -> (list, list, list):
        r_channel = []
        g_channel = []
        b_channel = []

        for i in range(len(y_channel)):
            r_channel_row = []
            g_channel_row = []
            b_channel_row = []
            
            for j in range(len(y_channel[0])):
                r = round(1.0*y_channel[i][j] + 0.956*i_channel[i][j] + 0.621*q_channel[i][j])
                g = round(1.0*y_channel[i][j] - 0.272*i_channel[i][j] - 0.647*q_channel[i][j])
                b = round(1.0*y_channel[i][j] - 1.106*i_channel[i][j] + 1.703*q_channel[i][j])

                r_channel_row.append(self._truncate_values_outside_limits(r, min_value=0, max_value=255))
                g_channel_row.append(self._truncate_values_outside_limits(g, min_value=0, max_value=255))
                b_channel_row.append(self._truncate_values_outside_limits(b, min_value=0, max_value=255))
                
            r_channel.append(r_channel_row)
            g_channel.append(g_channel_row)
            b_channel.append(b_channel_row)

        return r_channel, g_channel, b_channel
```
O uso de tais métodos é feito na `main` através de uma função específica de nome `functionality1` que as aplica da seguinte forma:
``` python 
def functionality1(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)

    y, i, q = Converter().rgb_to_yiq(image_data.get_matrix_red(), image_data.get_matrix_green(), image_data.get_matrix_blue())
    r, g, b = Converter().yiq_to_rgb(y, i, q)

    image_data.set_rgb_from_matrices(r, g, b)
    new_image_path = image_data.save_image(new_file_name_suffix='(rgb-yiq-rgb)')
    show_image(new_image_path, plot)
```

### Filtro Negativo
O filtro pontual negativo é uma operação unária que consiste em obter uma imagem com valores opostos aos da imagem original e isso é feito de diferentes formas dependendo do sistema de cores em que está sendo aplicado. No RGB, tal aplicação é feita subtraindo o valor máximo que um pixel pode possuir pelo valor de cada pixel em cada canal e atribui-se o resultado disso ao valor do pixel no canal. Já no YIQ é necessário apenas subtrair o valor máximo que um pixel pode possuir no canal Y pelo valor do pixel nesse canal e atribuir o resultado ao pixel.

Para a codificação desse filtro, foi desenvolvida a classe `PointFilter` que possui um método genérico para aplicar qualquer negativo em um canal do sistema fornecido:
``` python
class PointFilter:
    def apply_negative(self, channel: list, max_pixel_value=255) -> list:
        filtered_channel = []
        
        for row in channel:
            filtered_channel_row = []

            for pixel_value in row:
                filtered_channel_row.append(max_pixel_value - pixel_value)

            filtered_channel.append(filtered_channel_row)

        return filtered_channel
```

Os usos desse filtro são feitos na `main` através de uma função específica de nome `functionality2` que o aplica da seguinte forma:
``` python 
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
```

### Correlação
A correlação é uma operação entre duas matrizes de mesmo tamanho que consiste em somar a multiplicação de cada elemento de uma matriz com o elemento na mesma posição da outra. Essa operação pode ser usada para aplicar determinados filtros locais em uma imagem e, por causa disso, sua codificação foi feita internamente a classe `LocalFilter` através do método:
``` python
def correlation(self, matrix1: list, matrix2: list) -> float:
    result = 0
    # Opera em cada linha
    for i, row in enumerate(matrix1):
        # Extrai cada elemento de cada coluna da linha:
        for j in range(len(row)):
            # Aplica o somatório para cada multiplicação de elementos entre as matrizes:
            result += matrix1[i][j] * matrix2[i][j]

    return result
```

### Filtro de Sobel
O filtro de sobel é uma operação unária e local que tem como objetivo detectar contornos em uma imagem através de duas máscaras, uma para perceber os contornos horizontais e a outra para os verticais. A aplicação de tal filtro no sistema é feito internamente a classe `LocalFilter` com a utilização do método:
``` python
def apply_generic_filter(self, image: list, mask: list, zero_padding=True) -> list:
    # Se as dimensões da máscara forem maiores que a imagem o método para:
    if len(mask) > len(image) or len(mask[0]) > len(image[0]):
        return

    if zero_padding:
        image_operated = self.zero_padding(image, (len(mask), len(mask[0])))
    else:
        image_operated = image.copy()

    filtered_image = []

    for row_of_local_matrices in self.get_matrix_of_all_local_matrices(image_operated, (len(mask), len(mask[0]))):
        correlations_row = []

        for local_matrix in row_of_local_matrices:
            correlations_row.append(round(self.correlation(local_matrix, mask)))

        filtered_image.append(correlations_row)
    
    return filtered_image
```
Em conjunto com as máscaras horizontal e vertical de sobel que são extraídas de um arquivo txt. 

Sendo a máscara de detecção horizontal:
```
 1  2  1
 0  0  0
-1 -2 -1 
```
E a máscara de detecção vertical:
```
-1  0  1
-2  0  2
-1  0  1
```

Os usos desse filtro são feitos dentro da `main` na segunda parte da função `functionality3`, a qual tem como objetivo testar a correlação utilizando tanto esse filtro quanto o da média, que será descrita no próximo item.

``` python 
    [...]

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
```

### Filtro da Média
O filtro da média também é uma operação local, mas seu efeito consiste em borrar a imagem original dependendo do tamanho da máscara. A máscara da média é dada por uma matriz de elementos cujo valor é 1 sobre o total de elementos da matriz e quanto maior for esse total, mais borrada será a imagem resultante. Há duas formas de utilizar tal filtro pelo código, a primeira é utilizar o método `apply_generic_filter` descrito no item anterior em conjunto com a máscara da média extraida de um arquivo txt, já a segunda consiste en utilizar o método abaixo localizado na `LocalFilter` que automaticamente já produz a máscara baseado nas dimensões dela fornecidas como parâmetro:
``` python
def apply_mean_filter(self, image: list, mask_size=(3,3), zero_padding=True) -> list:
    mean_divider = 1 / (mask_size[0] * mask_size[1])
    mean_mask = []

    for _ in range(mask_size[0]):
        mean_mask.append([mean_divider for _ in range(mask_size[1])])

    return self.apply_generic_filter(image, mean_mask, zero_padding=zero_padding)
```
O uso de tal filtro é feito em duas ocasiões na `main`, a primeira na `functionality3` para testar a correlação:
``` python
def functionality3(image_name: str, plot):
    # PARTE 1 - Filtro Média
    image_data = ImageData("images\\" + image_name)
    red_mean   = LocalFilter().apply_mean_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_mean = LocalFilter().apply_mean_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_mean  = LocalFilter().apply_mean_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_mean, green_mean, blue_mean)
    image_filtered_mean_path = image_data.save_image(new_file_name_suffix='(media)')

    show_image(image_filtered_mean_path, plot)
    
    [...]
```
Já a segunda ocorre na `functionality4` para comparar a aplicação do filtro média 25x25 com a aplicação do 25x1 seguido pelo 1x25 em termos de tempo e resultado final:
``` python
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
```

### Filtro da Mediana
O filtro da mediana também é uma operação local, porém diferente das anteriores ele não possui uma máscara especifica em que se aplica uma correlação dela com as partes da imagem, sua máscara em si é apenas uma matriz a qual definirá o tamanho da parte da imagem que será analizado para obter o valor da mediana o qual vai ser atribuido a imagem resultante. Esse filtro serve tanto para obter um resultado visual que remete à uma pintura quanto para remover, em certa medida, o ruído conhecido com salt and pepper. Sua codificação é dada através de outro método presente na classe dos filtros locais:
``` python
def apply_median_filter(self, image: list, mask_size=(3,3), zero_padding=True) -> list:
    # Se as dimensões da máscara forem maiores que a imagem o método para:
    if mask_size[0] > len(image) or mask_size[1] > len(image[0]):
        return

    if zero_padding:
        image_operated = self.zero_padding(image, (mask_size[0], mask_size[1]))
    else:
        image_operated = image.copy()

    filtered_image = []

    for row_of_local_matrices in self.get_matrix_of_all_local_matrices(image_operated, (mask_size[0], mask_size[1]):
        filtered_image_row = []
        for local_matrix in row_of_local_matrices:
            filtered_image_row.append(Matrix().get_median_from_matrix(local_matrix))

        filtered_image.append(filtered_image_row)
    
    return filtered_image
```

Esse filtro é utilizado internamente na `main` através da função cujo nome é `functionality5`:

``` python 
def functionality5(image_name: str, plot):
    image_data = ImageData("images\\" + image_name)
    
    red_median   = LocalFilter().apply_median_filter(image_data.get_matrix_red()  , mask_size=(5,5))
    green_median = LocalFilter().apply_median_filter(image_data.get_matrix_green(), mask_size=(5,5))
    blue_median  = LocalFilter().apply_median_filter(image_data.get_matrix_blue() , mask_size=(5,5))
    
    image_data.set_rgb_from_matrices(red_median, green_median, blue_median)
    image_filtered_median_path = image_data.save_image(new_file_name_suffix='(mediana)')
    show_image(image_filtered_median_path, plot)
```

### Busca de um Padrão em uma Imagem
#### Correlação Cruzada Normalizada
Essa etapa consistiu em replicar o exemplo de uso da correlação cruzada normalizada fornecido para a função [normxcorr2 do MATLAB](https://la.mathworks.com/help/images/ref/normxcorr2.html?lang=en), mas em vez de replicar ela, foi necessário aplicar seu uso em cada banda da imagem RGB para, em seguida, obter a média simples do resultado da operação de correlação entre as três bandas de cores para cada pixel. Ou seja, o resultado dessa função seria uma matriz das médias entre três correlações cruzadas normalizadas (cada uma relativa a um canal do sistema RGB).

Para a implementação dessa funcionalidade utilizou-se do método `normalize` fornecido pela biblioteca sklearn para executar a normalização das bandas em conjunto com o método `correlate2d` da biblioteca scipy que executa uma correlação cruzada entre uma imagem e um padrão fornecido. Para um melhor desempenho devido ao alto custo das operações por pixel envolvidas, a extração das matrizes locais de cada vizinhança e as operações dos três canais foram todas executadas em um mesmo loop (distintamente das funcionalidades anteriores aonde a execução ocorria sequencialmente em fors separados). Tanto a operação quanto sua execução se encontram internamente na `main` através da função cujo nome é `functionality6`:

``` python
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
        show_gray_map(mean_cross_correlation, original_image_path="images\\" + image_name, save_plot_suffix="(cross-corr gray map)")
        # Exibição da imagem com a região de maior correlação destacada:
        show_image_with_dot_rectangle("images\\" + image_name, plot, (mean_col_center, mean_row_center), (pattern_data.number_columns, pattern_data.number_rows), save_plot_suffix="(cross-corr result)")
```

#### Correlação Simples
Por fim, a ultima funcionalidade tinha como requisito executar a mesma busca por padrão feita no item anterior, mas utilizando o método `correlation`, ou seja, uma correlação simples sem normalização. A implementação foi feita similarmente ao item anterior visando desempenho e seu código encontra-se na `main` internamente a função `functionality7`:

``` python
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
        show_gray_map(mean_cross_correlation, original_image_path="images\\" + image_name, save_plot_suffix="(corr gray map)")
        # Exibição da imagem com a região de maior correlação destacada:
        show_image_with_dot_rectangle("images\\" + image_name, plot, (mean_col_center, mean_row_center), (pattern_data.number_columns, pattern_data.number_rows), save_plot_suffix="(corr gray map)")
```

## Resultados e Discussão
Para a execução de testes foram fornecidas diversas imagens juntamente a especificação do trabalho. Para as funcionalidades de 1 a 5 foram utilizadas as seguintes imagens de teste: 

![CNN1.png](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/CNN1.png?raw=true)

![Detran_Minas-Gerais.jpg](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/Detran_Minas-Gerais.jpg?raw=true)

![2817540617.jpg](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/2817540617.jpg?raw=true)

Já para a funcionalidade 6 e 7 a especificação requisita o seguinte uso das imagens:

![baboon.png](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/baboon.png?raw=true)

![babooneye.png](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/babooneye.png?raw=true)

Sendo a primeira utilizada como base e a segunda como padrão de busca.

### Funcionalidade 1
A conversão de RGB para YIQ seguida da conversão contrária deve resultar na mesma imagem iniial e esse foi o primeiro teste a ser executado no sistema através do método `functionality1`, o qual resultou, como esperado, nas seguintes imagens de retorno:

![CNN1(rgb-yiq-rgb).png](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/CNN1(rgb-yiq-rgb).png?raw=true)

![Detran_Minas-Gerais(rgb-yiq-rgb).jpg](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/Detran_Minas-Gerais(rgb-yiq-rgb).jpg?raw=true)

![2817540617(rgb-yiq-rgb).jpg](https://github.com/Drayton80/University-DigitalImageProcessing/blob/main/Module%20I/images/2817540617.jpg?raw=true)

## Discussão

## Conclusão

 juntamente com um relatório, contendo pelo menos as seguintes
seções: introdução (contextualização e apresentação do tema, fundamentação
teórica, objetivos), materiais e métodos (descrição das atividades desenvolvidas e
das ferramentas e conhecimentos utilizados) resultados, discussão (problemas e
dificuldades encontradas, comentários críticos sobre os resultados) e conclusão. 