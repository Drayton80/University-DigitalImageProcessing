from PIL import Image
import re

class ImageData:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.red = self.extract_red()
        self.green = self.extract_green()
        self.blue = self.extract_blue()
        self.number_columns, self.number_rows = self.image.size
    
    def extract_red(self) -> list:
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 0 (vermelho)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(pixel[0], 0, 0) for pixel in self.image.getdata()]

    def extract_green(self) -> list:
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 1 (verde)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(0, pixel[1], 0) for pixel in self.image.getdata()]

    def extract_blue(self) -> list:
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 2 (azul)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(0, 0, pixel[2]) for pixel in self.image.getdata()]

    def get_rgb(self) -> (list, list, list):
        return self.red, self.green, self.blue

    def get_matrix_red(self) -> list:
        return self._get_matrix_color_channel(0)

    def get_matrix_green(self) -> list:
        return self._get_matrix_color_channel(1)

    def get_matrix_blue(self) -> list:
        return self._get_matrix_color_channel(2)
    
    def _get_matrix_color_channel(self, color_channel_index: int) -> list:
        matrix = []
        row = []
        # Como cada canal é basicamente uma lista de pixels sem divisão por colunas e linhas
        # é necessário um contador para checar se a linha chegou ao final
        columns_counter = 1

        for pixel in self.image.getdata():
            row.append(pixel[color_channel_index])
            # Caso o contador chegue no tamanho do numero de colunas 
            # da imagem isso significará que a linha acabou    
            if columns_counter == self.number_columns:
                # Adiciona na matriz a linha que estava sendo gerada
                matrix.append(row)
                # Reseta os valores de armazenamento e contagem:
                columns_counter = 1
                row = []
            else:
                columns_counter += 1
        
        return matrix

    def get_red_by_matrix(self, matrix: list) -> list:
        return self._get_channel_color_by_matrix(matrix, 'red')

    def get_green_by_matrix(self, matrix: list) -> list:
        return self._get_channel_color_by_matrix(matrix, 'green')
        
    def get_blue_by_matrix(self, matrix: list) -> list:
        return self._get_channel_color_by_matrix(matrix, 'blue')

    def _get_channel_color_by_matrix(self, matrix: list, color_channel: str) -> list:
        channel_color = []
        # Faz a formatação de cada canal para o estilo de lista de tuplas usado para
        # exibir a imagem (ou salvá-la) com o PIL
        for row in matrix:
            for color_intensity in row:
                if color_channel.lower() in ['red', 'r']:
                    channel_color.append((color_intensity, 0, 0))
                elif color_channel.lower() in ['green', 'g']:
                    channel_color.append((0, color_intensity, 0))
                elif color_channel.lower() in ['blue', 'b']:
                    channel_color.append((0, 0, color_intensity))

        return channel_color

    # Salva novas imagens para cada canal de cor (RGB):
    def save_images_per_channel(self) -> None:
        # Utiliza expressões regulares para obter apenas o nome da imagem entre
        # o caminho do diretório e o . que define o tipo do arquivo ([...]\nome.tipo):
        image_name = re.search(r'(\w*\\)*(.*)\.\w+', self.image_path).group(2)
        
        image_red = Image.new('RGB', self.image.size)
        image_red.putdata(self.red)
        image_red.save('images/' + image_name + '(R).png')

        image_green = Image.new('RGB', self.image.size)
        image_green.putdata(self.green)
        image_green.save('images/' + image_name + '(G).png')

        image_blue = Image.new('RGB', self.image.size)
        image_blue.putdata(self.blue)
        image_blue.save('images/' + image_name + '(B).png')
