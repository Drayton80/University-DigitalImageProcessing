from PIL import Image
import re

class ImageData:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.red = self.extract_red()
        self.green = self.extract_green()
        self.blue = self.extract_blue()

    def get_rgb(self):
        return self.red, self.green, self.blue
    
    def extract_red(self):
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 0 (vermelho)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(pixel[0], 0, 0) for pixel in self.image.getdata()]

    def extract_green(self):
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 1 (verde)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(0, pixel[1], 0) for pixel in self.image.getdata()]

    def extract_blue(self):
        # getdata() retorna uma lista de pixels os quais são listas [R, G, B]
        # Cria-se uma nova lista de pixels com apenas o canal de indice 2 (azul)
        # sem ser posto para zero, ou seja, elimina-se os outros dois canais
        return [(0, 0, pixel[2]) for pixel in self.image.getdata()]

    # Salva novas imagens para cada canal de cor (RGB):
    def save_images_per_channel(self):
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
