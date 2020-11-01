class Filter:
    def _zero_extension(self, image: list) -> list:
        # Garante que a imagem original não seja alterada na operação de extensão por zeros:
        image_extended = image.copy()
        
        image_number_columns = len(image[0])
        # Cria uma nova linha anterior a primeira preenchida com zeros
        image_extended.insert(0, [0] * image_number_columns)
        # Faz a mesma coisa, mas agora no final
        image_extended.append([0] * image_number_columns)
        # Após isso é preciso preencher em cada primeira e ultima coluna com zeros:
        for i in range(len(image_extended)):
            # Insere na primeira posição de cada linha:
            image_extended[i].insert(0, 0)
            # Insere na ultima posição de cada linha:
            image_extended[i].append(0)

        return image_extended
    
    def correlation(self, matrix1: list, matrix2: list) -> float:
        result = 0
        # Opera em cada linha
        for i, row in enumerate(matrix1):
            # Extrai cada elemento de cada coluna da linha:
            for j in range(len(row)):
                # Aplica o somatório para cada multiplicação de elementos entre as matrizes:
                result += matrix1[i][j] * matrix2[i][j]

        return result

    def aplly_generic_filter(self, image: list, mask: list, zero_extension=True):
        mask_number_rows = len(mask)
        mask_number_columns = len(mask[0])

        image_number_rows = len(image)
        image_number_columns = len(image[0])

        if mask_number_rows > image_number_rows or mask_number_columns > image_number_columns:
            return

        if zero_extension:
            image_operated = self._zero_extension(image)
            # Atualiza-se os valores de linhas e colunas:
            image_number_rows = len(image_operated)
            image_number_columns = len(image_operated[0])
        else:
            image_operated = image.copy()

        for row in image_operated:



        

        

        

        

        
        
                

