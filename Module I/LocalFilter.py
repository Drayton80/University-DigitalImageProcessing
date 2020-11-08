from Matrix import Matrix

class LocalFilter:
    def zero_padding(self, image: list, mask_size: tuple) -> list:
        mask_number_rows = mask_size[0]
        mask_number_columns = mask_size[1]
        # Garante que a imagem original não seja alterada na operação de extensão por zeros:
        image_extended = image.copy()
        
        image_number_columns = len(image[0])
        # O Número de linhas adicionados pela extensão por zero é definido pela
        # quantidade de linhas e colunas da mascara, então, por exemplo, se ela
        # possuir três linhas, a imagem original apenas será extendida em 1 linha
        # (piso(3/2)=1), caso possua 5, será extendida em 2 linhas (piso(5/2)=2)
        for _ in range(int(mask_number_rows/2)):
            # Cria uma nova linha anterior a primeira preenchida com zeros
            image_extended.insert(0, [0] * image_number_columns)
            # Faz a mesma coisa, mas agora no final
            image_extended.append([0] * image_number_columns)
        # Após isso é preciso preencher em cada primeira e ultima coluna com zeros:
        for i in range(len(image_extended)):
            for _ in range(int(mask_number_columns/2)):
                # Insere na primeira posição de cada linha:
                image_extended[i].insert(0, 0)
                # Insere na ultima posição de cada linha:
                image_extended[i].append(0)

        return image_extended

    def get_matrix_of_all_local_matrices(self, image: list, mask_size: tuple) -> list:
        mask_number_rows = mask_size[0]
        mask_number_columns = mask_size[1]

        image_number_rows = len(image)
        image_number_columns = len(image[0])
        # Matriz que guardará cada matriz local em que será aplicado a máscara
        matrix_of_local_matrices = []

        for i in range(image_number_rows):
            if i + mask_number_rows > image_number_rows:
                break
            
            row_local_matrices = []
            
            for j in range(image_number_columns):
                if j + mask_number_columns > image_number_columns:
                    break
                
                local_matrix = []

                for local_i in range(mask_number_rows):
                    local_matrix_row = []

                    for local_j in range(mask_number_columns):
                        local_matrix_row.append(image[i+local_i][j+local_j])

                    local_matrix.append(local_matrix_row)
                
                row_local_matrices.append(local_matrix)

            matrix_of_local_matrices.append(row_local_matrices)

        return matrix_of_local_matrices
    
    def correlation(self, matrix1: list, matrix2: list) -> float:
        result = 0
        # Opera em cada linha
        for i, row in enumerate(matrix1):
            # Extrai cada elemento de cada coluna da linha:
            for j in range(len(row)):
                # Aplica o somatório para cada multiplicação de elementos entre as matrizes:
                result += matrix1[i][j] * matrix2[i][j]

        return result
    
    def apply_median_filter(self, image: list, mask_size=(3,3), zero_padding=True) -> list:
        # Se as dimensões da máscara forem maiores que a imagem o método para:
        if mask_size[0] > len(image) or mask_size[1] > len(image[0]):
            return

        if zero_padding:
            image_operated = self.zero_padding(image, (mask_size[0], mask_size[1]))
        else:
            image_operated = image.copy()

        filtered_image = []

        for row_of_local_matrices in self.get_matrix_of_all_local_matrices(image_operated, (mask_size[0], mask_size[1])):
            filtered_image_row = []
            for local_matrix in row_of_local_matrices:
                filtered_image_row.append(Matrix().get_median_from_matrix(local_matrix))

            filtered_image.append(filtered_image_row)
        
        return filtered_image


    def apply_mean_filter(self, image: list, mask_size=(3,3), zero_padding=True) -> list:
        mean_divider = 1 / (mask_size[0] * mask_size[1])
        mean_mask = []

        for _ in range(mask_size[0]):
            mean_mask.append([mean_divider for _ in range(mask_size[1])])

        return self.apply_generic_filter(image, mean_mask, zero_padding=zero_padding)

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

        




        

        

        

        

        
        
                

