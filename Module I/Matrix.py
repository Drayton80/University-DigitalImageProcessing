import re
import statistics

class Matrix:
    def get_matrix_from_file(self, file_path: str) -> list:
        matrix = []
        
        with open(file_path, 'r') as file:
            for line in file:
                row = []
                # Remove quaisquer letras:
                line = re.sub(r'[a-zA-Z\n]', '', line)
                # Remove os espaços, virgulas e traços iniciais e finais caso eles existam:
                #line = re.sub(r'^[\s\,\|]*(\w+)', re.match(r'^[\s\,\|]*(\w+)', line), line)
                #line = re.sub(r'(\w+)[\s\,\|]*$', re.match(r'(\w+)[\s\,\|]*$', line), line)
                # Substitui qualquer espaço, traço e virgula por apenas uma virgula:
                line = re.sub(r'[\s\,\|]+', ',', line)
                
                # Itera por cada elemento da linha do arquivo (separados agora por virgula):
                for element in line.split(','):
                    # Se houver o contrabarra é necessário fazer um tratamento para executar a divisão:
                    if "/" in element:
                        dividend = re.search(r'(\w+)\/\w+', element).group(1)
                        divider = re.search(r'\w+\/(\w+)', element).group(1)

                        row.append(float(dividend)/float(divider))
                    else:
                        row.append(float(element))
                
                matrix.append(row)
        
        return matrix

    def get_median_from_matrix(self, matrix: list):
        all_matrix_elements = []
        
        for row in matrix:
            for element in row:
                all_matrix_elements.append(element)

        return sorted(all_matrix_elements)[int(len(all_matrix_elements)/2)]

    def normalize_matrix(self, matrix: list):
        biggest_element = 0
        
        for row in matrix:
            for element in row:
                if element > biggest_element:
                    biggest_element = element
        
        normalized_matrix = []

        for row in matrix:
            normalized_row = []

            for element in row:
                normalized_row.append(element/biggest_element)

            normalized_matrix.append(normalized_row)
        
        return normalized_matrix

    
