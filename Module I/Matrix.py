import re

class Matrix:
    def get_matrix_from_file(self, file_path: str) -> list:
        matrix = []
        
        with open(file_path, 'r') as file:
            for line in file:
                row = []
                # Remove quaisquer letras:
                line = re.sub(r'[a-zA-Z\n]', '', line)
                # Remove os espaços, virgulas e traços iniciais e finais caso eles existam:
                line = re.sub(r'^[\s\,\|]+(\w+)', '$1', line)
                line = re.sub(r'(\w+)[\s\,\|]+$', '$1', line)
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