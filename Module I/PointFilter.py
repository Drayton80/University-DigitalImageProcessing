class PointFilter:
    def apply_negative(self, image: list, max_component_value=255) -> list:
        filtered_image = []
        
        for row in image:
            filtered_image_row = []

            for pixel_value in row:
                filtered_image_row.append(max_component_value - pixel_value)

            filtered_image.append(filtered_image_row)

        return filtered_image

