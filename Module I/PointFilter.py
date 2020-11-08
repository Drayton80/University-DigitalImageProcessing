class PointFilter:
    def apply_negative(self, channel: list, max_pixel_value=255) -> list:
        filtered_channel = []
        
        for row in channel:
            filtered_channel_row = []

            for pixel_value in row:
                filtered_channel_row.append(max_pixel_value - pixel_value)

            filtered_channel.append(filtered_channel_row)

        return filtered_channel

