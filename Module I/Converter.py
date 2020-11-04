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

                r_channel_row.append(self._truncate_values_outside_limits(r))
                g_channel_row.append(self._truncate_values_outside_limits(g))
                b_channel_row.append(self._truncate_values_outside_limits(b))
                
            r_channel.append(r_channel_row)
            g_channel.append(g_channel_row)
            b_channel.append(b_channel_row)

        return r_channel, g_channel, b_channel