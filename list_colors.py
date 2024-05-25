import numpy as np

def get_color():

    list_color = []
    # food store
    lower_yellow = np.array([14,88,211])
    upper_yellow = np.array([19,255,249])

    # closing store
    lower_blue = np.array([107,77,231])
    upper_blue = np.array([109,229,245])

    # hotel
    lower_pink = np.array([168,57,231])
    upper_pink = np.array([171,224,249])

    yellow = (lower_yellow, upper_yellow)
    blue = (lower_blue, upper_blue)
    pink = (lower_pink, upper_pink)
    list_color.append(yellow)
    list_color.append(blue)
    list_color.append(pink)

    return list_color

if __name__ == "__main__":

    get_color()