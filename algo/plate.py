


class Plate:

    def __init__(self, x,y,z):
        """
        конструктор плоскости по тому, в каких точках она пересекает оси
        """
        self.A = 1/x
        self.B = 1/y
        self.C = 1/z
        self.D = -1

    def is_out(self, x,y,z):
        """
        по какую сторону от плоскости находится точка
        """

        return self.A*x + self.B*y + self.C*z + self.D >=0












