from numpy import mean

#
# Perform a linear regression fit
#


def linear_regression(x, y):
    """Simple linear regression of two variables

    y = a + bx

    a, b = linear_regression(x, y)

    """

    if x.size != y.size:
        raise ValueError("x and y inputs must be the same size")

    mx = mean(x)
    my = mean(y)

    b = (mean(x * y) - mx * my) / (mean(x**2) - mx**2)
    a = my - b * mx

    return a, b
