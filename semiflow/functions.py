def add_backward(grad_output, a, b):
    # Local derivatives: df/da = 1, df/db = 1
    grad_a = grad_output * 1.0
    grad_b = grad_output * 1.0

    return [grad_a, grad_b]


def sub_backward(grad_output, a, b):
    # Local derivatives: df/da = 1, df/db = -1
    grad_a = grad_output * 1.0
    grad_b = grad_output * -1.0

    return [grad_a, grad_b]


def mul_backward(grad_output, a, b):
    # Local derivatives: df/da = b, df/db = a
    grad_a = grad_output * b
    grad_b = grad_output * a

    return [grad_a, grad_b]
