from semiflow.nn.parameter import Parameter


class Module:
    def __init__(self):
        self._submodules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        """Route __call__ to forward method (like PyTorch)"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Override this in subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")

    def parameters(self):
        """Recursively collect all parameters"""
        params = []

        # Get my own parameters
        for param in self._parameters.values():
            params.append(param)

        # Get parameters from submodules
        for submodule in self._submodules.values():
            params.extend(submodule.parameters())

        return params
