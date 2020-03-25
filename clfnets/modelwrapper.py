import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(ModelWrapper, self).__init__()

        if "verbose" in kwargs.keys():
            self.verbose = kwargs["verbose"]
        else:
            self.verbose = False
        self.__init__config(**kwargs)

    def __init__config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                print(f"{self} object already has '{k}' instance. Overriding {self.__dict__[k]} with {v}")

            setattr(self, k, v)

        if self.verbose:
            print(f"Resulting {self} attributes")
            for k, v in self.__dict__.items():
                print(f"...self.{k} = {v}")
            print("\n")

    def __build_layers(self):
        # Implement Layer-Building
        raise NotImplementedError

    def forward(self, *x):
        # Implement forward path by your self
        raise NotImplementedError
