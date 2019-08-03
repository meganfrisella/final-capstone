from mynn.layers.dense import dense
from mynn.initializers.normal import normal
from mynn.activations.relu import relu


class Autoencoder:
    def __init__(self, d_data, d_embed):
        """
        Initializes the layers in the model.

        Parameters
        ----------
        d_desc : int
            dimension of the descriptor vector.

        d_embed : int
            dimension of the embedding
        """
        self.dense1 = dense(d_data, 1500, weight_initializer=normal)
        self.dense2 = dense(1500, 750, weight_initializer=normal)
        self.dense3 = dense(750, d_embed, weight_initializer=normal)

    def __call__(self, x):
        """
        Performs a forward pass on the model.

        Parameters
        ----------
        x : np.array, shape=(d_desc,)
            The descriptor vector

        Returns
        -------
        mg.Tensor, shape=(d_embed,)
            The embedding
        """
        activation = relu
        return self.dense3(activation(self.dense2(activation(self.dense1(x)))))

    @property
    def parameters(self):
        """
        Access learnable parameters

        Returns
        -------
        Tuple[mg.Tensor, ...]
            tuple containing of learnable parameters
        """
        return self.dense1.parameters + self.dense2.parameters + self.dense3.parameters

    def load_parameters(self, weight1, bias1, weight2, bias2, weight3, bias3):
        self.dense1 = lambda x: x @ weight1 + bias1
        self.dense2 = lambda x: x @ weight2 + bias2
        self.dense3 = lambda x: x @ weight3 + bias3
