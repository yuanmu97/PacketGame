from mindspore import nn, ops


class Conv1dTwoview(nn.Cell):
    """
    build two-view neural network 

    Args:
        inp_len1, inp_len2 (int): input length of two views, 5 by default
        conv_units (list of int): number of conv1d units
            by default, two conv1d layers with 32 units
    """
    def __init__(self, inp_len1=5, inp_len2=5, conv_units=[32, 32]):
        super(Conv1dTwoview, self).__init__()
        
        self.view1_layers = nn.CellList([nn.Conv1d(inp_len1, conv_units[0], 1, has_bias=True, pad_mode='valid'),
                                         nn.ReLU()])
        self.view2_layers = nn.CellList([nn.Conv1d(inp_len2, conv_units[0], 1, has_bias=True, pad_mode='valid'),
                                         nn.ReLU()])
        
        for i in range(len(conv_units)-1):
            self.view1_layers.append(nn.Conv1d(conv_units[i], conv_units[i+1], 1, has_bias=True, pad_mode='valid'))
            self.view1_layers.append(nn.ReLU())
            
            self.view2_layers.append(nn.Conv1d(conv_units[i], conv_units[i+1], 1, has_bias=True, pad_mode='valid'))
            self.view2_layers.append(nn.ReLU())
        
        self.view1_layers.append(nn.AdaptiveMaxPool1d(1))
        self.view2_layers.append(nn.AdaptiveMaxPool1d(1))
        
        self.dense = nn.Dense(conv_units[-1]*2, 1, has_bias=True)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x1, x2):
        for v1_layer, v2_layer in zip(self.view1_layers, self.view2_layers):
            x1 = v1_layer(x1)
            x2 = v2_layer(x2)
        x = ops.cat((x1, x2), axis=1)
        x = ops.squeeze(x, axis=-1)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
    

class EnsembleThreeview(nn.Cell):
    """
    build three-view neural network 

    Args:
        inp_len1, inp_len2, inp_len3 (int): input length of three views, 5 for inp1 & inp2 and 1 for inp3 by default
        conv_units (list of int): number of conv1d units
            by default, two conv1d layers with 32 units
        dense_unit (int): number of dense units, 128 by default
    """
    def __init__(self, inp_len1=5, inp_len2=5, inp_len3=1, conv_units=[32, 32], dense_unit=128):
        super(EnsembleThreeview, self).__init__()
        
        self.view1_layers = nn.CellList([nn.Conv1d(inp_len1, conv_units[0], 1, has_bias=True, pad_mode='valid'),
                                         nn.ReLU()])
        self.view2_layers = nn.CellList([nn.Conv1d(inp_len2, conv_units[0], 1, has_bias=True, pad_mode='valid'),
                                         nn.ReLU()])
        
        for i in range(len(conv_units)-1):
            self.view1_layers.append(nn.Conv1d(conv_units[i], conv_units[i+1], 1, has_bias=True, pad_mode='valid'))
            self.view1_layers.append(nn.ReLU())
            
            self.view2_layers.append(nn.Conv1d(conv_units[i], conv_units[i+1], 1, has_bias=True, pad_mode='valid'))
            self.view2_layers.append(nn.ReLU())
        
        self.view1_layers.append(nn.AdaptiveMaxPool1d(1))
        self.view2_layers.append(nn.AdaptiveMaxPool1d(1))
        
        self.dense = nn.Dense(conv_units[-1]*2, 1, has_bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.dense2 = nn.Dense(1+inp_len3, dense_unit, has_bias=True)
        self.relu = nn.ReLU()
        self.dense3 = nn.Dense(dense_unit, 1, has_bias=True)

    def construct(self, x1, x2, x3):
        for v1_layer, v2_layer in zip(self.view1_layers, self.view2_layers):
            x1 = v1_layer(x1)
            x2 = v2_layer(x2)
        x = ops.cat((x1, x2), axis=1)
        x = ops.squeeze(x, axis=-1)
        x = self.dense(x)
        x = self.sigmoid(x)
        
        x = ops.cat((x, x3), axis=1)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        
        return x