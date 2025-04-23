from model_compiler.utils import OperationType, TensorId, TensorWithSize, Function, SubFunction, Model, CompiledModel


def create_glu_ffn_model(hidden_dim: int, ffn_dim: int, layer_idx: int = 1) -> Model:
    """
    Create a model for a GLU-based FFN layer
    
    Args:
        hidden_dim: Model dimension (hidden state size)
        ffn_dim: FFN dimension (intermediate size)
        layer_idx: Decoder layer index
        
    Returns:
        Model object representing the FFN layer
    """
    model = Model()
    
    
    # 1. Up projection (k=1, m=1)
    up_proj = Function(op_type=OperationType.MVM, name="up_proj", k=1, m=1, n=layer_idx)
    up_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(up_proj)
    
    # 2. Gate projection (k=2, m=1)
    gate_proj = Function(op_type=OperationType.MVM, name="gate_proj", k=2, m=1, n=layer_idx)
    gate_proj.set_shape((hidden_dim, ffn_dim))
    model.add_function(gate_proj)
    
    # 3. Activation for gate_proj (k=2, m=2)
    activation = Function(op_type=OperationType.ACTIVATION, name="activation", k=2, m=2, n=layer_idx)
    activation.set_shape((1, ffn_dim))
    model.add_function(activation)
    
    # 4. GLU operation (k=1, m=3)
    glu = Function(op_type=OperationType.GLU, name="glu", k=1, m=3, n=layer_idx)
    glu.set_shape((1, ffn_dim))
    model.add_function(glu)
    
    # 5. Down projection (k=1, m=4)
    down_proj = Function(op_type=OperationType.MVM, name="down_proj", k=1, m=4, n=layer_idx)
    down_proj.set_shape((ffn_dim, hidden_dim))
    model.add_function(down_proj)

    model.model_dimension = hidden_dim
    model.ffn_dimension = ffn_dim
    
    return model
