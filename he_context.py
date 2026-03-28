import tenseal as ts

def create_context():
    """
    Creates a TenSEAL context for CKKS.
    We need enough multiplicative depth for:
      1. Matmul (Conv layer equivalent)
      2. Square (Activation)
      3. Matmul (FC layer)
    This implies at least 3 levels.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 26, 26, 26, 40]
    )
    context.global_scale = 2 ** 26
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.auto_relin = True
    context.auto_rescale = True
    context.auto_mod_switch = True
    return context
