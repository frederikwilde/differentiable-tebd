import os
import jax


try:
    type = os.environ['TEBD_COMPLEX_TYPE']
    if type == 'complex64':
        COMPLEX_TYPE = jax.numpy.complex64
    elif type == 'complex128':
        jax.config.update("jax_enable_x64", True)
        COMPLEX_TYPE = jax.numpy.complex128
    else:
        raise EnvironmentError(f"Environment variable can only be 'complex64' or 'complex128', was {type}.")
except KeyError:
    jax.config.update("jax_enable_x64", True)
    COMPLEX_TYPE = jax.numpy.complex128
