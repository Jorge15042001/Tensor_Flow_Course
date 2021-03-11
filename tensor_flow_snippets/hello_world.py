"""learning tensor flow """
import tensorflow as tf
if tf.__version__!="2.2.0":
    print(tf.__version__)
    raise ValueError("Upgrate the tensorflow version to 2.2.0")
