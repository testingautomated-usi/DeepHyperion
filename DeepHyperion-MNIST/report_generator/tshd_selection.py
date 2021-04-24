import rasterization_tools, vectorization_tools
from report_generator.samples_extractor import DeepHyperionSample
import keras
import numpy as np

from utils import reshape

mnist = keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def apply_direct_tshd(image, seed, tshd_val):
    seed = reshape(x_test[int(seed)])
    distance_seed = get_distance(seed, image)
    print("DIRECT %s" % distance_seed)
    if distance_seed < tshd_val:
        return True
    else:
        return False


def apply_rast_tshd(image, seed, tshd_val):
    seed_image = x_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    seed = rasterization_tools.rasterize_in_memory(xml_desc)
    distance_seed = get_distance(seed, image)

    print("RAST %s" % distance_seed)

    if distance_seed < tshd_val:
        return True
    else:
        return False


def is_valid_digit(sample, tshd_type, tshd_val):
   if tshd_type == '1':
       is_valid = apply_rast_tshd(sample.image, sample.seed, tshd_val)
   if tshd_type == '2':
       is_valid = apply_direct_tshd(sample.image, sample.seed, tshd_val)

   return is_valid


if __name__ == "__main__":
   sample = DeepHyperionSample("exp_tshd/mbr1")
   val = is_valid_digit(sample, '1', 2.0)
   print(val)
