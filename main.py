import os
import torch
import numpy as np
from Preprocess import Preprocess
import sys


from Decoder import Decoder

MODEL_FOLDER = os.path.abspath(os.path.join(__file__, os.pardir,"MODEL"))
model_file = os.path.join(MODEL_FOLDER, "model.pt")
model = torch.load(model_file)
decoder = Decoder()
pre = Preprocess()


def diacritize(input_text):
    processed_line = pre.prepare_text(input_text)
    predicted_line = model.predict(processed_line)
    predict_argmaxed = np.argmax(predicted_line,axis=2)
    decoded_line = decoder.decode(processed_line, predict_argmaxed)
    return decoded_line


if __name__ == '__main__':
    print(sys.argv)
    diacritized_text = diacritize(sys.argv[1])

    print(diacritized_text)