import pandas as pd
import numpy as np
import torch
from utils import generate_function_from_softmax

# select vae
vae = 5

# load model
decoder_path = "trained_vaes/vae" + str(vae) + "_decoder.pt"
decoder = torch.load(decoder_path)
decoder.eval()

latent_bounds_path = f"trained_vaes/vae{vae}_latent_bounds.csv"
latent_bounds = pd.read_csv(latent_bounds_path)

# simple uniform random sampling that shows vae outputs
rand_sample = 10
for i in range(rand_sample):
    # choose sampling bound not in the outer part of the latent bounds as these are often more obscure
    lower_bounds = latent_bounds['min'].values/5
    upper_bounds = latent_bounds['max'].values/5
    sampled_tensor = np.random.uniform(low=lower_bounds, high=upper_bounds)

    # read inputs
    input_tensor = torch.from_numpy(sampled_tensor).float().unsqueeze(0)
    x_hat, x2_hat = decoder(input_tensor)

    # save softmax prediction as csv
    x_hat_np = x_hat.detach().numpy()
    tf_softmax = pd.DataFrame(x_hat_np[0, :, :])

    # save quantile prediction as csv
    x2_hat_np = x2_hat.detach().numpy()
    tf_quantiles = pd.DataFrame(x2_hat_np)

    vocab_path = f"trained_vaes/vae{vae}_vocabulary.csv"
    # Load vocabulary
    vocab_df = pd.read_csv(vocab_path)

    TF = generate_function_from_softmax(tf_softmax, vocab_df)
    print(f"TF: {TF}")