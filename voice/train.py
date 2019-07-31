import numpy as np
import get_voice_data
import get_triplets_for_megan as triplets
import autoencoder
from mynn.optimizers.adam import Adam
from mygrad.nnet import margin_ranking_loss
import mygrad as mg
from noggin import create_plot

database = get_voice_data.grab_voice_data()

sample_duration = 4  # seconds
sampling_rate = 44100  # Hz
sample_length = sample_duration * sampling_rate

model = autoencoder.Autoencoder(sample_length, 100)
optim = Adam(model.parameters)

batch_size = 15

plotter, fig, ax = create_plot(metrics=["loss", "accuracy"])

if __name__ == "__main__":
    for epoch_cnt in range(75):

        batch = np.zeros((batch_size, 3, sample_length))
        for i in range(batch_size):
            batch[i] = triplets.get_triplet_for_megan(database)

        good1 = batch[:, 0, :]
        good2 = batch[:, 1, :]
        bad = batch[:, 2, :]

        embed1 = model(good1)
        embed2 = model(good2)
        bad_embed = model(bad)

        embed1 /= mg.sqrt(mg.sum(mg.square(embed1)))
        embed2 /= mg.sqrt(mg.sum(mg.square(embed2)))
        bad_embed /= mg.sqrt(mg.sum(mg.square(bad_embed)))

        good_sim = (embed2 * embed1).sum(axis=1)
        bad_sim = (bad_embed * embed1).sum(axis=1)

        loss = margin_ranking_loss(good_sim, bad_sim, 1, 0.1)
        acc = np.mean(good_sim > bad_sim)

        loss.backward()
        optim.step()
        loss.null_gradients()

        # """
        plotter.set_train_batch({"loss": loss.item(),
                                 "accuracy": acc},
                                batch_size=1)
        # """
