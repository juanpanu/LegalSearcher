import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import os

root_folder = "LegalSearcher/SentenceEncoder"
embedd_model = f'{root_folder}/model/'
filepath = os.path.abspath(os.path.join('',embedd_model))
embed = hub.load(filepath)

def similarity_value(s1, s2):
    
    embeddings = embed([s1, s2])
    sim = tf.keras.losses.cosine_similarity(
        embeddings[0], embeddings[1], axis=0)

    print(sim.numpy())

    if sim.numpy() <= -0.6:
        msg = f"Cosine Similary is {sim.numpy()}, this indicates high similarity"
    elif sim.numpy() >= 0.6:
        msg = f"Cosine Similary is {sim.numpy()}, this indicates high dissimilarity"
    else:
        msg = f"Cosine Similary is {sim.numpy()}, can't decide much with this value"
    return msg