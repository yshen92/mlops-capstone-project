from deployment import inference

import numpy as np
from sentence_transformers import SentenceTransformer


def test_text_embeddings_length():
    test_text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

    exptected_embedding_len = 768

    sentence_transformer = inference.load_preprocessor()
    embeddings = inference.embed_text(test_text, sentence_transformer)

    assert len(embeddings) == exptected_embedding_len, 'Embedding lengths are not equal'

    print('Embedding length test passed')


def test_text_embeddings():
    test_text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

    inference_transformer = inference.load_preprocessor()
    inference_embeddings = inference.embed_text(test_text, inference_transformer)

    expected_transformer = SentenceTransformer('all-mpnet-base-v2')
    expected_embeddings = expected_transformer.encode(test_text)

    inference_embeddings = np.round(inference_embeddings, decimals=3)
    expected_embeddings = np.round(expected_embeddings, decimals=3)

    assert (
        inference_embeddings == expected_embeddings
    ).all(), 'Embedding values are not equal'

    print('Embedding values test passed')


if __name__ == '__main__':
    test_text_embeddings_length()
    test_text_embeddings()
