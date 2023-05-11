from datasets import load_dataset

# Rewrite this to return your actual dataset and do any prep work required
def get_datasets():
    dataset = load_dataset("mnist")

    return dataset.with_format(type='torch')