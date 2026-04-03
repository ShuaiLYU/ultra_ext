
def load_labels_from_cache( cache_file):
    from ultralytics.data.utils import load_dataset_cache_file
    cache=load_dataset_cache_file(cache_file)
    labels=cache['labels']
    return labels






def print_first_n_labels_from_cache(cache_file, n=3):
    from ultra_ext.utils import super_print
    labels=load_labels_from_cache(cache_file)
    print(f"First {n} labels from cache {cache_file}:")
    for i in range(min(n, len(labels))):
        super_print(f"Label {i}", labels[i])