def _prepare_weights(self, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

    # Initialize a dictionary to count occurrences for each label component
    value_dicts = [{x: 0 for x in range(max_target)} for _ in range(4)]
    labels = self.data[:, -4:]  # Assuming last 4 columns are the labels for each sample

    # Count occurrences for each component in the labels
    for label_set in labels:
        for i, label in enumerate(label_set):
            value_dicts[i][min(max_target - 1, int(label))] += 1

    # Apply reweighting strategies to each label component
    for i in range(4):
        if reweight == 'sqrt_inv':
            value_dicts[i] = {k: np.sqrt(v) for k, v in value_dicts[i].items()}
        elif reweight == 'inverse':
            value_dicts[i] = {k: np.clip(v, 5, 1000) for k, v in value_dicts[i].items()}  # clip weights for inverse re-weight

    # Generate weights for each label component
    num_per_label = [
        [value_dicts[i][min(max_target - 1, int(label))] for i, label in enumerate(label_set)]
        for label_set in labels
    ]

    # If reweighting is 'none' or there are no labels, return None
    if not any(len(label_counts) for label_counts in num_per_label) or reweight == 'none':
        return None

    print(f"Using re-weighting: [{reweight.upper()}]")

    # If LDS is enabled, apply smoothing to each label component
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_values = [
            convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant'
            ) for value_dict in value_dicts
        ]
        num_per_label = [
            [smoothed_values[i][min(max_target - 1, int(label))] for i, label in enumerate(label_set)]
            for label_set in labels
        ]

    # Calculate the final weights: average or combine weights across the 4 label components
    weights = [
        np.float32(1 / max(np.mean(label_counts), 1e-6)) for label_counts in num_per_label  # Avoid div by zero
    ]

    # Normalize weights to keep the scale consistent
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * w for w in weights]

    return weights
