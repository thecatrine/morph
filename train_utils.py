def scheduler_function(step, warmup_frac):
        single_gpu_batch = loader_batch_size * ACCUMULATION
        single_gpu_examples = EPOCHS * N / PARALLELISM
        total_steps = math.ceil(single_gpu_examples / single_gpu_batch)

        frac = step / total_steps

        if frac < warmup_frac:
            return frac / warmup_frac
        else:
            return 1.0 - ((frac - warmup_frac) / (1.0 - warmup_frac))