"""Data integrity checks for FD001 preparation."""


def validate_engine_split(train_engine_ids, validation_engine_ids):
    overlap = set(train_engine_ids).intersection(validation_engine_ids)
    if overlap:
        raise ValueError(f"Engine leakage detected: {sorted(overlap)}")
    return True


def validate_cycle_order(engine_ids, cycles):
    previous = {}
    for engine_id, cycle in zip(engine_ids, cycles):
        if engine_id in previous and cycle <= previous[engine_id]:
            raise ValueError(f"Cycle order violation for engine {engine_id}.")
        previous[engine_id] = cycle
    return True


def validate_target_alignment(current_cycles, final_cycles, targets):
    if not (len(current_cycles) == len(final_cycles) == len(targets)):
        raise ValueError("Cycle and target arrays must have equal length.")
    expected = [f - c for c, f in zip(current_cycles, final_cycles)]
    if any(abs(a - b) > 1e-9 for a, b in zip(expected, targets)):
        raise ValueError("RUL target alignment check failed.")
    return True
