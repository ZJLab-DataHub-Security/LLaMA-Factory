import torch

def extract_local(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    if device == None:
        return local_value
    return local_value.to(device)

def extract_local_varlen(value, cu_seqlens, rank, world_size, device, dim=1):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[:,start:end].chunk(2 * world_size, dim=dim)
        local_values.extend(
            [
                local_value[rank].detach().clone(),
                local_value[2 * world_size - 1 - rank].detach().clone(),
            ]
        )

    if device == None:
        return torch.cat(local_values, dim=dim).contiguous()
    return torch.cat(local_values, dim=dim).contiguous().to(device)

def prepare_zigzag_ring_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }

def prepare_zigzag_ring_attn_sft_inputs(
    input_ids, attention_mask, position_ids, labels, rank, world_size, device
):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    local_attention_mask = extract_local(
        attention_mask,
        rank,
        world_size,
        device
    )
    local_labels = extract_local(
        labels,
        rank,
        world_size,
        device,
    )
    return {
        "input_ids": local_input_ids,
        "attention_mask": None,
        "position_ids": local_position_ids,
        "labels": local_labels,
    }

def prepare_zigzag_ring_attn_varlen_sft_inputs(
    input_ids, attention_mask, position_ids, labels, rank, world_size, device, cu_seqlens, cu_seq_lens_q, max_length_q
):
    local_input_ids = extract_local_varlen(
        input_ids,
        cu_seqlens,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local_varlen(
        position_ids,
        cu_seqlens,
        rank,
        world_size,
        device,
    )
    local_labels = extract_local_varlen(
        labels,
        cu_seqlens,
        rank,
        world_size,
        device,
    )
    return {
        "input_ids": local_input_ids,
        "attention_mask": None,
        "position_ids": local_position_ids,
        "labels": local_labels,
        "cu_seq_lens_q": cu_seq_lens_q,
        "max_length_q": max_length_q
    }